"""显式、默认 dry-run 的旧 Pagent 数据导入。"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .storage.migrations import LATEST_SCHEMA_VERSION
from .store import Store


class ImportPagentError(RuntimeError):
    """旧 Pagent 数据无法在 fail-closed 条件下导入。"""


@dataclass(frozen=True)
class ImportFile:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ImportPlan:
    source_dir: Path
    target_dir: Path
    source_schema_version: int
    papers: int
    chunks: int
    source_database_fingerprint: str
    files: tuple[ImportFile, ...]
    total_bytes: int
    path_rewrites: tuple[tuple[str, str], ...]
    external_paper_paths: tuple[str, ...]

    @property
    def required_bytes(self) -> int:
        """staging DB、迁移前备份和文件复制所需的保守空间估算。"""

        return self.total_bytes * 3 + 16 * 1024 * 1024


@dataclass(frozen=True)
class ImportResult:
    plan: ImportPlan
    executed: bool
    target_schema_version: Optional[int] = None


def plan_import_pagent(
    source_dir: str | Path,
    target_dir: str | Path,
) -> ImportPlan:
    """只读验证旧库并返回导入计划，不创建目标或 staging。"""

    source = _resolve_source_dir(source_dir)
    target = _resolve_target_dir(target_dir)
    _validate_separate_roots(source, target)
    _fail_if_target_exists(target)

    source_db = source / "library.db"
    if not source_db.is_file() or source_db.is_symlink():
        raise ImportPagentError(f"旧数据目录缺少普通文件 library.db：{source_db}")

    try:
        files = _build_file_manifest(source)
        inspection = _inspect_legacy_database(source_db)
    except ImportPagentError:
        raise
    except OSError as exc:
        raise ImportPagentError(f"无法读取旧数据目录：{exc}") from exc
    file_by_path = {item.relative_path: item for item in files}
    rewrites: list[tuple[str, str]] = []
    external_paths: list[str] = []
    for raw_path, expected_sha256 in inspection.paper_paths:
        paper_path = _validate_indexed_pdf(raw_path, expected_sha256)
        if _is_within(paper_path, source):
            relative = paper_path.relative_to(source)
            relative_text = relative.as_posix()
            if relative_text not in file_by_path:
                raise ImportPagentError(
                    f"索引论文未进入复制清单：{relative_text}"
                )
            final_path = target / relative
            rewrites.append((raw_path, str(final_path)))
        else:
            external_paths.append(str(paper_path))

    if inspection.library_dir:
        library_dir = _resolve_existing_directory(inspection.library_dir, "旧论文目录")
        if _is_within(library_dir, source):
            relative = library_dir.relative_to(source)
            rewrites.append((inspection.library_dir, str(target / relative)))

    database_bytes = source_db.stat().st_size
    wal_path = source_db.with_name(source_db.name + "-wal")
    if wal_path.is_file():
        database_bytes += wal_path.stat().st_size
    total_bytes = database_bytes + sum(item.size for item in files)
    return ImportPlan(
        source_dir=source,
        target_dir=target,
        source_schema_version=inspection.schema_version,
        papers=inspection.papers,
        chunks=inspection.chunks,
        source_database_fingerprint=inspection.logical_fingerprint,
        files=files,
        total_bytes=total_bytes,
        path_rewrites=tuple(rewrites),
        external_paper_paths=tuple(sorted(external_paths)),
    )


def import_pagent_data(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    execute: bool = False,
) -> ImportResult:
    """执行或 dry-run 旧库导入；默认 ``execute=False``。"""

    plan = plan_import_pagent(source_dir, target_dir)
    if not execute:
        return ImportResult(plan=plan, executed=False)

    target = plan.target_dir
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _ensure_free_space(parent, plan.required_bytes)
    lock_path = parent / f".{target.name}.import.lock"
    lock_fd: Optional[int] = None
    staging: Optional[Path] = None
    try:
        try:
            lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise ImportPagentError(
                f"检测到另一个导入或遗留锁：{lock_path}；请确认无导入运行后再处理"
            ) from exc
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        _fail_if_target_exists(target)

        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.import-", dir=parent)
        )
        _copy_manifest(plan, staging)
        staged_db = staging / "library.db"
        _backup_legacy_database(plan.source_dir / "library.db", staged_db)
        _verify_backup_matches_plan(staged_db, plan)
        _rewrite_internal_paths(staged_db, plan)
        _migrate_and_validate_staging(staged_db, plan)
        _verify_copied_files(plan, staging)
        _fsync_tree(staging)

        # 锁只协调 PRAgent import；最后一刻仍必须拒绝外部创建的目标。
        _fail_if_target_exists(target)
        staging_path = staging
        staging.rename(target)
        try:
            _fsync_directory(parent)
        except Exception as sync_exc:
            # 落位后的目录同步失败时，在仍持有导入锁的情况下撤回目标。
            try:
                target.rename(staging_path)
            except Exception as rollback_exc:
                staging = None
                raise ImportPagentError(
                    f"目标已落位到 {target}，但目录同步及自动撤回均失败；"
                    "请保留目录并人工检查"
                ) from rollback_exc
            staging = staging_path
            raise sync_exc
        else:
            staging = None
    except ImportPagentError:
        raise
    except Exception as exc:
        raise ImportPagentError(f"导入失败，目标未落位：{exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
        lock_path.unlink(missing_ok=True)

    return ImportResult(
        plan=plan,
        executed=True,
        target_schema_version=LATEST_SCHEMA_VERSION,
    )


@dataclass(frozen=True)
class _LegacyInspection:
    schema_version: int
    papers: int
    chunks: int
    library_dir: Optional[str]
    paper_paths: tuple[tuple[str, str], ...]
    logical_fingerprint: str


_REQUIRED_V1_COLUMNS = {
    "papers": {
        "id",
        "path",
        "sha256",
        "title",
        "authors",
        "year",
        "page_count",
        "has_text",
        "indexed_at",
    },
    "chunks": {"id", "paper_id", "seq", "page", "text", "embedding"},
    "meta": {"key", "value"},
}
_REQUIRED_V2_COLUMNS = {
    "evidence": {
        "id",
        "paper_id",
        "chunk_id",
        "source_hash",
        "paper_sha256",
        "chunk_text_sha256",
        "title",
        "authors",
        "year",
        "path",
        "page",
        "chunk_seq",
        "text",
        "annotation",
        "pinned_at",
    },
    "agent_runs": {
        "id",
        "objective",
        "status",
        "plan",
        "budget",
        "error",
        "created_at",
        "updated_at",
    },
    "agent_events": {
        "id",
        "run_id",
        "seq",
        "event_type",
        "payload",
        "created_at",
    },
}


def _inspect_legacy_database(db_path: Path) -> _LegacyInspection:
    connection = _open_read_only(db_path)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise ImportPagentError("旧数据库未通过 SQLite quick_check")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "meta" not in tables:
            raise ImportPagentError("旧数据库缺少 meta/schema_version")
        version_row = connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if version_row is None:
            raise ImportPagentError("旧数据库缺少 schema_version")
        raw_version = version_row[0]
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise ImportPagentError(
                f"旧数据库 schema_version 无效：{raw_version!r}"
            ) from exc
        if str(version) != str(raw_version).strip() or version not in {1, 2}:
            raise ImportPagentError(
                f"只支持 Pagent schema v1/v2，当前为 {raw_version!r}"
            )
        required = dict(_REQUIRED_V1_COLUMNS)
        if version == 2:
            required.update(_REQUIRED_V2_COLUMNS)
        _validate_columns(connection, required)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ImportPagentError(
                f"旧数据库 foreign_key_check 失败（{len(violations)} 项）"
            )
        papers = int(connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        library_row = connection.execute(
            "SELECT value FROM meta WHERE key='library_dir'"
        ).fetchone()
        paper_paths = tuple(
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT path, sha256 FROM papers ORDER BY id"
            ).fetchall()
        )
        return _LegacyInspection(
            schema_version=version,
            papers=papers,
            chunks=chunks,
            library_dir=str(library_row[0]) if library_row and library_row[0] else None,
            paper_paths=paper_paths,
            logical_fingerprint=_database_logical_fingerprint(connection),
        )
    except ImportPagentError:
        raise
    except sqlite3.Error as exc:
        raise ImportPagentError(f"旧数据库读取失败：{exc}") from exc
    finally:
        connection.close()


def _validate_columns(
    connection: sqlite3.Connection,
    required: dict[str, set[str]],
) -> None:
    for table, required_columns in required.items():
        columns = {
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        }
        if not columns:
            raise ImportPagentError(f"旧数据库缺少表：{table}")
        missing = sorted(required_columns - columns)
        if missing:
            raise ImportPagentError(
                f"旧数据库表 {table} 缺少列：{', '.join(missing)}"
            )


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection
    except sqlite3.Error as exc:
        raise ImportPagentError(f"无法只读打开旧数据库：{exc}") from exc


def _backup_legacy_database(source_db: Path, destination_db: Path) -> None:
    source = _open_read_only(source_db)
    destination = sqlite3.connect(destination_db)
    try:
        source.backup(destination)
        destination.commit()
    except Exception:
        destination.close()
        source.close()
        destination_db.unlink(missing_ok=True)
        raise
    else:
        destination.close()
        source.close()


def _verify_backup_matches_plan(db_path: Path, plan: ImportPlan) -> None:
    inspection = _inspect_legacy_database(db_path)
    if (
        inspection.schema_version != plan.source_schema_version
        or inspection.papers != plan.papers
        or inspection.chunks != plan.chunks
        or inspection.logical_fingerprint != plan.source_database_fingerprint
    ):
        raise ImportPagentError("SQLite backup 与 dry-run 计划不一致；旧库可能正在变化")


def _rewrite_internal_paths(db_path: Path, plan: ImportPlan) -> None:
    path_map = dict(plan.path_rewrites)
    if not path_map:
        return
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        for old_path, new_path in path_map.items():
            connection.execute(
                "UPDATE papers SET path=? WHERE path=?", (new_path, old_path)
            )
            if _table_exists(connection, "evidence"):
                connection.execute(
                    "UPDATE evidence SET path=? WHERE path=?", (new_path, old_path)
                )
        library_dir = connection.execute(
            "SELECT value FROM meta WHERE key='library_dir'"
        ).fetchone()
        if library_dir and library_dir[0] in path_map:
            connection.execute(
                "UPDATE meta SET value=? WHERE key='library_dir'",
                (path_map[library_dir[0]],),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _migrate_and_validate_staging(db_path: Path, plan: ImportPlan) -> None:
    store = Store(db_path)
    try:
        if store.stats() != (plan.papers, plan.chunks):
            raise ImportPagentError("迁移后论文/分块计数与旧库不一致")
        if store.meta_get("schema_version") != str(LATEST_SCHEMA_VERSION):
            raise ImportPagentError("staging 数据库未迁移到当前 schema")
    finally:
        store.close()

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ImportPagentError("迁移后 foreign_key_check 失败")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise ImportPagentError("迁移后 SQLite quick_check 失败")
        paths = {
            str(row[0])
            for row in connection.execute("SELECT path FROM papers").fetchall()
        }
        for old_path, new_path in plan.path_rewrites:
            is_paper_path = Path(old_path).suffix.lower() == ".pdf"
            if old_path in paths or (is_paper_path and new_path not in paths):
                raise ImportPagentError("迁移后的 indexed paper 路径重写不完整")
    finally:
        connection.close()


def _database_logical_fingerprint(connection: sqlite3.Connection) -> str:
    """对全部旧应用表内容做确定性摘要，用于发现 plan/backup 间变化。"""

    digest = hashlib.sha256()
    tables = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    ]
    for table in tables:
        digest.update(table.encode("utf-8"))
        digest.update(b"\0")
        rows = connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
        for row in rows:
            for value in row:
                if value is None:
                    payload = b"n"
                elif isinstance(value, bytes):
                    payload = b"b" + value
                else:
                    payload = b"t" + str(value).encode("utf-8")
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
    return digest.hexdigest()


def _build_file_manifest(source: Path) -> tuple[ImportFile, ...]:
    files: list[ImportFile] = []
    for root, directories, names in os.walk(source, topdown=True, followlinks=False):
        root_path = Path(root)
        for directory in list(directories):
            candidate = root_path / directory
            if candidate.is_symlink():
                raise ImportPagentError(f"旧数据目录包含符号链接目录：{candidate}")
        for name in names:
            path = root_path / name
            relative = path.relative_to(source)
            if relative.parent == Path(".") and name.startswith("library.db"):
                continue
            if path.is_symlink():
                raise ImportPagentError(f"旧数据目录包含符号链接文件：{path}")
            if not path.is_file():
                raise ImportPagentError(f"旧数据目录包含非普通文件：{path}")
            stat = path.stat()
            files.append(
                ImportFile(
                    relative_path=relative.as_posix(),
                    size=int(stat.st_size),
                    sha256=_sha256_file(path),
                )
            )
    files.sort(key=lambda item: item.relative_path)
    return tuple(files)


def _copy_manifest(plan: ImportPlan, staging: Path) -> None:
    for item in plan.files:
        relative = Path(item.relative_path)
        source = plan.source_dir / relative
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            source_fd = os.open(source, flags)
        except OSError as exc:
            raise ImportPagentError(f"无法安全读取旧文件 {item.relative_path}：{exc}") from exc
        try:
            with os.fdopen(source_fd, "rb") as source_stream, open(
                destination, "xb"
            ) as destination_stream:
                while True:
                    block = source_stream.read(1 << 20)
                    if not block:
                        break
                    digest.update(block)
                    destination_stream.write(block)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        if destination.stat().st_size != item.size or digest.hexdigest() != item.sha256:
            raise ImportPagentError(
                f"复制期间旧文件发生变化：{item.relative_path}"
            )


def _verify_copied_files(plan: ImportPlan, staging: Path) -> None:
    for item in plan.files:
        source = plan.source_dir / item.relative_path
        destination = staging / item.relative_path
        if (
            not destination.is_file()
            or destination.stat().st_size != item.size
            or _sha256_file(destination) != item.sha256
            or _sha256_file(source) != item.sha256
        ):
            raise ImportPagentError(f"文件复制校验失败：{item.relative_path}")


def _resolve_source_dir(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ImportPagentError(f"旧数据目录不能是符号链接：{candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImportPagentError(f"旧数据目录不存在或无法访问：{candidate}") from exc
    if not resolved.is_dir():
        raise ImportPagentError(f"旧数据路径不是目录：{resolved}")
    return resolved


def _resolve_target_dir(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.name:
        raise ImportPagentError("目标数据目录无效")
    return candidate.resolve(strict=False)


def _validate_separate_roots(source: Path, target: Path) -> None:
    if source == target or _is_within(target, source) or _is_within(source, target):
        raise ImportPagentError("旧数据目录与目标目录必须相互独立")


def _fail_if_target_exists(target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise ImportPagentError(f"目标数据目录已存在，拒绝覆盖：{target}")


def _validate_indexed_pdf(raw_path: str, expected_sha256: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        raise ImportPagentError(f"旧索引包含非绝对论文路径：{raw_path}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImportPagentError(f"旧索引论文不存在：{raw_path}") from exc
    if not resolved.is_file() or resolved.suffix.lower() != ".pdf":
        raise ImportPagentError(f"旧索引路径不是 PDF 普通文件：{raw_path}")
    try:
        actual_sha256 = _sha256_file(resolved)
    except OSError as exc:
        raise ImportPagentError(f"无法读取旧索引论文：{raw_path}") from exc
    if actual_sha256 != expected_sha256:
        raise ImportPagentError(f"旧索引论文内容哈希已变化：{raw_path}")
    return resolved


def _resolve_existing_directory(raw_path: str, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        raise ImportPagentError(f"{label}不是绝对路径：{raw_path}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImportPagentError(f"{label}不存在：{raw_path}") from exc
    if not resolved.is_dir():
        raise ImportPagentError(f"{label}不是目录：{raw_path}")
    return resolved


def _ensure_free_space(parent: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(parent).free
    if free < required_bytes:
        raise ImportPagentError(
            f"目标磁盘空间不足：需要约 {required_bytes} 字节，可用 {free} 字节"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise ImportPagentError(f"无法同步 staging 文件：{path.name}") from exc
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ImportPagentError(f"无法同步目录：{path}") from exc
