import hashlib
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pragent.cli import app
from pragent.import_pagent import ImportPagentError, import_pagent_data, plan_import_pagent
from pragent.storage.migrations import LATEST_SCHEMA_VERSION
from pragent.store import Store


_LEGACY_SCHEMA = """
CREATE TABLE papers (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    year INTEGER,
    page_count INTEGER NOT NULL,
    has_text INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    UNIQUE(paper_id, seq)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE SET NULL,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    source_hash TEXT NOT NULL,
    paper_sha256 TEXT NOT NULL,
    chunk_text_sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    year INTEGER,
    path TEXT NOT NULL,
    page INTEGER NOT NULL,
    chunk_seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    annotation TEXT NOT NULL DEFAULT '',
    pinned_at TEXT NOT NULL
);
CREATE TABLE agent_runs (
    id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    plan TEXT,
    budget TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, seq)
);
"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_legacy_source(root: Path, *, version: int = 2):
    root.mkdir()
    papers_dir = root / "papers"
    papers_dir.mkdir()
    notes_dir = root / "notes"
    notes_dir.mkdir()
    (notes_dir / "reading.md").write_text("# 旧笔记\n", encoding="utf-8")
    pdf_bytes = b"%PDF-1.4\nlegacy-one\n%%EOF\n"
    pdf = papers_dir / "one.pdf"
    pdf.write_bytes(pdf_bytes)

    connection = sqlite3.connect(root / "library.db")
    connection.execute("PRAGMA foreign_keys=ON")
    if version == 1:
        connection.executescript(
            _LEGACY_SCHEMA.split("CREATE TABLE evidence", 1)[0]
        )
    else:
        connection.executescript(_LEGACY_SCHEMA)
    connection.execute(
        """
        INSERT INTO papers VALUES(1, ?, ?, '旧论文', '["旧作者"]', 2020, 1, 1, ?)
        """,
        (str(pdf.resolve()), _digest(pdf_bytes), "2026-01-01T00:00:00"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES(1, 1, 0, 1, '旧正文', NULL)"
    )
    connection.executemany(
        "INSERT INTO meta(key, value) VALUES(?, ?)",
        [
            ("schema_version", str(version)),
            ("index_revision", "9"),
            ("library_dir", str(papers_dir.resolve())),
            ("embed_model", "legacy-embed"),
        ],
    )
    connection.commit()
    return connection, pdf


def _schema_version(db_path: Path) -> str:
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
    finally:
        connection.close()


def test_import_is_dry_run_by_default_and_does_not_create_target(tmp_path):
    source = tmp_path / "old"
    connection, _ = _create_legacy_source(source)
    connection.close()
    target = tmp_path / "new"
    source_db_hash = _digest((source / "library.db").read_bytes())

    result = import_pagent_data(source, target)

    assert result.executed is False
    assert result.target_schema_version is None
    assert result.plan.source_schema_version == 2
    assert (result.plan.papers, result.plan.chunks) == (1, 1)
    assert {item.relative_path for item in result.plan.files} == {
        "notes/reading.md",
        "papers/one.pdf",
    }
    assert len(result.plan.path_rewrites) == 2
    assert not target.exists()
    assert list(tmp_path.glob(".new.import-*")) == []
    assert _digest((source / "library.db").read_bytes()) == source_db_hash
    assert _schema_version(source / "library.db") == "2"


def test_schema_v1_import_is_supported(tmp_path):
    source = tmp_path / "v1-old"
    connection, _ = _create_legacy_source(source, version=1)
    connection.close()
    target = tmp_path / "v1-new"

    result = import_pagent_data(source, target, execute=True)

    assert result.plan.source_schema_version == 1
    imported = Store(target / "library.db")
    assert imported.stats() == (1, 1)
    assert imported.meta_get("schema_version") == str(LATEST_SCHEMA_VERSION)
    imported.close()
    assert _schema_version(source / "library.db") == "1"


def test_execute_import_migrates_staging_rebases_internal_paths_and_preserves_source(
    tmp_path,
):
    source = tmp_path / "old"
    source_connection, old_pdf = _create_legacy_source(source)
    source_connection.close()
    target = tmp_path / "new"

    result = import_pagent_data(source, target, execute=True)

    assert result.executed is True
    assert result.target_schema_version == LATEST_SCHEMA_VERSION
    assert target.is_dir()
    assert (target / "notes" / "reading.md").read_text(encoding="utf-8") == "# 旧笔记\n"
    assert (target / "papers" / "one.pdf").read_bytes() == old_pdf.read_bytes()
    assert _schema_version(source / "library.db") == "2"
    source_check = sqlite3.connect(source / "library.db")
    assert source_check.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_projects'"
    ).fetchone() is None
    source_check.close()
    assert old_pdf.is_file()

    imported = Store(target / "library.db")
    assert imported.stats() == (1, 1)
    paper = imported.paper_by_id(1)
    assert paper.path == str((target / "papers" / "one.pdf").resolve())
    assert imported.meta_get("library_dir") == str((target / "papers").resolve())
    assert imported.meta_get("index_revision") == "9"
    imported.close()
    assert list(target.glob("library.db.pre-migration-v2-to-v*.bak"))
    assert list(tmp_path.glob(".new.import-*")) == []


def test_import_target_conflict_and_failure_leave_source_and_target_untouched(
    tmp_path, monkeypatch
):
    source = tmp_path / "old"
    connection, _ = _create_legacy_source(source)
    connection.close()
    target = tmp_path / "new"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ImportPagentError, match="目标数据目录已存在"):
        import_pagent_data(source, target, execute=True)
    assert marker.read_text(encoding="utf-8") == "keep"

    marker.unlink()
    target.rmdir()

    def fail_migration(*args, **kwargs):
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(
        "pragent.import_pagent._migrate_and_validate_staging", fail_migration
    )
    with pytest.raises(ImportPagentError, match="目标未落位"):
        import_pagent_data(source, target, execute=True)
    assert not target.exists()
    assert list(tmp_path.glob(".new.import-*")) == []
    assert list(tmp_path.glob(".new.import.lock")) == []
    assert _schema_version(source / "library.db") == "2"


def test_source_database_change_between_plan_and_backup_fails_without_target(
    tmp_path, monkeypatch
):
    source = tmp_path / "changing-old"
    connection, _ = _create_legacy_source(source)
    connection.close()
    target = tmp_path / "changing-new"

    from pragent import import_pagent as importer

    original_copy = importer._copy_manifest

    def copy_then_change(plan, staging):
        original_copy(plan, staging)
        changed = sqlite3.connect(source / "library.db")
        changed.execute("UPDATE papers SET title='并发变化' WHERE id=1")
        changed.commit()
        changed.close()

    monkeypatch.setattr(importer, "_copy_manifest", copy_then_change)
    with pytest.raises(ImportPagentError, match="旧库可能正在变化"):
        import_pagent_data(source, target, execute=True)
    assert not target.exists()
    assert list(tmp_path.glob(".changing-new.import-*")) == []


def test_import_rejects_future_schema_missing_files_and_symlinks(tmp_path):
    source = tmp_path / "future"
    connection, pdf = _create_legacy_source(source)
    connection.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(ImportPagentError, match="只支持"):
        plan_import_pagent(source, tmp_path / "future-target")

    missing = tmp_path / "missing"
    connection, pdf = _create_legacy_source(missing)
    connection.close()
    pdf.unlink()
    with pytest.raises(ImportPagentError, match="不存在"):
        plan_import_pagent(missing, tmp_path / "missing-target")

def test_import_rejects_symlinks_when_supported(tmp_path):
    unsafe = tmp_path / "unsafe"
    connection, _ = _create_legacy_source(unsafe)
    connection.close()
    try:
        (unsafe / "linked-note").symlink_to(unsafe / "notes" / "reading.md")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("当前 Windows 账户没有创建符号链接的权限")
        raise
    with pytest.raises(ImportPagentError, match="符号链接"):
        plan_import_pagent(unsafe, tmp_path / "unsafe-target")


def test_import_uses_online_backup_and_includes_committed_wal_rows(tmp_path):
    source = tmp_path / "wal-old"
    writer, _ = _create_legacy_source(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    pdf_bytes = b"%PDF-1.4\nlegacy-two\n%%EOF\n"
    second_pdf = source / "papers" / "two.pdf"
    second_pdf.write_bytes(pdf_bytes)
    writer.execute(
        """
        INSERT INTO papers VALUES(2, ?, ?, 'WAL 论文', '["作者"]', 2021, 1, 1, ?)
        """,
        (str(second_pdf.resolve()), _digest(pdf_bytes), "2026-01-01T00:00:00"),
    )
    writer.execute("INSERT INTO chunks VALUES(2, 2, 0, 1, 'WAL 正文', NULL)")
    writer.commit()
    assert (source / "library.db-wal").is_file()

    target = tmp_path / "wal-new"
    result = import_pagent_data(source, target, execute=True)
    writer.close()

    assert result.plan.papers == 2 and result.plan.chunks == 2
    imported = Store(target / "library.db")
    assert imported.stats() == (2, 2)
    assert imported.paper_by_id(2).title == "WAL 论文"
    imported.close()


def test_import_pagent_cli_defaults_to_dry_run_then_executes(tmp_path):
    source = tmp_path / "cli-old"
    connection, _ = _create_legacy_source(source)
    connection.close()
    target = tmp_path / "cli-new"
    runner = CliRunner()

    dry_run = runner.invoke(
        app,
        ["import-pagent", "--source", str(source), "--target", str(target)],
    )
    assert dry_run.exit_code == 0
    assert "Dry-run 完成" in dry_run.stdout
    assert not target.exists()

    executed = runner.invoke(
        app,
        [
            "import-pagent",
            "--source",
            str(source),
            "--target",
            str(target),
            "--execute",
        ],
    )
    assert executed.exit_code == 0
    assert f"目标 schema v{LATEST_SCHEMA_VERSION}" in executed.stdout
    assert target.is_dir()
