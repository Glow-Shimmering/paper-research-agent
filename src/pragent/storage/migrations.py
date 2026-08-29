"""有序、可审计的 SQLite schema migrations。

每个数据库版本只允许由本模块中的一个 migration 产生。所有待执行的
migration 位于同一个 ``BEGIN IMMEDIATE`` 事务中：任一步失败都会回滚到
打开数据库前的 schema version。已有的磁盘数据库在迁移前还会通过 SQLite
backup API 留下一份一致备份，作为事务回滚之外的恢复边界。
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class SchemaMigrationError(RuntimeError):
    """Schema migration 无法安全完成。"""


class InvalidSchemaVersionError(SchemaMigrationError):
    """数据库声明了无效或与实际表结构不一致的版本。"""


class FutureSchemaVersionError(SchemaMigrationError):
    """数据库来自比当前程序更新的 schema，必须 fail closed。"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        payload = "\n-- statement --\n".join((self.name, *self.statements))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MigrationReport:
    previous_version: int
    current_version: int
    applied_versions: tuple[int, ...]
    backup_path: Optional[Path] = None


_MIGRATION_1 = Migration(
    version=1,
    name="index_core",
    statements=(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            sha256 TEXT NOT NULL,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            year INTEGER,
            page_count INTEGER NOT NULL,
            has_text INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            UNIQUE(paper_id, seq)
        )
        """,
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
    ),
)

_MIGRATION_2 = Migration(
    version=2,
    name="evidence_and_agent_audit",
    statements=(
        """
        CREATE TABLE IF NOT EXISTS evidence (
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
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_evidence_pinned_at ON evidence(pinned_at DESC, id)",
        "CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(path, chunk_seq)",
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            objective TEXT NOT NULL,
            status TEXT NOT NULL,
            plan TEXT,
            budget TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, seq)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_agent_events_run_seq ON agent_events(run_id, seq)",
    ),
)

_MIGRATION_3 = Migration(
    version=3,
    name="research_projects_and_sources",
    statements=(
        "ALTER TABLE papers ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'pdf' CHECK(source_kind IN ('pdf', 'web'))",
        "ALTER TABLE papers ADD COLUMN canonical_uri TEXT",
        "ALTER TABLE papers ADD COLUMN locator TEXT NOT NULL DEFAULT '{}'",
        """
        CREATE TABLE research_projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL CHECK(length(trim(title)) > 0),
            description TEXT NOT NULL DEFAULT '',
            default_language TEXT NOT NULL DEFAULT 'zh-CN',
            citation_style TEXT NOT NULL DEFAULT 'gb-t-7714-2015-numeric',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'archived')),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_research_projects_updated ON research_projects(updated_at DESC, id)",
        """
        CREATE TABLE research_questions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
            question TEXT NOT NULL CHECK(length(trim(question)) > 0),
            position INTEGER NOT NULL DEFAULT 0 CHECK(position >= 0),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_research_questions_project ON research_questions(project_id, position, id)",
        """
        CREATE TABLE research_sources (
            id TEXT PRIMARY KEY,
            canonical_key TEXT NOT NULL UNIQUE CHECK(length(trim(canonical_key)) > 0),
            source_kind TEXT NOT NULL CHECK(source_kind IN ('paper', 'web')),
            title TEXT NOT NULL DEFAULT '',
            authors TEXT NOT NULL DEFAULT '[]',
            year INTEGER,
            doi TEXT,
            arxiv_id TEXT,
            canonical_url TEXT,
            content_sha256 TEXT,
            indexed_paper_id INTEGER UNIQUE REFERENCES papers(id) ON DELETE SET NULL,
            status TEXT NOT NULL DEFAULT 'discovered'
                CHECK(status IN ('discovered', 'fetching', 'ready', 'failed', 'archived')),
            metadata TEXT NOT NULL DEFAULT '{}',
            locator TEXT NOT NULL DEFAULT '{}',
            snapshot_path TEXT,
            snapshot_sha256 TEXT,
            extracted_text TEXT,
            fetched_at TEXT,
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE UNIQUE INDEX idx_research_sources_doi ON research_sources(doi) WHERE doi IS NOT NULL",
        "CREATE UNIQUE INDEX idx_research_sources_arxiv ON research_sources(arxiv_id) WHERE arxiv_id IS NOT NULL",
        "CREATE INDEX idx_research_sources_updated ON research_sources(updated_at DESC, id)",
        "CREATE INDEX idx_research_sources_status ON research_sources(status, source_kind, id)",
        """
        CREATE TABLE source_identities (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
            identity_kind TEXT NOT NULL
                CHECK(identity_kind IN ('doi', 'arxiv', 'url', 'content_sha256')),
            normalized_value TEXT NOT NULL CHECK(length(trim(normalized_value)) > 0),
            is_primary INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0, 1)),
            created_at TEXT NOT NULL,
            UNIQUE(identity_kind, normalized_value)
        )
        """,
        "CREATE INDEX idx_source_identities_source ON source_identities(source_id, identity_kind, id)",
        "CREATE UNIQUE INDEX idx_source_identities_primary ON source_identities(source_id) WHERE is_primary=1",
        """
        CREATE TABLE source_records (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
            provider TEXT NOT NULL CHECK(length(trim(provider)) > 0),
            provider_record_id TEXT NOT NULL CHECK(length(trim(provider_record_id)) > 0),
            record_url TEXT,
            raw_metadata TEXT NOT NULL DEFAULT '{}',
            retrieved_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(provider, provider_record_id)
        )
        """,
        "CREATE INDEX idx_source_records_source ON source_records(source_id, provider, id)",
        """
        CREATE TABLE project_sources (
            project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
            source_id TEXT NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
            position INTEGER NOT NULL DEFAULT 0 CHECK(position >= 0),
            note TEXT NOT NULL DEFAULT '',
            added_at TEXT NOT NULL,
            PRIMARY KEY(project_id, source_id)
        )
        """,
        "CREATE INDEX idx_project_sources_order ON project_sources(project_id, position, source_id)",
        "CREATE INDEX idx_project_sources_source ON project_sources(source_id, project_id)",
    ),
)

_MIGRATION_4 = Migration(
    version=4,
    name="research_artifacts_and_notes",
    statements=(
        """
        CREATE TABLE research_artifacts (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
            source_id TEXT REFERENCES research_sources(id) ON DELETE SET NULL,
            artifact_type TEXT NOT NULL CHECK(length(trim(artifact_type)) > 0),
            title TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK(status IN ('draft', 'generating', 'ready', 'failed', 'archived')),
            current_revision_number INTEGER NOT NULL DEFAULT 0
                CHECK(current_revision_number >= 0),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_research_artifacts_project ON research_artifacts(project_id, artifact_type, updated_at DESC, id)",
        "CREATE INDEX idx_research_artifacts_source ON research_artifacts(source_id, artifact_type, id)",
        """
        CREATE TABLE artifact_revisions (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES research_artifacts(id) ON DELETE CASCADE,
            revision_number INTEGER NOT NULL CHECK(revision_number >= 1),
            parent_revision_id TEXT REFERENCES artifact_revisions(id) ON DELETE SET NULL,
            content TEXT NOT NULL,
            created_by TEXT NOT NULL
                CHECK(created_by IN ('user', 'model', 'system', 'import')),
            source_fingerprint TEXT,
            model TEXT,
            usage TEXT,
            finish_reason TEXT,
            prompt_version TEXT,
            schema_version INTEGER,
            created_at TEXT NOT NULL,
            UNIQUE(artifact_id, revision_number)
        )
        """,
        "CREATE INDEX idx_artifact_revisions_artifact ON artifact_revisions(artifact_id, revision_number DESC)",
        """
        CREATE TABLE artifact_evidence (
            artifact_revision_id TEXT NOT NULL
                REFERENCES artifact_revisions(id) ON DELETE CASCADE,
            evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE RESTRICT,
            field_path TEXT NOT NULL DEFAULT '$',
            ordinal INTEGER NOT NULL DEFAULT 0 CHECK(ordinal >= 0),
            created_at TEXT NOT NULL,
            PRIMARY KEY(artifact_revision_id, field_path, evidence_id)
        )
        """,
        "CREATE INDEX idx_artifact_evidence_evidence ON artifact_evidence(evidence_id, artifact_revision_id)",
        """
        CREATE TABLE research_notes (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
            scope_kind TEXT NOT NULL DEFAULT 'project'
                CHECK(scope_kind IN ('project', 'source', 'evidence')),
            source_id TEXT REFERENCES research_sources(id) ON DELETE SET NULL,
            evidence_id TEXT REFERENCES evidence(id) ON DELETE SET NULL,
            title TEXT NOT NULL DEFAULT '',
            content_markdown TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(
                (scope_kind='project' AND source_id IS NULL AND evidence_id IS NULL)
                OR (scope_kind='source' AND source_id IS NOT NULL AND evidence_id IS NULL)
                OR (scope_kind='evidence' AND source_id IS NULL AND evidence_id IS NOT NULL)
            )
        )
        """,
        "CREATE INDEX idx_research_notes_project ON research_notes(project_id, updated_at DESC, id)",
        "CREATE INDEX idx_research_notes_source ON research_notes(source_id, updated_at DESC, id)",
        "CREATE INDEX idx_research_notes_evidence ON research_notes(evidence_id, id)",
    ),
)

_MIGRATION_5 = Migration(
    version=5,
    name="jobs_and_persistent_agent_sessions",
    statements=(
        """
        CREATE TABLE research_jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES research_projects(id) ON DELETE CASCADE,
            artifact_id TEXT REFERENCES research_artifacts(id) ON DELETE SET NULL,
            job_type TEXT NOT NULL CHECK(length(trim(job_type)) > 0),
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK(status IN (
                    'queued', 'running', 'succeeded', 'failed',
                    'cancel_requested', 'cancelled', 'interrupted'
                )),
            payload TEXT NOT NULL DEFAULT '{}',
            result TEXT,
            error_code TEXT,
            error_message TEXT,
            progress_current INTEGER NOT NULL DEFAULT 0 CHECK(progress_current >= 0),
            progress_total INTEGER CHECK(progress_total IS NULL OR progress_total >= 0),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
            max_attempts INTEGER NOT NULL DEFAULT 1 CHECK(max_attempts >= 1),
            idempotent INTEGER NOT NULL DEFAULT 0 CHECK(idempotent IN (0, 1)),
            priority INTEGER NOT NULL DEFAULT 0,
            run_after TEXT,
            timeout_seconds INTEGER CHECK(timeout_seconds IS NULL OR timeout_seconds > 0),
            idempotency_key TEXT,
            lease_owner TEXT,
            lease_expires_at TEXT,
            cancel_requested_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_research_jobs_claim ON research_jobs(priority DESC, run_after, created_at, id) WHERE status='queued'",
        "CREATE INDEX idx_research_jobs_lease ON research_jobs(lease_expires_at, id) WHERE status='running'",
        "CREATE UNIQUE INDEX idx_research_jobs_idempotency ON research_jobs(idempotency_key) WHERE idempotency_key IS NOT NULL",
        "CREATE INDEX idx_research_jobs_project ON research_jobs(project_id, created_at DESC, id)",
        """
        CREATE TABLE agent_sessions (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES research_projects(id) ON DELETE SET NULL,
            title TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'closed')),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_agent_sessions_project ON agent_sessions(project_id, updated_at DESC, id)",
        """
        CREATE TABLE agent_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL CHECK(seq >= 1),
            role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool')),
            content TEXT NOT NULL,
            run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL,
            UNIQUE(session_id, seq)
        )
        """,
        "CREATE INDEX idx_agent_messages_session ON agent_messages(session_id, seq)",
        """
        CREATE TABLE pending_actions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            tool_call_id TEXT NOT NULL,
            tool_name TEXT NOT NULL CHECK(length(trim(tool_name)) > 0),
            tool_version TEXT NOT NULL DEFAULT '1',
            arguments TEXT NOT NULL,
            arguments_sha256 TEXT NOT NULL CHECK(length(arguments_sha256) = 64),
            confirmation TEXT NOT NULL,
            result TEXT,
            error TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN (
                    'pending', 'approved', 'rejected', 'executed',
                    'cancelled', 'expired'
                )),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT,
            resolved_at TEXT,
            UNIQUE(run_id, tool_call_id)
        )
        """,
        "CREATE INDEX idx_pending_actions_session ON pending_actions(session_id, status, created_at, id)",
        "CREATE UNIQUE INDEX idx_pending_actions_one_live ON pending_actions(session_id) WHERE status='pending'",
        "ALTER TABLE agent_runs ADD COLUMN project_id TEXT REFERENCES research_projects(id) ON DELETE SET NULL",
        "ALTER TABLE agent_runs ADD COLUMN session_id TEXT REFERENCES agent_sessions(id) ON DELETE SET NULL",
        "CREATE INDEX idx_agent_runs_project ON agent_runs(project_id, created_at DESC, id)",
        "CREATE INDEX idx_agent_runs_session ON agent_runs(session_id, created_at, id)",
    ),
)

_MIGRATION_6 = Migration(
    version=6,
    name="unique_deep_read_per_project_source",
    statements=(
        """
        CREATE UNIQUE INDEX idx_research_artifacts_deep_read_source
        ON research_artifacts(project_id, source_id, artifact_type)
        WHERE artifact_type='deep_read' AND source_id IS NOT NULL
        """,
    ),
)

MIGRATIONS: tuple[Migration, ...] = (
    _MIGRATION_1,
    _MIGRATION_2,
    _MIGRATION_3,
    _MIGRATION_4,
    _MIGRATION_5,
    _MIGRATION_6,
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version

_REQUIRED_TABLES: dict[int, frozenset[str]] = {
    1: frozenset({"papers", "chunks", "meta"}),
    2: frozenset({"evidence", "agent_runs", "agent_events"}),
    3: frozenset(
        {
            "research_projects",
            "research_questions",
            "research_sources",
            "source_identities",
            "source_records",
            "project_sources",
        }
    ),
    4: frozenset(
        {
            "research_artifacts",
            "artifact_revisions",
            "artifact_evidence",
            "research_notes",
        }
    ),
    5: frozenset(
        {"research_jobs", "agent_sessions", "agent_messages", "pending_actions"}
    ),
}

_REQUIRED_COLUMNS: dict[int, dict[str, frozenset[str]]] = {
    1: {
        "papers": frozenset(
            {
                "id",
                "path",
                "sha256",
                "title",
                "authors",
                "year",
                "page_count",
                "has_text",
                "indexed_at",
            }
        ),
        "chunks": frozenset(
            {"id", "paper_id", "seq", "page", "text", "embedding"}
        ),
        "meta": frozenset({"key", "value"}),
    },
    2: {
        "evidence": frozenset(
            {
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
            }
        ),
        "agent_runs": frozenset(
            {
                "id",
                "objective",
                "status",
                "plan",
                "budget",
                "error",
                "created_at",
                "updated_at",
            }
        ),
        "agent_events": frozenset(
            {"id", "run_id", "seq", "event_type", "payload", "created_at"}
        ),
    },
    3: {
        "papers": frozenset({"source_kind", "canonical_uri", "locator"}),
        "research_projects": frozenset(
            {"id", "title", "status", "version", "created_at", "updated_at"}
        ),
        "research_sources": frozenset(
            {
                "id",
                "canonical_key",
                "source_kind",
                "indexed_paper_id",
                "status",
                "version",
            }
        ),
        "source_identities": frozenset(
            {"id", "source_id", "identity_kind", "normalized_value", "is_primary"}
        ),
    },
    4: {
        "research_artifacts": frozenset(
            {"id", "project_id", "artifact_type", "current_revision_number", "version"}
        ),
        "artifact_revisions": frozenset(
            {"id", "artifact_id", "revision_number", "content", "source_fingerprint"}
        ),
        "artifact_evidence": frozenset(
            {"artifact_revision_id", "evidence_id", "field_path", "ordinal"}
        ),
        "research_notes": frozenset(
            {"id", "project_id", "scope_kind", "content_markdown", "version"}
        ),
    },
    5: {
        "research_jobs": frozenset(
            {
                "id",
                "job_type",
                "status",
                "payload",
                "idempotency_key",
                "lease_owner",
                "version",
            }
        ),
        "agent_sessions": frozenset(
            {"id", "project_id", "status", "version", "created_at", "updated_at"}
        ),
        "agent_messages": frozenset({"id", "session_id", "seq", "role", "content"}),
        "pending_actions": frozenset(
            {
                "id",
                "session_id",
                "run_id",
                "tool_call_id",
                "tool_name",
                "tool_version",
                "arguments",
                "arguments_sha256",
                "status",
                "version",
            }
        ),
        "agent_runs": frozenset({"project_id", "session_id"}),
    },
}


def migrate_schema(
    connection: sqlite3.Connection,
    *,
    db_path: Optional[Path] = None,
) -> MigrationReport:
    """把 ``connection`` 顺序迁移到当前版本并返回可审计报告。

    ``db_path`` 仅用于磁盘数据库的一致迁移前备份。调用方必须先启用
    ``PRAGMA foreign_keys=ON``，且不能在已有事务中调用。
    """

    if connection.in_transaction:
        raise SchemaMigrationError("schema migration 不能嵌套在已有事务中")

    previous = _read_schema_version(connection)
    if previous > LATEST_SCHEMA_VERSION:
        raise FutureSchemaVersionError(
            f"数据库 schema v{previous} 高于当前程序支持的 v{LATEST_SCHEMA_VERSION}；"
            "请升级 PRAgent，当前数据库未被修改"
        )
    _validate_declared_schema(connection, previous)

    pending = tuple(m for m in MIGRATIONS if m.version > previous)
    if not pending:
        _verify_migration_history(connection, previous)
        return MigrationReport(previous, previous, ())

    backup_path: Optional[Path] = None
    applied: list[int] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        locked_version = _read_schema_version(connection)
        if locked_version != previous:
            raise SchemaMigrationError(
                "数据库 schema 在预检与迁移锁之间发生变化；本次迁移已取消"
            )
        backup_path = _create_backup(db_path, previous)
        _create_migration_history_table(connection)
        _record_legacy_baseline(connection, previous)
        _verify_migration_history(connection, previous)

        for migration in pending:
            expected = previous + len(applied) + 1
            if migration.version != expected:
                raise SchemaMigrationError(
                    f"migration 序列不连续：期望 v{expected}，实际 v{migration.version}"
                )
            for statement in migration.statements:
                connection.execute(statement)
            _set_schema_version(connection, migration.version)
            connection.execute(
                """
                INSERT INTO schema_migrations
                    (version, name, checksum, applied_at, baseline)
                VALUES (?, ?, ?, ?, 0)
                """,
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                    _now_iso(),
                ),
            )
            applied.append(migration.version)

        _validate_declared_schema(connection, LATEST_SCHEMA_VERSION)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise SchemaMigrationError(
                f"migration 后 foreign_key_check 失败（{len(violations)} 项）"
            )
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise SchemaMigrationError("migration 后 SQLite quick_check 失败")
        connection.commit()
    except Exception as exc:
        if connection.in_transaction:
            connection.rollback()
        if isinstance(exc, SchemaMigrationError):
            raise
        backup_hint = f"；迁移前备份：{backup_path}" if backup_path else ""
        failed_version = pending[len(applied)].version if len(applied) < len(pending) else "?"
        raise SchemaMigrationError(
            f"schema migration v{failed_version} 失败，数据库事务已回滚{backup_hint}"
        ) from exc

    return MigrationReport(
        previous_version=previous,
        current_version=LATEST_SCHEMA_VERSION,
        applied_versions=tuple(applied),
        backup_path=backup_path,
    )


def _read_schema_version(connection: sqlite3.Connection) -> int:
    if not _table_exists(connection, "meta"):
        return 0
    row = connection.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    raw = row[0]
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidSchemaVersionError(
            f"无效的 schema_version：{raw!r}；数据库未被修改"
        ) from exc
    if version < 0 or str(version) != str(raw).strip():
        raise InvalidSchemaVersionError(
            f"无效的 schema_version：{raw!r}；数据库未被修改"
        )
    return version


def _validate_declared_schema(connection: sqlite3.Connection, version: int) -> None:
    existing = _user_tables(connection)
    if version == 0:
        if existing:
            raise InvalidSchemaVersionError(
                "未声明 schema_version 的数据库包含应用表；数据库未被修改"
            )
        return

    required: set[str] = set()
    for schema_version, tables in _REQUIRED_TABLES.items():
        if schema_version <= min(version, LATEST_SCHEMA_VERSION):
            required.update(tables)
    missing = sorted(required - existing)
    if missing:
        raise InvalidSchemaVersionError(
            f"数据库声明 schema v{version}，但缺少表：{', '.join(missing)}；"
            "数据库未被修改"
        )

    for schema_version, table_columns in _REQUIRED_COLUMNS.items():
        if schema_version > min(version, LATEST_SCHEMA_VERSION):
            continue
        for table, required_columns in table_columns.items():
            # pragma_table_info 是表值函数，允许绑定参数，避免拼接 SQL。
            actual_columns = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM pragma_table_info(?)", (table,)
                ).fetchall()
            }
            missing_columns = sorted(required_columns - actual_columns)
            if missing_columns:
                raise InvalidSchemaVersionError(
                    f"数据库声明 schema v{version}，但表 {table} 缺少列："
                    f"{', '.join(missing_columns)}；数据库未被修改"
                )


def _create_migration_history_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            baseline INTEGER NOT NULL DEFAULT 0 CHECK(baseline IN (0, 1))
        )
        """
    )


def _record_legacy_baseline(connection: sqlite3.Connection, version: int) -> None:
    applied_at = _now_iso()
    for migration in MIGRATIONS:
        if migration.version > version:
            break
        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations
                (version, name, checksum, applied_at, baseline)
            VALUES (?, ?, ?, ?, 1)
            """,
            (migration.version, migration.name, migration.checksum, applied_at),
        )


def _verify_migration_history(connection: sqlite3.Connection, version: int) -> None:
    if version == 0 or not _table_exists(connection, "schema_migrations"):
        return
    rows = connection.execute(
        "SELECT version, name, checksum FROM schema_migrations WHERE version <= ?",
        (version,),
    ).fetchall()
    by_version = {int(row[0]): (row[1], row[2]) for row in rows}
    for migration in MIGRATIONS:
        if migration.version > version:
            break
        recorded = by_version.get(migration.version)
        if recorded is None:
            raise InvalidSchemaVersionError(
                f"schema migration 历史缺少 v{migration.version}；数据库未被修改"
            )
        if recorded != (migration.name, migration.checksum):
            raise InvalidSchemaVersionError(
                f"schema migration v{migration.version} 校验和不匹配；数据库未被修改"
            )


def _set_schema_version(connection: sqlite3.Connection, version: int) -> None:
    connection.execute(
        """
        INSERT INTO meta(key, value) VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (str(version),),
    )


def _create_backup(
    db_path: Optional[Path],
    previous_version: int,
) -> Optional[Path]:
    if db_path is None or previous_version == 0:
        return None
    path = Path(db_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(
        f"{path.name}.pre-migration-v{previous_version}-to-v{LATEST_SCHEMA_VERSION}-"
        f"{stamp}-{uuid.uuid4().hex[:8]}.bak"
    )
    temporary_path = backup_path.with_suffix(backup_path.suffix + ".tmp")
    source = sqlite3.connect(path)
    destination = sqlite3.connect(temporary_path)
    try:
        source.backup(destination)
        destination.commit()
        quick_check = destination.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise SchemaMigrationError("迁移前备份未通过 SQLite quick_check")
        if _read_schema_version(destination) != previous_version:
            raise SchemaMigrationError("迁移前备份的 schema version 与源数据库不一致")
    except Exception:
        destination.close()
        source.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        destination.close()
        source.close()
    try:
        temporary_path.chmod(0o600)
        temporary_path.replace(backup_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return backup_path


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
