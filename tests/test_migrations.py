import sqlite3

import pytest

from pragent.storage.migrations import (
    LATEST_SCHEMA_VERSION,
    FutureSchemaVersionError,
    InvalidSchemaVersionError,
    SchemaMigrationError,
)
from pragent.store import Store


_V2_SCHEMA = """
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
INSERT INTO meta(key, value) VALUES('schema_version', '2');
INSERT INTO meta(key, value) VALUES('index_revision', '7');
INSERT INTO papers VALUES(
    1, 'legacy.pdf', 'legacy-sha', '旧论文', '["旧作者"]', 2020, 2, 1,
    '2026-01-01T00:00:00'
);
INSERT INTO chunks VALUES(1, 1, 0, 1, '旧正文', NULL);
"""


def _create_v2_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(_V2_SCHEMA)
    connection.commit()
    connection.close()


def _tables(path):
    connection = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        connection.close()


def test_fresh_database_runs_all_migrations_in_order(tmp_path):
    db_path = tmp_path / "fresh.db"

    store = Store(db_path)
    report = store.migration_report
    store.close()

    assert report.previous_version == 0
    assert report.current_version == LATEST_SCHEMA_VERSION
    assert report.applied_versions == tuple(range(1, LATEST_SCHEMA_VERSION + 1))
    assert report.backup_path is None

    expected_tables = {
        "papers",
        "chunks",
        "evidence",
        "agent_runs",
        "agent_events",
        "schema_migrations",
        "research_projects",
        "research_questions",
        "research_sources",
        "source_identities",
        "source_records",
        "project_sources",
        "research_artifacts",
        "artifact_revisions",
        "artifact_evidence",
        "research_notes",
        "research_jobs",
        "agent_sessions",
        "agent_messages",
        "pending_actions",
    }
    assert expected_tables <= _tables(db_path)

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == str(LATEST_SCHEMA_VERSION)
    assert connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall() == [(version,) for version in range(1, LATEST_SCHEMA_VERSION + 1)]
    paper_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(papers)").fetchall()
    }
    assert {"source_kind", "canonical_uri", "locator"} <= paper_columns
    agent_run_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(agent_runs)").fetchall()
    }
    assert {"project_id", "session_id"} <= agent_run_columns
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    connection.close()

    reopened = Store(db_path)
    assert reopened.migration_report.previous_version == LATEST_SCHEMA_VERSION
    assert reopened.migration_report.applied_versions == ()
    assert reopened.migration_report.backup_path is None
    reopened.close()


def test_existing_v2_database_is_backed_up_then_migrated(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_v2_database(db_path)

    store = Store(db_path)
    report = store.migration_report

    assert report.previous_version == 2
    assert report.applied_versions == (3, 4, 5)
    assert report.backup_path is not None and report.backup_path.is_file()
    assert store.meta_get("schema_version") == str(LATEST_SCHEMA_VERSION)
    assert store.paper_by_id(1).title == "旧论文"
    store.close()

    backup = sqlite3.connect(report.backup_path)
    assert backup.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == "2"
    assert backup.execute("SELECT title FROM papers WHERE id=1").fetchone()[0] == "旧论文"
    assert "research_projects" not in {
        row[0]
        for row in backup.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    backup.close()


def test_failed_migration_rolls_back_and_keeps_recovery_backup(tmp_path):
    db_path = tmp_path / "broken-v2.db"
    _create_v2_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE research_projects(id TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(SchemaMigrationError, match="事务已回滚"):
        Store(db_path)

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == "2"
    assert connection.execute("SELECT title FROM papers WHERE id=1").fetchone()[0] == "旧论文"
    assert "research_sources" not in {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    connection.close()

    backups = list(tmp_path.glob("broken-v2.db.pre-migration-v2-to-v*.bak"))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    assert backup.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == "2"
    backup.close()


def test_future_and_invalid_versions_fail_closed(tmp_path):
    future_path = tmp_path / "future.db"
    connection = sqlite3.connect(future_path)
    connection.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);"
        f"INSERT INTO meta VALUES('schema_version', '{LATEST_SCHEMA_VERSION + 1}');"
    )
    connection.commit()
    connection.close()

    with pytest.raises(FutureSchemaVersionError, match="高于当前程序"):
        Store(future_path)
    assert list(tmp_path.glob("future.db.pre-migration-*.bak")) == []

    invalid_path = tmp_path / "invalid.db"
    connection = sqlite3.connect(invalid_path)
    connection.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);"
        "INSERT INTO meta VALUES('schema_version', 'not-a-number');"
    )
    connection.commit()
    connection.close()

    with pytest.raises(InvalidSchemaVersionError, match="无效"):
        Store(invalid_path)
    check = sqlite3.connect(invalid_path)
    assert check.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == "not-a-number"
    check.close()


def test_unversioned_application_tables_are_not_treated_as_fresh(tmp_path):
    db_path = tmp_path / "unversioned.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE papers(id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(InvalidSchemaVersionError, match="未声明"):
        Store(db_path)
    assert _tables(db_path) == {"papers"}
    assert list(tmp_path.glob("unversioned.db.pre-migration-*.bak")) == []


def test_wal_aware_backup_contains_committed_uncheckpointed_rows(tmp_path):
    db_path = tmp_path / "wal-v2.db"
    _create_v2_database(db_path)
    writer = sqlite3.connect(db_path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        """
        INSERT INTO papers VALUES(
            2, 'wal.pdf', 'wal-sha', 'WAL 论文', '["作者"]', 2021, 1, 1,
            '2026-01-01T00:00:00'
        )
        """
    )
    writer.commit()
    assert db_path.with_name(db_path.name + "-wal").is_file()

    store = Store(db_path)
    backup_path = store.migration_report.backup_path
    store.close()
    writer.close()

    assert backup_path is not None
    backup = sqlite3.connect(backup_path)
    assert backup.execute("SELECT title FROM papers WHERE id=2").fetchone()[0] == "WAL 论文"
    backup.close()


def test_research_table_checks_and_cascades_are_enforced(tmp_path):
    db_path = tmp_path / "constraints.db"
    store = Store(db_path)
    store.close()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO research_projects
                (id, title, status, created_at, updated_at)
            VALUES ('project_bad', 'Bad', 'unknown', 'now', 'now')
            """
        )

    connection.execute(
        """
        INSERT INTO research_projects(id, title, created_at, updated_at)
        VALUES ('project_1', '项目', 'now', 'now')
        """
    )
    connection.execute(
        """
        INSERT INTO research_questions
            (id, project_id, question, position, created_at, updated_at)
        VALUES ('question_1', 'project_1', '研究问题？', 0, 'now', 'now')
        """
    )
    connection.commit()
    connection.execute("DELETE FROM research_projects WHERE id='project_1'")
    connection.commit()
    assert connection.execute("SELECT COUNT(*) FROM research_questions").fetchone()[0] == 0
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    connection.close()
