"""Project/source/artifact/note persistence, kept outside the legacy Store facade."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pragent.models import (
    ArtifactEvidenceLink,
    ArtifactFreshness,
    ArtifactRevision,
    Page,
    ProjectSourceMembership,
    ResearchArtifact,
    ResearchNote,
    ResearchProject,
    ResearchQuestion,
    ResearchSource,
    SourceIdentity,
    SourceRecord,
)

from ._repository import RecordVersionConflictError, SQLiteRepository


class SourceIdentityConflictError(RuntimeError):
    """同一个规范化 identity 已属于另一个 canonical source。"""


class ArtifactValidationError(RuntimeError):
    """Artifact revision 的来源、证据或模型审计信息不满足保存合同。"""


_UNSET = object()
_PROJECT_STATUSES = frozenset({"active", "archived"})
_SOURCE_KINDS = frozenset({"paper", "web"})
_SOURCE_STATUSES = frozenset({"discovered", "fetching", "ready", "failed", "archived"})
_ARTIFACT_STATUSES = frozenset({"draft", "generating", "ready", "failed", "archived"})
_ARTIFACT_CREATORS = frozenset({"user", "model", "system", "import"})
_IDENTITY_KINDS = frozenset({"doi", "arxiv", "url", "content_sha256"})
_NOTE_SCOPES = frozenset({"project", "source", "evidence"})
_DEEP_READ_FIELD_PATHS = frozenset(
    {
        "$.research_question",
        "$.related_work",
        "$.core_method",
        "$.contributions",
        "$.datasets_and_experiments",
        "$.main_results",
        "$.limitations",
        "$.future_work",
        "$.key_evidence",
    }
)


class ResearchRepository(SQLiteRepository):
    """研究工作区 repository；复合写入均在本类事务内完成。"""

    # ---------- projects ----------

    def create_project(
        self,
        title: str,
        *,
        description: str = "",
        default_language: str = "zh-CN",
        citation_style: str = "gb-t-7714-2015-numeric",
        status: str = "active",
        project_id: Optional[str] = None,
    ) -> ResearchProject:
        title = _required_text(title, "title")
        default_language = _required_text(default_language, "default_language")
        citation_style = _required_text(citation_style, "citation_style")
        _validate_choice(status, _PROJECT_STATUSES, "status")
        project_id = project_id or _new_id("project")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO research_projects(
                    id, title, description, default_language, citation_style,
                    status, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    project_id,
                    title,
                    str(description),
                    default_language,
                    citation_style,
                    status,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM research_projects WHERE id=?", (project_id,)
            ).fetchone()
        return self._project_from_row(row)

    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_projects WHERE id=?", (project_id,)
            ).fetchone()
        return self._project_from_row(row) if row else None

    def list_projects(
        self,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[ResearchProject]:
        limit, offset = _validate_page(limit, offset)
        where: list[str] = []
        params: list[Any] = []
        if q:
            where.append("(title LIKE ? OR description LIKE ?)")
            like = f"%{q}%"
            params.extend((like, like))
        if status:
            _validate_choice(status, _PROJECT_STATUSES, "status")
            where.append("status=?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM research_projects {clause}", params
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                f"""
                SELECT * FROM research_projects {clause}
                ORDER BY updated_at DESC, id LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return Page(total, tuple(self._project_from_row(row) for row in rows), limit, offset)

    def update_project(
        self,
        project_id: str,
        *,
        expected_version: int,
        title: Any = _UNSET,
        description: Any = _UNSET,
        default_language: Any = _UNSET,
        citation_style: Any = _UNSET,
        status: Any = _UNSET,
    ) -> ResearchProject:
        values: dict[str, Any] = {}
        if title is not _UNSET:
            values["title"] = _required_text(title, "title")
        if description is not _UNSET:
            values["description"] = str(description)
        if default_language is not _UNSET:
            values["default_language"] = _required_text(
                default_language, "default_language"
            )
        if citation_style is not _UNSET:
            values["citation_style"] = _required_text(citation_style, "citation_style")
        if status is not _UNSET:
            _validate_choice(status, _PROJECT_STATUSES, "status")
            values["status"] = status
        if not values:
            project = self.get_project(project_id)
            if project is None:
                raise KeyError(f"研究项目不存在：{project_id}")
            if project.version != expected_version:
                raise RecordVersionConflictError(
                    f"研究项目 {project_id} 版本冲突：期望 {expected_version}，当前 {project.version}"
                )
            return project
        return self._cas_update_project(project_id, expected_version, values)

    def _cas_update_project(
        self, project_id: str, expected_version: int, values: dict[str, Any]
    ) -> ResearchProject:
        assignments = [f"{column}=?" for column in values]
        params = [*values.values(), _now_iso(), project_id, expected_version]
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE research_projects
                SET {', '.join(assignments)}, updated_at=?, version=version+1
                WHERE id=? AND version=?
                """,
                params,
            )
            if not cursor.rowcount:
                self._raise_cas_failure_locked(
                    connection,
                    "research_projects",
                    project_id,
                    expected_version,
                    "研究项目",
                )
            row = connection.execute(
                "SELECT * FROM research_projects WHERE id=?", (project_id,)
            ).fetchone()
        return self._project_from_row(row)

    # ---------- questions ----------

    def create_question(
        self,
        project_id: str,
        question: str,
        *,
        position: Optional[int] = None,
        question_id: Optional[str] = None,
    ) -> ResearchQuestion:
        question = _required_text(question, "question")
        question_id = question_id or _new_id("question")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            self._require_row_locked(
                connection, "research_projects", project_id, "研究项目"
            )
            if position is None:
                position = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(position), -1) + 1
                        FROM research_questions WHERE project_id=?
                        """,
                        (project_id,),
                    ).fetchone()[0]
                )
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("position 必须是非负整数")
            connection.execute(
                """
                INSERT INTO research_questions(
                    id, project_id, question, position, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?)
                """,
                (question_id, project_id, question, position, now, now),
            )
            row = connection.execute(
                "SELECT * FROM research_questions WHERE id=?", (question_id,)
            ).fetchone()
        return self._question_from_row(row)

    def list_questions(self, project_id: str) -> tuple[ResearchQuestion, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM research_questions
                WHERE project_id=? ORDER BY position, id
                """,
                (project_id,),
            ).fetchall()
        return tuple(self._question_from_row(row) for row in rows)

    def update_question(
        self,
        question_id: str,
        *,
        expected_version: int,
        question: Any = _UNSET,
        position: Any = _UNSET,
        project_id: Optional[str] = None,
    ) -> ResearchQuestion:
        values: dict[str, Any] = {}
        if question is not _UNSET:
            values["question"] = _required_text(question, "question")
        if position is not _UNSET:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("position 必须是非负整数")
            values["position"] = position
        if not values:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM research_questions WHERE id=?", (question_id,)
                ).fetchone()
            if row is None or (
                project_id is not None and row["project_id"] != project_id
            ):
                raise KeyError(f"研究问题不存在或不属于当前项目：{question_id}")
            current = self._question_from_row(row)
            if current.version != expected_version:
                raise RecordVersionConflictError(
                    f"研究问题 {question_id} 版本冲突：期望 {expected_version}，当前 {current.version}"
                )
            return current
        assignments = [f"{column}=?" for column in values]
        scope_clause = " AND project_id=?" if project_id is not None else ""
        params = [*values.values(), _now_iso(), question_id, expected_version]
        if project_id is not None:
            params.append(project_id)
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE research_questions
                SET {', '.join(assignments)}, updated_at=?, version=version+1
                WHERE id=? AND version=?{scope_clause}
                """,
                params,
            )
            if not cursor.rowcount:
                current = connection.execute(
                    "SELECT project_id FROM research_questions WHERE id=?",
                    (question_id,),
                ).fetchone()
                if current is None or (
                    project_id is not None and current["project_id"] != project_id
                ):
                    raise KeyError(
                        f"研究问题不存在或不属于当前项目：{question_id}"
                    )
                self._raise_cas_failure_locked(
                    connection,
                    "research_questions",
                    question_id,
                    expected_version,
                    "研究问题",
                )
            row = connection.execute(
                "SELECT * FROM research_questions WHERE id=?", (question_id,)
            ).fetchone()
        return self._question_from_row(row)

    def delete_question(
        self,
        question_id: str,
        *,
        expected_version: int,
        project_id: Optional[str] = None,
    ) -> None:
        scope_clause = " AND project_id=?" if project_id is not None else ""
        params: list[Any] = [question_id, expected_version]
        if project_id is not None:
            params.append(project_id)
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM research_questions
                WHERE id=? AND version=?{scope_clause}
                """,
                params,
            )
            if not cursor.rowcount:
                current = connection.execute(
                    "SELECT project_id FROM research_questions WHERE id=?",
                    (question_id,),
                ).fetchone()
                if current is None or (
                    project_id is not None and current["project_id"] != project_id
                ):
                    raise KeyError(
                        f"研究问题不存在或不属于当前项目：{question_id}"
                    )
                self._raise_cas_failure_locked(
                    connection,
                    "research_questions",
                    question_id,
                    expected_version,
                    "研究问题",
                )

    # ---------- sources and provenance ----------

    def create_source(
        self,
        canonical_key: str,
        source_kind: str,
        *,
        title: str = "",
        authors: Iterable[str] = (),
        year: Optional[int] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        canonical_url: Optional[str] = None,
        content_sha256: Optional[str] = None,
        indexed_paper_id: Optional[int] = None,
        status: str = "discovered",
        metadata: Any = None,
        locator: Any = None,
        snapshot_path: Optional[str] = None,
        snapshot_sha256: Optional[str] = None,
        extracted_text: Optional[str] = None,
        fetched_at: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> ResearchSource:
        canonical_key = _required_text(canonical_key, "canonical_key")
        _validate_choice(source_kind, _SOURCE_KINDS, "source_kind")
        _validate_choice(status, _SOURCE_STATUSES, "status")
        source_id = source_id or _new_id("source")
        authors_json = _json_dump([str(author) for author in authors])
        metadata_json = _json_dump({} if metadata is None else metadata)
        locator_json = _json_dump({} if locator is None else locator)
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO research_sources(
                        id, canonical_key, source_kind, title, authors, year, doi,
                        arxiv_id, canonical_url, content_sha256, indexed_paper_id,
                        status, metadata, locator, snapshot_path, snapshot_sha256,
                        extracted_text, fetched_at, version, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?
                    )
                    """,
                    (
                        source_id,
                        canonical_key,
                        source_kind,
                        str(title),
                        authors_json,
                        year,
                        doi,
                        arxiv_id,
                        canonical_url,
                        content_sha256,
                        indexed_paper_id,
                        status,
                        metadata_json,
                        locator_json,
                        snapshot_path,
                        snapshot_sha256,
                        extracted_text,
                        fetched_at,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = connection.execute(
                    "SELECT id FROM research_sources WHERE canonical_key=?",
                    (canonical_key,),
                ).fetchone()
                if existing:
                    raise SourceIdentityConflictError(
                        f"canonical source 已存在：{existing['id']}"
                    ) from exc
                raise
            row = connection.execute(
                "SELECT * FROM research_sources WHERE id=?", (source_id,)
            ).fetchone()
        return self._source_from_row(row)

    def upsert_merged_source(self, merged: Any) -> ResearchSource:
        """原子持久化 canonical metadata、全部 identity 和 provider provenance。

        如果一条桥接记录同时命中多个既有 source，会先把 membership、artifact、
        note、identity 和 provider record 合并到一个 winner，再删除重复 source。
        """

        canonical_key = _required_text(merged.canonical_key, "canonical_key")
        source = merged.source
        _validate_choice(source.source_kind, _SOURCE_KINDS, "source_kind")
        identities = tuple(
            (str(kind), _required_text(value, "normalized_value"))
            for kind, value in merged.identities
        )
        for kind, _ in identities:
            _validate_choice(kind, _IDENTITY_KINDS, "identity_kind")
        if len(set(identities)) != len(identities):
            raise ValueError("merged source 包含重复 identity")
        records = tuple(merged.provenance)
        if not records:
            raise ValueError("merged source 必须保留 provider provenance")
        authors_json = _json_dump([str(author) for author in source.authors])
        metadata = dict(source.metadata)
        if source.abstract:
            metadata.setdefault("abstract", source.abstract)
        if source.pdf_url:
            metadata.setdefault("pdf_url", source.pdf_url)
        metadata["providers"] = sorted({record.provider for record in records})
        metadata_json = _json_dump(metadata)
        serialized_records = tuple(
            (
                _required_text(record.provider, "provider"),
                _required_text(record.record_id, "provider_record_id"),
                record.record_url,
                _json_dump(record.raw_metadata),
                record.retrieved_at or _now_iso(),
            )
            for record in records
        )

        with self._transaction(immediate=True) as connection:
            candidate_ids: set[str] = set()
            by_key = connection.execute(
                "SELECT id FROM research_sources WHERE canonical_key=?",
                (canonical_key,),
            ).fetchone()
            if by_key:
                candidate_ids.add(by_key["id"])
            for kind, value in identities:
                row = connection.execute(
                    """
                    SELECT source_id FROM source_identities
                    WHERE identity_kind=? AND normalized_value=?
                    """,
                    (kind, value),
                ).fetchone()
                if row:
                    candidate_ids.add(row["source_id"])
            for provider, provider_record_id, *_ in serialized_records:
                row = connection.execute(
                    """
                    SELECT source_id FROM source_records
                    WHERE provider=? AND provider_record_id=?
                    """,
                    (provider, provider_record_id),
                ).fetchone()
                if row:
                    candidate_ids.add(row["source_id"])
            for column, value in (
                ("doi", source.doi),
                ("arxiv_id", source.arxiv_id),
            ):
                if value:
                    row = connection.execute(
                        f"SELECT id FROM research_sources WHERE {column}=?", (value,)
                    ).fetchone()
                    if row:
                        candidate_ids.add(row["id"])

            rows = [
                connection.execute(
                    "SELECT * FROM research_sources WHERE id=?", (source_id,)
                ).fetchone()
                for source_id in candidate_ids
            ]
            rows = [row for row in rows if row is not None]
            created = not rows
            if created:
                source_id = _new_id("source")
                now = _now_iso()
                connection.execute(
                    """
                    INSERT INTO research_sources(
                        id, canonical_key, source_kind, title, authors, year, doi,
                        arxiv_id, canonical_url, content_sha256, status, metadata,
                        locator, version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'discovered', ?, '{}', 1, ?, ?)
                    """,
                    (
                        source_id,
                        canonical_key,
                        source.source_kind,
                        source.title,
                        authors_json,
                        source.year,
                        source.doi,
                        source.arxiv_id,
                        source.canonical_url,
                        source.content_sha256,
                        metadata_json,
                        now,
                        now,
                    ),
                )
            else:
                winner = min(
                    rows,
                    key=lambda row: (
                        0 if row["canonical_key"] == canonical_key else 1,
                        0 if row["indexed_paper_id"] is not None else 1,
                        0 if row["status"] == "ready" else 1,
                        row["created_at"],
                        row["id"],
                    ),
                )
                source_id = winner["id"]
                for loser in sorted(rows, key=lambda row: row["id"]):
                    if loser["id"] != source_id:
                        self._merge_source_locked(connection, source_id, loser["id"])

            for kind, value in identities:
                existing = connection.execute(
                    """
                    SELECT source_id FROM source_identities
                    WHERE identity_kind=? AND normalized_value=?
                    """,
                    (kind, value),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO source_identities(
                            id, source_id, identity_kind, normalized_value,
                            is_primary, created_at
                        ) VALUES (?, ?, ?, ?, 0, ?)
                        """,
                        (_new_id("identity"), source_id, kind, value, _now_iso()),
                    )
                elif existing["source_id"] != source_id:
                    raise SourceIdentityConflictError(
                        f"{kind}:{value} 已属于来源 {existing['source_id']}"
                    )

            for provider, provider_record_id, record_url, raw_json, retrieved_at in serialized_records:
                existing = connection.execute(
                    """
                    SELECT id, source_id FROM source_records
                    WHERE provider=? AND provider_record_id=?
                    """,
                    (provider, provider_record_id),
                ).fetchone()
                if existing and existing["source_id"] != source_id:
                    raise SourceIdentityConflictError(
                        f"{provider}:{provider_record_id} 已属于来源 {existing['source_id']}"
                    )
                now = _now_iso()
                if existing:
                    connection.execute(
                        """
                        UPDATE source_records SET record_url=?, raw_metadata=?,
                            retrieved_at=?, updated_at=? WHERE id=?
                        """,
                        (record_url, raw_json, retrieved_at, now, existing["id"]),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO source_records(
                            id, source_id, provider, provider_record_id, record_url,
                            raw_metadata, retrieved_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _new_id("record"),
                            source_id,
                            provider,
                            provider_record_id,
                            record_url,
                            raw_json,
                            retrieved_at,
                            now,
                            now,
                        ),
                    )

            persisted_providers = {
                row["provider"]
                for row in connection.execute(
                    "SELECT DISTINCT provider FROM source_records WHERE source_id=?",
                    (source_id,),
                ).fetchall()
            }
            metadata["providers"] = sorted(persisted_providers)
            metadata_json = _json_dump(metadata)

            all_identities = connection.execute(
                """
                SELECT identity_kind, normalized_value FROM source_identities
                WHERE source_id=?
                """,
                (source_id,),
            ).fetchall()
            ordered_identities = sorted(
                (
                    (row["identity_kind"], row["normalized_value"])
                    for row in all_identities
                ),
                key=lambda item: (
                    {"doi": 0, "arxiv": 1, "url": 2, "content_sha256": 3}[
                        item[0]
                    ],
                    item[1],
                ),
            )
            identity_values: dict[str, str] = {}
            for identity_kind, normalized_value in ordered_identities:
                identity_values.setdefault(identity_kind, normalized_value)
            if ordered_identities:
                primary_kind, primary_value = ordered_identities[0]
                preferred_kind, separator, preferred_value = canonical_key.partition(":")
                preferred = (preferred_kind, preferred_value)
                if (
                    separator
                    and preferred in ordered_identities
                    and {"doi": 0, "arxiv": 1, "url": 2, "content_sha256": 3}[
                        preferred_kind
                    ]
                    == {"doi": 0, "arxiv": 1, "url": 2, "content_sha256": 3}[
                        primary_kind
                    ]
                ):
                    primary_kind, primary_value = preferred
                final_key = f"{primary_kind}:{primary_value}"
                connection.execute(
                    "UPDATE source_identities SET is_primary=0 WHERE source_id=?",
                    (source_id,),
                )
                connection.execute(
                    """
                    UPDATE source_identities SET is_primary=1
                    WHERE source_id=? AND identity_kind=? AND normalized_value=?
                    """,
                    (source_id, primary_kind, primary_value),
                )
            else:
                final_key = canonical_key

            current = connection.execute(
                "SELECT * FROM research_sources WHERE id=?", (source_id,)
            ).fetchone()
            if not created:
                existing_metadata = json.loads(current["metadata"])
                existing_metadata.update(metadata)
                metadata_json = _json_dump(existing_metadata)
                indexed_paper_id = current["indexed_paper_id"]
                status = "ready" if current["status"] == "ready" else "discovered"
                connection.execute(
                    """
                    UPDATE research_sources SET canonical_key=?, source_kind=?,
                        title=?, authors=?, year=?, doi=?, arxiv_id=?,
                        canonical_url=?, content_sha256=?, status=?, metadata=?,
                        updated_at=?, version=version+1
                    WHERE id=?
                    """,
                    (
                        final_key,
                        "paper"
                        if source.source_kind == "paper"
                        or current["source_kind"] == "paper"
                        else "web",
                        source.title or current["title"],
                        authors_json if source.authors else current["authors"],
                        source.year if source.year is not None else current["year"],
                        identity_values.get("doi") or current["doi"],
                        identity_values.get("arxiv") or current["arxiv_id"],
                        identity_values.get("url")
                        or source.canonical_url
                        or current["canonical_url"],
                        identity_values.get("content_sha256")
                        or current["content_sha256"],
                        status if indexed_paper_id is None else "ready",
                        metadata_json,
                        _now_iso(),
                        source_id,
                    ),
                )
            elif final_key != canonical_key:
                connection.execute(
                    """
                    UPDATE research_sources SET canonical_key=?, updated_at=?
                    WHERE id=?
                    """,
                    (final_key, _now_iso(), source_id),
                )
            row = connection.execute(
                "SELECT * FROM research_sources WHERE id=?", (source_id,)
            ).fetchone()
        return self._source_from_row(row)

    @staticmethod
    def _merge_source_locked(
        connection: sqlite3.Connection, winner_id: str, loser_id: str
    ) -> None:
        winner = connection.execute(
            "SELECT * FROM research_sources WHERE id=?", (winner_id,)
        ).fetchone()
        loser = connection.execute(
            "SELECT * FROM research_sources WHERE id=?", (loser_id,)
        ).fetchone()
        if winner is None or loser is None:
            raise KeyError("待合并研究来源不存在")
        memberships = connection.execute(
            "SELECT * FROM project_sources WHERE source_id=?", (loser_id,)
        ).fetchall()
        for membership in memberships:
            connection.execute(
                """
                INSERT INTO project_sources(
                    project_id, source_id, position, note, added_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, source_id) DO UPDATE SET
                    position=MIN(project_sources.position, excluded.position),
                    note=CASE WHEN project_sources.note<>'' THEN project_sources.note
                              ELSE excluded.note END
                """,
                (
                    membership["project_id"],
                    winner_id,
                    membership["position"],
                    membership["note"],
                    membership["added_at"],
                ),
            )
        connection.execute("DELETE FROM project_sources WHERE source_id=?", (loser_id,))
        connection.execute(
            "UPDATE research_artifacts SET source_id=? WHERE source_id=?",
            (winner_id, loser_id),
        )
        connection.execute(
            "UPDATE research_notes SET source_id=? WHERE source_id=?",
            (winner_id, loser_id),
        )
        connection.execute(
            "UPDATE source_records SET source_id=? WHERE source_id=?",
            (winner_id, loser_id),
        )
        connection.execute(
            "UPDATE source_identities SET is_primary=0 WHERE source_id IN (?, ?)",
            (winner_id, loser_id),
        )
        connection.execute(
            "UPDATE source_identities SET source_id=? WHERE source_id=?",
            (winner_id, loser_id),
        )
        inherited_paper_id = (
            winner["indexed_paper_id"]
            if winner["indexed_paper_id"] is not None
            else loser["indexed_paper_id"]
        )
        if loser["indexed_paper_id"] is not None:
            connection.execute(
                "UPDATE research_sources SET indexed_paper_id=NULL WHERE id=?",
                (loser_id,),
            )
        connection.execute(
            """
            UPDATE research_sources SET
                source_kind=CASE WHEN source_kind='paper' OR ?='paper'
                                 THEN 'paper' ELSE 'web' END,
                title=CASE WHEN title='' THEN ? ELSE title END,
                authors=CASE WHEN authors='[]' THEN ? ELSE authors END,
                year=COALESCE(year, ?),
                indexed_paper_id=?,
                status=CASE WHEN status='ready' OR ?='ready'
                            THEN 'ready' ELSE status END,
                locator=CASE WHEN locator='{}' THEN ? ELSE locator END,
                snapshot_path=COALESCE(snapshot_path, ?),
                snapshot_sha256=COALESCE(snapshot_sha256, ?),
                extracted_text=COALESCE(extracted_text, ?),
                fetched_at=COALESCE(fetched_at, ?)
            WHERE id=?
            """,
            (
                loser["source_kind"],
                loser["title"],
                loser["authors"],
                loser["year"],
                inherited_paper_id,
                loser["status"],
                loser["locator"],
                loser["snapshot_path"],
                loser["snapshot_sha256"],
                loser["extracted_text"],
                loser["fetched_at"],
                winner_id,
            ),
        )
        connection.execute("DELETE FROM research_sources WHERE id=?", (loser_id,))

    def get_source(self, source_id: str) -> Optional[ResearchSource]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_sources WHERE id=?", (source_id,)
            ).fetchone()
        return self._source_from_row(row) if row else None

    def get_source_by_canonical_key(self, canonical_key: str) -> Optional[ResearchSource]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_sources WHERE canonical_key=?",
                (canonical_key,),
            ).fetchone()
        return self._source_from_row(row) if row else None

    def list_sources(
        self,
        *,
        q: Optional[str] = None,
        source_kind: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[ResearchSource]:
        limit, offset = _validate_page(limit, offset)
        where: list[str] = []
        params: list[Any] = []
        if q:
            where.append(
                "(title LIKE ? OR authors LIKE ? OR doi LIKE ? OR arxiv_id LIKE ?)"
            )
            like = f"%{q}%"
            params.extend((like, like, like, like))
        if source_kind:
            _validate_choice(source_kind, _SOURCE_KINDS, "source_kind")
            where.append("source_kind=?")
            params.append(source_kind)
        if status:
            _validate_choice(status, _SOURCE_STATUSES, "status")
            where.append("status=?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM research_sources {clause}", params
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                f"""
                SELECT * FROM research_sources {clause}
                ORDER BY updated_at DESC, id LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return Page(total, tuple(self._source_from_row(row) for row in rows), limit, offset)

    def update_source(
        self,
        source_id: str,
        *,
        expected_version: int,
        title: Any = _UNSET,
        authors: Any = _UNSET,
        year: Any = _UNSET,
        doi: Any = _UNSET,
        arxiv_id: Any = _UNSET,
        canonical_url: Any = _UNSET,
        content_sha256: Any = _UNSET,
        indexed_paper_id: Any = _UNSET,
        status: Any = _UNSET,
        metadata: Any = _UNSET,
        locator: Any = _UNSET,
        snapshot_path: Any = _UNSET,
        snapshot_sha256: Any = _UNSET,
        extracted_text: Any = _UNSET,
        fetched_at: Any = _UNSET,
    ) -> ResearchSource:
        values: dict[str, Any] = {}
        for column, value in (
            ("title", title),
            ("year", year),
            ("doi", doi),
            ("arxiv_id", arxiv_id),
            ("canonical_url", canonical_url),
            ("content_sha256", content_sha256),
            ("indexed_paper_id", indexed_paper_id),
            ("snapshot_path", snapshot_path),
            ("snapshot_sha256", snapshot_sha256),
            ("extracted_text", extracted_text),
            ("fetched_at", fetched_at),
        ):
            if value is not _UNSET:
                values[column] = value
        if authors is not _UNSET:
            values["authors"] = _json_dump([str(author) for author in authors])
        if status is not _UNSET:
            _validate_choice(status, _SOURCE_STATUSES, "status")
            values["status"] = status
        if metadata is not _UNSET:
            values["metadata"] = _json_dump(metadata)
        if locator is not _UNSET:
            values["locator"] = _json_dump(locator)
        if not values:
            source = self.get_source(source_id)
            if source is None:
                raise KeyError(f"研究来源不存在：{source_id}")
            if source.version != expected_version:
                raise RecordVersionConflictError(
                    f"研究来源 {source_id} 版本冲突：期望 {expected_version}，当前 {source.version}"
                )
            return source
        assignments = [f"{column}=?" for column in values]
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE research_sources
                SET {', '.join(assignments)}, updated_at=?, version=version+1
                WHERE id=? AND version=?
                """,
                (*values.values(), _now_iso(), source_id, expected_version),
            )
            if not cursor.rowcount:
                self._raise_cas_failure_locked(
                    connection,
                    "research_sources",
                    source_id,
                    expected_version,
                    "研究来源",
                )
            row = connection.execute(
                "SELECT * FROM research_sources WHERE id=?", (source_id,)
            ).fetchone()
        return self._source_from_row(row)

    def add_source_identity(
        self,
        source_id: str,
        identity_kind: str,
        normalized_value: str,
        *,
        is_primary: bool = False,
        identity_id: Optional[str] = None,
    ) -> SourceIdentity:
        _validate_choice(identity_kind, _IDENTITY_KINDS, "identity_kind")
        normalized_value = _required_text(normalized_value, "normalized_value")
        identity_id = identity_id or _new_id("identity")
        with self._transaction(immediate=True) as connection:
            self._require_row_locked(connection, "research_sources", source_id, "研究来源")
            existing = connection.execute(
                """
                SELECT * FROM source_identities
                WHERE identity_kind=? AND normalized_value=?
                """,
                (identity_kind, normalized_value),
            ).fetchone()
            if existing:
                if existing["source_id"] != source_id:
                    raise SourceIdentityConflictError(
                        f"{identity_kind}:{normalized_value} 已属于来源 {existing['source_id']}"
                    )
                return self._identity_from_row(existing)
            if is_primary:
                connection.execute(
                    "UPDATE source_identities SET is_primary=0 WHERE source_id=? AND is_primary=1",
                    (source_id,),
                )
            connection.execute(
                """
                INSERT INTO source_identities(
                    id, source_id, identity_kind, normalized_value, is_primary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    identity_id,
                    source_id,
                    identity_kind,
                    normalized_value,
                    int(is_primary),
                    _now_iso(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM source_identities WHERE id=?", (identity_id,)
            ).fetchone()
        return self._identity_from_row(row)

    def list_source_identities(self, source_id: str) -> tuple[SourceIdentity, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM source_identities WHERE source_id=?
                ORDER BY is_primary DESC, identity_kind, normalized_value
                """,
                (source_id,),
            ).fetchall()
        return tuple(self._identity_from_row(row) for row in rows)

    def add_source_record(
        self,
        source_id: str,
        provider: str,
        provider_record_id: str,
        raw_metadata: Any,
        *,
        record_url: Optional[str] = None,
        retrieved_at: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> SourceRecord:
        provider = _required_text(provider, "provider")
        provider_record_id = _required_text(provider_record_id, "provider_record_id")
        raw_json = _json_dump(raw_metadata)
        retrieved_at = retrieved_at or _now_iso()
        record_id = record_id or _new_id("record")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            self._require_row_locked(connection, "research_sources", source_id, "研究来源")
            existing = connection.execute(
                """
                SELECT * FROM source_records
                WHERE provider=? AND provider_record_id=?
                """,
                (provider, provider_record_id),
            ).fetchone()
            if existing and existing["source_id"] != source_id:
                raise SourceIdentityConflictError(
                    f"{provider}:{provider_record_id} 已属于来源 {existing['source_id']}"
                )
            if existing:
                connection.execute(
                    """
                    UPDATE source_records
                    SET record_url=?, raw_metadata=?, retrieved_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (record_url, raw_json, retrieved_at, now, existing["id"]),
                )
                record_id = existing["id"]
            else:
                connection.execute(
                    """
                    INSERT INTO source_records(
                        id, source_id, provider, provider_record_id, record_url,
                        raw_metadata, retrieved_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        source_id,
                        provider,
                        provider_record_id,
                        record_url,
                        raw_json,
                        retrieved_at,
                        now,
                        now,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM source_records WHERE id=?", (record_id,)
            ).fetchone()
        return self._record_from_row(row)

    def list_source_records(self, source_id: str) -> tuple[SourceRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM source_records WHERE source_id=?
                ORDER BY provider, provider_record_id
                """,
                (source_id,),
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def ensure_source_for_paper(self, paper_id: int) -> ResearchSource:
        """把现有 indexed paper 幂等提升为 research source。"""

        with self._transaction(immediate=True) as connection:
            row = self._ensure_source_for_paper_locked(connection, paper_id)
        return self._source_from_row(row)

    def attach_indexed_paper(
        self,
        source_id: str,
        paper_id: int,
        *,
        expected_version: Optional[int] = None,
    ) -> ResearchSource:
        """把 canonical source 与全文文档/content identity 原子关联。"""

        with self._transaction(immediate=True) as connection:
            source = self._require_row_locked(
                connection, "research_sources", source_id, "研究来源"
            )
            if expected_version is not None and int(source["version"]) != expected_version:
                raise RecordVersionConflictError(
                    f"研究来源 {source_id} 版本冲突：期望 {expected_version}，"
                    f"当前 {source['version']}"
                )
            paper = self._require_row_locked(connection, "papers", paper_id, "索引文档")
            linked = connection.execute(
                "SELECT * FROM research_sources WHERE indexed_paper_id=?",
                (paper_id,),
            ).fetchone()
            identity = connection.execute(
                """
                SELECT rs.* FROM source_identities si
                JOIN research_sources rs ON rs.id=si.source_id
                WHERE si.identity_kind='content_sha256' AND si.normalized_value=?
                """,
                (paper["sha256"],),
            ).fetchone()
            candidates = {
                row["id"]: row
                for row in (source, linked, identity)
                if row is not None
            }
            if linked is not None:
                winner_id = linked["id"]
            elif identity is not None and identity["indexed_paper_id"] is not None:
                winner_id = identity["id"]
            else:
                winner_id = source_id
            for candidate_id in sorted(candidates):
                if candidate_id != winner_id:
                    self._merge_source_locked(connection, winner_id, candidate_id)

            current = connection.execute(
                "SELECT * FROM research_sources WHERE id=?", (winner_id,)
            ).fetchone()
            chosen_paper_id = current["indexed_paper_id"] or paper_id
            chosen_paper = self._require_row_locked(
                connection, "papers", chosen_paper_id, "索引文档"
            )
            existing_identity = connection.execute(
                """
                SELECT source_id FROM source_identities
                WHERE identity_kind='content_sha256' AND normalized_value=?
                """,
                (chosen_paper["sha256"],),
            ).fetchone()
            if existing_identity is None:
                has_identity = connection.execute(
                    "SELECT 1 FROM source_identities WHERE source_id=? LIMIT 1",
                    (winner_id,),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO source_identities(
                        id, source_id, identity_kind, normalized_value,
                        is_primary, created_at
                    ) VALUES (?, ?, 'content_sha256', ?, ?, ?)
                    """,
                    (
                        _new_id("identity"),
                        winner_id,
                        chosen_paper["sha256"],
                        0 if has_identity else 1,
                        _now_iso(),
                    ),
                )
            elif existing_identity["source_id"] != winner_id:
                raise SourceIdentityConflictError(
                    f"content_sha256:{chosen_paper['sha256']} 已属于来源 "
                    f"{existing_identity['source_id']}"
                )
            canonical_key = current["canonical_key"]
            primary = connection.execute(
                """
                SELECT identity_kind, normalized_value FROM source_identities
                WHERE source_id=? AND is_primary=1
                """,
                (winner_id,),
            ).fetchone()
            if primary:
                canonical_key = f"{primary['identity_kind']}:{primary['normalized_value']}"
            connection.execute(
                """
                UPDATE research_sources SET indexed_paper_id=?, content_sha256=?,
                    canonical_key=?, status='ready', updated_at=?, version=version+1
                WHERE id=?
                """,
                (
                    chosen_paper_id,
                    chosen_paper["sha256"],
                    canonical_key,
                    _now_iso(),
                    winner_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM research_sources WHERE id=?", (winner_id,)
            ).fetchone()
        return self._source_from_row(row)

    def add_paper_to_project(
        self,
        project_id: str,
        paper_id: int,
        *,
        position: Optional[int] = None,
        note: str = "",
    ) -> ProjectSourceMembership:
        """在一个事务内提升 indexed paper 并加入 project。"""

        with self._transaction(immediate=True) as connection:
            self._require_row_locked(
                connection, "research_projects", project_id, "研究项目"
            )
            source = self._ensure_source_for_paper_locked(connection, paper_id)
            if position is None:
                position = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(position), -1) + 1
                        FROM project_sources WHERE project_id=?
                        """,
                        (project_id,),
                    ).fetchone()[0]
                )
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("position 必须是非负整数")
            connection.execute(
                """
                INSERT INTO project_sources(project_id, source_id, position, note, added_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, source_id) DO UPDATE SET
                    position=excluded.position, note=excluded.note
                """,
                (project_id, source["id"], position, str(note), _now_iso()),
            )
            row = connection.execute(
                """
                SELECT ps.position, ps.note, ps.added_at, rs.*
                FROM project_sources ps
                JOIN research_sources rs ON rs.id=ps.source_id
                WHERE ps.project_id=? AND ps.source_id=?
                """,
                (project_id, source["id"]),
            ).fetchone()
        return self._membership_from_row(project_id, row)

    def _ensure_source_for_paper_locked(
        self, connection: sqlite3.Connection, paper_id: int
    ) -> sqlite3.Row:
        paper = connection.execute(
            "SELECT * FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
        if paper is None:
            raise KeyError(f"索引论文不存在：{paper_id}")
        existing = connection.execute(
            """
            SELECT rs.* FROM source_identities si
            JOIN research_sources rs ON rs.id=si.source_id
            WHERE si.identity_kind='content_sha256' AND si.normalized_value=?
            """,
            (paper["sha256"],),
        ).fetchone()
        if existing:
            if existing["indexed_paper_id"] is None:
                connection.execute(
                    """
                    UPDATE research_sources
                    SET indexed_paper_id=?, status='ready', updated_at=?, version=version+1
                    WHERE id=?
                    """,
                    (paper_id, _now_iso(), existing["id"]),
                )
                existing = connection.execute(
                    "SELECT * FROM research_sources WHERE id=?", (existing["id"],)
                ).fetchone()
            return existing

        source_id = _new_id("source")
        identity_id = _new_id("identity")
        now = _now_iso()
        connection.execute(
            """
            INSERT INTO research_sources(
                id, canonical_key, source_kind, title, authors, year,
                content_sha256, indexed_paper_id, status, metadata, locator,
                version, created_at, updated_at
            ) VALUES (?, ?, 'paper', ?, ?, ?, ?, ?, 'ready', '{}', '{}', 1, ?, ?)
            """,
            (
                source_id,
                f"content_sha256:{paper['sha256']}",
                paper["title"],
                paper["authors"],
                paper["year"],
                paper["sha256"],
                paper_id,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO source_identities(
                id, source_id, identity_kind, normalized_value, is_primary, created_at
            ) VALUES (?, ?, 'content_sha256', ?, 1, ?)
            """,
            (identity_id, source_id, paper["sha256"], now),
        )
        return connection.execute(
            "SELECT * FROM research_sources WHERE id=?", (source_id,)
        ).fetchone()

    # ---------- project source membership ----------

    def add_project_source(
        self,
        project_id: str,
        source_id: str,
        *,
        position: Optional[int] = None,
        note: str = "",
    ) -> ProjectSourceMembership:
        with self._transaction(immediate=True) as connection:
            self._require_row_locked(
                connection, "research_projects", project_id, "研究项目"
            )
            self._require_row_locked(connection, "research_sources", source_id, "研究来源")
            if position is None:
                position = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(position), -1) + 1
                        FROM project_sources WHERE project_id=?
                        """,
                        (project_id,),
                    ).fetchone()[0]
                )
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("position 必须是非负整数")
            connection.execute(
                """
                INSERT INTO project_sources(project_id, source_id, position, note, added_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, source_id) DO UPDATE SET
                    position=excluded.position, note=excluded.note
                """,
                (project_id, source_id, position, str(note), _now_iso()),
            )
            row = connection.execute(
                """
                SELECT ps.position, ps.note, ps.added_at, rs.*
                FROM project_sources ps
                JOIN research_sources rs ON rs.id=ps.source_id
                WHERE ps.project_id=? AND ps.source_id=?
                """,
                (project_id, source_id),
            ).fetchone()
        return self._membership_from_row(project_id, row)

    def remove_project_source(self, project_id: str, source_id: str) -> bool:
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM project_sources WHERE project_id=? AND source_id=?",
                (project_id, source_id),
            )
        return bool(cursor.rowcount)

    def list_project_sources(
        self,
        project_id: str,
        *,
        q: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[ProjectSourceMembership]:
        limit, offset = _validate_page(limit, offset)
        where = "ps.project_id=?"
        params: list[Any] = [project_id]
        if q:
            where += " AND (rs.title LIKE ? OR rs.authors LIKE ?)"
            like = f"%{q}%"
            params.extend((like, like))
        with self._lock:
            total = int(
                self._conn.execute(
                    f"""
                    SELECT COUNT(*) FROM project_sources ps
                    JOIN research_sources rs ON rs.id=ps.source_id
                    WHERE {where}
                    """,
                    params,
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                f"""
                SELECT ps.position, ps.note, ps.added_at, rs.*
                FROM project_sources ps
                JOIN research_sources rs ON rs.id=ps.source_id
                WHERE {where}
                ORDER BY ps.position, rs.id LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        items = tuple(self._membership_from_row(project_id, row) for row in rows)
        return Page(total, items, limit, offset)

    # ---------- artifacts and immutable revisions ----------

    def create_artifact(
        self,
        project_id: str,
        artifact_type: str,
        *,
        source_id: Optional[str] = None,
        title: str = "",
        status: str = "draft",
        artifact_id: Optional[str] = None,
    ) -> ResearchArtifact:
        artifact_type = _required_text(artifact_type, "artifact_type")
        _validate_choice(status, _ARTIFACT_STATUSES, "status")
        artifact_id = artifact_id or _new_id("artifact")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            self._require_row_locked(
                connection, "research_projects", project_id, "研究项目"
            )
            if source_id is not None:
                membership = connection.execute(
                    """
                    SELECT 1 FROM project_sources
                    WHERE project_id=? AND source_id=?
                    """,
                    (project_id, source_id),
                ).fetchone()
                if membership is None:
                    raise ValueError("artifact 来源必须先加入当前研究项目")
            connection.execute(
                """
                INSERT INTO research_artifacts(
                    id, project_id, source_id, artifact_type, title, status,
                    current_revision_number, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
                """,
                (
                    artifact_id,
                    project_id,
                    source_id,
                    artifact_type,
                    str(title),
                    status,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM research_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
        return self._artifact_from_row(row)

    def get_artifact(self, artifact_id: str) -> Optional[ResearchArtifact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
        return self._artifact_from_row(row) if row else None

    def get_source_artifact(
        self, project_id: str, source_id: str, artifact_type: str
    ) -> Optional[ResearchArtifact]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM research_artifacts
                WHERE project_id=? AND source_id=? AND artifact_type=?
                ORDER BY created_at, id LIMIT 1
                """,
                (project_id, source_id, artifact_type),
            ).fetchone()
        return self._artifact_from_row(row) if row else None

    def list_artifacts(
        self,
        project_id: str,
        *,
        artifact_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[ResearchArtifact]:
        limit, offset = _validate_page(limit, offset)
        where = ["project_id=?"]
        params: list[Any] = [project_id]
        if artifact_type:
            where.append("artifact_type=?")
            params.append(artifact_type)
        if status:
            _validate_choice(status, _ARTIFACT_STATUSES, "status")
            where.append("status=?")
            params.append(status)
        clause = " AND ".join(where)
        with self._lock:
            total = int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM research_artifacts WHERE {clause}", params
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                f"""
                SELECT * FROM research_artifacts WHERE {clause}
                ORDER BY updated_at DESC, id LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return Page(total, tuple(self._artifact_from_row(row) for row in rows), limit, offset)

    def append_artifact_revision(
        self,
        artifact_id: str,
        content: Any,
        *,
        expected_artifact_version: int,
        created_by: str,
        evidence_links: Iterable[tuple[str, str, int]] = (),
        source_fingerprint: Optional[str] = None,
        model: Optional[str] = None,
        usage: Any = None,
        finish_reason: Optional[str] = None,
        prompt_version: Optional[str] = None,
        schema_version: Optional[int] = None,
        status: str = "ready",
        revision_id: Optional[str] = None,
    ) -> ArtifactRevision:
        _validate_choice(created_by, _ARTIFACT_CREATORS, "created_by")
        _validate_choice(status, _ARTIFACT_STATUSES, "status")
        content_json = _json_dump(content)
        usage_json = None if usage is None else _json_dump(usage)
        links = tuple(evidence_links)
        for evidence_id, field_path, ordinal in links:
            _required_text(evidence_id, "evidence_id")
            _required_text(field_path, "field_path")
            if not isinstance(ordinal, int) or ordinal < 0:
                raise ValueError("evidence ordinal 必须是非负整数")
        revision_id = revision_id or _new_id("revision")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            artifact_row = connection.execute(
                "SELECT * FROM research_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if artifact_row is None:
                raise KeyError(f"研究 artifact 不存在：{artifact_id}")
            if int(artifact_row["version"]) != expected_artifact_version:
                raise RecordVersionConflictError(
                    f"研究 artifact {artifact_id} 版本冲突：期望 "
                    f"{expected_artifact_version}，当前 {artifact_row['version']}"
                )
            revision_number = int(artifact_row["current_revision_number"]) + 1
            parent = connection.execute(
                """
                SELECT id FROM artifact_revisions
                WHERE artifact_id=? AND revision_number=?
                """,
                (artifact_id, revision_number - 1),
            ).fetchone()
            if source_fingerprint is None:
                source_fingerprint = self._current_fingerprint_locked(
                    connection, artifact_row
                )
            connection.execute(
                """
                INSERT INTO artifact_revisions(
                    id, artifact_id, revision_number, parent_revision_id, content,
                    created_by, source_fingerprint, model, usage, finish_reason,
                    prompt_version, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    artifact_id,
                    revision_number,
                    parent["id"] if parent else None,
                    content_json,
                    created_by,
                    source_fingerprint,
                    model,
                    usage_json,
                    finish_reason,
                    prompt_version,
                    schema_version,
                    now,
                ),
            )
            seen_links: set[tuple[str, str]] = set()
            for evidence_id, field_path, ordinal in links:
                key = (field_path, evidence_id)
                if key in seen_links:
                    raise ValueError("同一 artifact field 不能重复关联同一 evidence")
                seen_links.add(key)
                if connection.execute(
                    "SELECT 1 FROM evidence WHERE id=?", (evidence_id,)
                ).fetchone() is None:
                    raise KeyError(f"Evidence 不存在：{evidence_id}")
                connection.execute(
                    """
                    INSERT INTO artifact_evidence(
                        artifact_revision_id, evidence_id, field_path, ordinal, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (revision_id, evidence_id, field_path, ordinal, now),
                )
            cursor = connection.execute(
                """
                UPDATE research_artifacts
                SET current_revision_number=?, status=?, updated_at=?, version=version+1
                WHERE id=? AND version=?
                """,
                (
                    revision_number,
                    status,
                    now,
                    artifact_id,
                    expected_artifact_version,
                ),
            )
            if not cursor.rowcount:  # BEGIN IMMEDIATE 下仅防御数据库触发器/损坏
                raise RecordVersionConflictError(f"研究 artifact {artifact_id} 版本冲突")
            row = connection.execute(
                "SELECT * FROM artifact_revisions WHERE id=?", (revision_id,)
            ).fetchone()
        return self._revision_from_row(row)

    def append_validated_deep_read_revision(
        self,
        artifact_id: str,
        content: Any,
        *,
        expected_artifact_version: int,
        expected_source_fingerprint: str,
        created_by: str,
        evidence_refs: Iterable[tuple[str, str, int, str]],
        model: Optional[str],
        usage: Any,
        finish_reason: Optional[str],
        prompt_version: str,
        schema_version: int,
        revision_id: Optional[str] = None,
    ) -> ArtifactRevision:
        """在同一事务验证 Deep Read scope/quote/fingerprint 并保存 revision。"""

        _validate_choice(created_by, {"model", "system"}, "created_by")
        if not isinstance(expected_source_fingerprint, str) or not expected_source_fingerprint:
            raise ArtifactValidationError("必须提供生成开始时的 source fingerprint")
        if not isinstance(model, str) or not model.strip():
            raise ArtifactValidationError("模型 revision 必须保存真实 model metadata")
        if not isinstance(usage, dict):
            raise ArtifactValidationError("模型 revision 必须保存 usage metadata")
        prompt_version = _required_text(prompt_version, "prompt_version")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise ArtifactValidationError("schema_version 必须是正整数")
        content_json = _json_dump(content)
        usage_json = _json_dump(usage)
        refs = tuple(evidence_refs)
        revision_id = revision_id or _new_id("revision")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            artifact = connection.execute(
                "SELECT * FROM research_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if artifact is None:
                raise KeyError(f"研究 artifact 不存在：{artifact_id}")
            if artifact["artifact_type"] != "deep_read" or artifact["source_id"] is None:
                raise ArtifactValidationError("Deep Read revision 必须绑定单一来源")
            if int(artifact["version"]) != expected_artifact_version:
                raise RecordVersionConflictError(
                    f"研究 artifact {artifact_id} 版本冲突：期望 "
                    f"{expected_artifact_version}，当前 {artifact['version']}"
                )
            source = connection.execute(
                """
                SELECT rs.* FROM research_sources rs
                JOIN project_sources ps ON ps.source_id=rs.id
                WHERE ps.project_id=? AND rs.id=?
                """,
                (artifact["project_id"], artifact["source_id"]),
            ).fetchone()
            if source is None or source["indexed_paper_id"] is None:
                raise ArtifactValidationError("Deep Read 来源未加入项目或尚未建立全文索引")
            current_fingerprint = self._current_fingerprint_locked(connection, artifact)
            if current_fingerprint != expected_source_fingerprint:
                raise ArtifactValidationError("来源内容已变化，请重新生成精读卡")

            seen: set[tuple[str, str]] = set()
            validated_links: list[tuple[str, str, int]] = []
            for evidence_id, field_path, ordinal, quote in refs:
                if field_path not in _DEEP_READ_FIELD_PATHS:
                    raise ArtifactValidationError("Deep Read evidence field_path 无效")
                if not isinstance(ordinal, int) or ordinal < 0:
                    raise ArtifactValidationError("Deep Read evidence ordinal 无效")
                key = (field_path, evidence_id)
                if key in seen:
                    raise ArtifactValidationError("同一字段不能重复引用 evidence")
                seen.add(key)
                evidence = connection.execute(
                    "SELECT * FROM evidence WHERE id=?", (evidence_id,)
                ).fetchone()
                if evidence is None:
                    raise ArtifactValidationError("Deep Read 引用了不存在的 evidence")
                if int(evidence["paper_id"] or -1) != int(source["indexed_paper_id"]):
                    raise ArtifactValidationError("Evidence 不属于当前 Deep Read 来源")
                paper = connection.execute(
                    "SELECT * FROM papers WHERE id=?", (source["indexed_paper_id"],)
                ).fetchone()
                chunk = connection.execute(
                    "SELECT * FROM chunks WHERE paper_id=? AND seq=?",
                    (source["indexed_paper_id"], evidence["chunk_seq"]),
                ).fetchone()
                if (
                    paper is None
                    or chunk is None
                    or paper["sha256"] != evidence["paper_sha256"]
                    or int(chunk["page"]) != int(evidence["page"])
                    or hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
                    != evidence["chunk_text_sha256"]
                ):
                    raise ArtifactValidationError("Evidence 已过期，不能保存到新 revision")
                if not isinstance(quote, str) or not quote or quote not in evidence["text"]:
                    raise ArtifactValidationError("Evidence quote 不是原文的精确子串")
                validated_links.append((evidence_id, field_path, ordinal))

            revision_number = int(artifact["current_revision_number"]) + 1
            parent = connection.execute(
                """
                SELECT id FROM artifact_revisions
                WHERE artifact_id=? AND revision_number=?
                """,
                (artifact_id, revision_number - 1),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO artifact_revisions(
                    id, artifact_id, revision_number, parent_revision_id, content,
                    created_by, source_fingerprint, model, usage, finish_reason,
                    prompt_version, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    artifact_id,
                    revision_number,
                    parent["id"] if parent else None,
                    content_json,
                    created_by,
                    current_fingerprint,
                    model.strip(),
                    usage_json,
                    finish_reason,
                    prompt_version,
                    schema_version,
                    now,
                ),
            )
            for evidence_id, field_path, ordinal in validated_links:
                connection.execute(
                    """
                    INSERT INTO artifact_evidence(
                        artifact_revision_id, evidence_id, field_path, ordinal, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (revision_id, evidence_id, field_path, ordinal, now),
                )
            cursor = connection.execute(
                """
                UPDATE research_artifacts
                SET current_revision_number=?, status='ready', updated_at=?, version=version+1
                WHERE id=? AND version=?
                """,
                (revision_number, now, artifact_id, expected_artifact_version),
            )
            if not cursor.rowcount:
                raise RecordVersionConflictError(f"研究 artifact {artifact_id} 版本冲突")
            row = connection.execute(
                "SELECT * FROM artifact_revisions WHERE id=?", (revision_id,)
            ).fetchone()
        return self._revision_from_row(row)

    def get_artifact_revision(self, revision_id: str) -> Optional[ArtifactRevision]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifact_revisions WHERE id=?", (revision_id,)
            ).fetchone()
        return self._revision_from_row(row) if row else None

    def get_current_artifact_revision(
        self, artifact_id: str
    ) -> Optional[ArtifactRevision]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT ar.* FROM research_artifacts a
                JOIN artifact_revisions ar
                  ON ar.artifact_id=a.id
                 AND ar.revision_number=a.current_revision_number
                WHERE a.id=?
                """,
                (artifact_id,),
            ).fetchone()
        return self._revision_from_row(row) if row else None

    def list_artifact_revisions(
        self, artifact_id: str, *, limit: int = 50, offset: int = 0
    ) -> Page[ArtifactRevision]:
        limit, offset = _validate_page(limit, offset)
        with self._lock:
            total = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM artifact_revisions WHERE artifact_id=?",
                    (artifact_id,),
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                """
                SELECT * FROM artifact_revisions WHERE artifact_id=?
                ORDER BY revision_number DESC LIMIT ? OFFSET ?
                """,
                (artifact_id, limit, offset),
            ).fetchall()
        return Page(total, tuple(self._revision_from_row(row) for row in rows), limit, offset)

    def list_artifact_evidence(
        self, revision_id: str
    ) -> tuple[ArtifactEvidenceLink, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM artifact_evidence WHERE artifact_revision_id=?
                ORDER BY field_path, ordinal, evidence_id
                """,
                (revision_id,),
            ).fetchall()
        return tuple(self._artifact_evidence_from_row(row) for row in rows)

    def artifact_freshness(self, artifact_id: str) -> ArtifactFreshness:
        with self._transaction() as connection:
            artifact = connection.execute(
                "SELECT * FROM research_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if artifact is None:
                raise KeyError(f"研究 artifact 不存在：{artifact_id}")
            if int(artifact["current_revision_number"]) == 0:
                return ArtifactFreshness(
                    artifact_id,
                    None,
                    True,
                    None,
                    self._current_fingerprint_locked(connection, artifact),
                    "artifact 尚无 revision",
                )
            revision = connection.execute(
                """
                SELECT * FROM artifact_revisions
                WHERE artifact_id=? AND revision_number=?
                """,
                (artifact_id, artifact["current_revision_number"]),
            ).fetchone()
            if revision is None:
                raise RuntimeError("artifact current revision 指针损坏")
            current = self._current_fingerprint_locked(connection, artifact)
            saved = revision["source_fingerprint"]
            if saved is None:
                reason = "revision 未保存 source fingerprint"
                stale = True
            elif saved != current:
                reason = "项目来源集合或来源内容已变化"
                stale = True
            else:
                reason = None
                stale = False
            return ArtifactFreshness(
                artifact_id=artifact_id,
                revision_id=revision["id"],
                stale=stale,
                saved_fingerprint=saved,
                current_fingerprint=current,
                reason=reason,
            )

    # ---------- notes ----------

    def create_note(
        self,
        project_id: str,
        *,
        scope_kind: str = "project",
        source_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        title: str = "",
        content_markdown: str = "",
        note_id: Optional[str] = None,
    ) -> ResearchNote:
        _validate_choice(scope_kind, _NOTE_SCOPES, "scope_kind")
        _validate_note_scope(scope_kind, source_id, evidence_id)
        note_id = note_id or _new_id("note")
        now = _now_iso()
        with self._transaction(immediate=True) as connection:
            self._require_row_locked(
                connection, "research_projects", project_id, "研究项目"
            )
            if source_id is not None and connection.execute(
                """
                SELECT 1 FROM project_sources WHERE project_id=? AND source_id=?
                """,
                (project_id, source_id),
            ).fetchone() is None:
                raise ValueError("note 来源必须属于当前研究项目")
            if evidence_id is not None:
                self._require_row_locked(connection, "evidence", evidence_id, "Evidence")
            connection.execute(
                """
                INSERT INTO research_notes(
                    id, project_id, scope_kind, source_id, evidence_id, title,
                    content_markdown, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    note_id,
                    project_id,
                    scope_kind,
                    source_id,
                    evidence_id,
                    str(title),
                    str(content_markdown),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM research_notes WHERE id=?", (note_id,)
            ).fetchone()
        return self._note_from_row(row)

    def update_note(
        self,
        note_id: str,
        *,
        expected_version: int,
        title: Any = _UNSET,
        content_markdown: Any = _UNSET,
    ) -> ResearchNote:
        values: dict[str, str] = {}
        if title is not _UNSET:
            values["title"] = str(title)
        if content_markdown is not _UNSET:
            values["content_markdown"] = str(content_markdown)
        if not values:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM research_notes WHERE id=?", (note_id,)
                ).fetchone()
            if row is None:
                raise KeyError(f"研究笔记不存在：{note_id}")
            note = self._note_from_row(row)
            if note.version != expected_version:
                raise RecordVersionConflictError(
                    f"研究笔记 {note_id} 版本冲突：期望 {expected_version}，当前 {note.version}"
                )
            return note
        assignments = [f"{column}=?" for column in values]
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE research_notes
                SET {', '.join(assignments)}, updated_at=?, version=version+1
                WHERE id=? AND version=?
                """,
                (*values.values(), _now_iso(), note_id, expected_version),
            )
            if not cursor.rowcount:
                self._raise_cas_failure_locked(
                    connection,
                    "research_notes",
                    note_id,
                    expected_version,
                    "研究笔记",
                )
            row = connection.execute(
                "SELECT * FROM research_notes WHERE id=?", (note_id,)
            ).fetchone()
        return self._note_from_row(row)

    def list_notes(
        self,
        project_id: str,
        *,
        source_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[ResearchNote]:
        limit, offset = _validate_page(limit, offset)
        where = ["project_id=?"]
        params: list[Any] = [project_id]
        if source_id is not None:
            where.append("source_id=?")
            params.append(source_id)
        if evidence_id is not None:
            where.append("evidence_id=?")
            params.append(evidence_id)
        clause = " AND ".join(where)
        with self._lock:
            total = int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM research_notes WHERE {clause}", params
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                f"""
                SELECT * FROM research_notes WHERE {clause}
                ORDER BY updated_at DESC, id LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return Page(total, tuple(self._note_from_row(row) for row in rows), limit, offset)

    # ---------- internal ----------

    def _current_fingerprint_locked(
        self, connection: sqlite3.Connection, artifact: sqlite3.Row
    ) -> str:
        if artifact["source_id"] is not None:
            source_rows = connection.execute(
                """
                SELECT rs.id, rs.canonical_key, rs.content_sha256,
                       rs.snapshot_sha256, p.sha256 AS paper_sha256
                FROM research_sources rs
                LEFT JOIN papers p ON p.id=rs.indexed_paper_id
                WHERE rs.id=?
                """,
                (artifact["source_id"],),
            ).fetchall()
        else:
            source_rows = connection.execute(
                """
                SELECT rs.id, rs.canonical_key, rs.content_sha256,
                       rs.snapshot_sha256, p.sha256 AS paper_sha256
                FROM project_sources ps
                JOIN research_sources rs ON rs.id=ps.source_id
                LEFT JOIN papers p ON p.id=rs.indexed_paper_id
                WHERE ps.project_id=? ORDER BY rs.id
                """,
                (artifact["project_id"],),
            ).fetchall()
        payload = [
            {
                "source_id": row["id"],
                "canonical_key": row["canonical_key"],
                "content_sha256": row["content_sha256"],
                "snapshot_sha256": row["snapshot_sha256"],
                "paper_sha256": row["paper_sha256"],
            }
            for row in source_rows
        ]
        canonical = _json_dump(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _require_row_locked(
        connection: sqlite3.Connection,
        table: str,
        record_id: Any,
        label: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            f'SELECT * FROM "{table}" WHERE id=?', (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"{label}不存在：{record_id}")
        return row

    @staticmethod
    def _raise_cas_failure_locked(
        connection: sqlite3.Connection,
        table: str,
        record_id: str,
        expected_version: int,
        label: str,
    ) -> None:
        row = connection.execute(
            f'SELECT version FROM "{table}" WHERE id=?', (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"{label}不存在：{record_id}")
        raise RecordVersionConflictError(
            f"{label} {record_id} 版本冲突：期望 {expected_version}，当前 {row['version']}"
        )

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> ResearchProject:
        return ResearchProject(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            default_language=row["default_language"],
            citation_style=row["citation_style"],
            status=row["status"],
            version=int(row["version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _question_from_row(row: sqlite3.Row) -> ResearchQuestion:
        return ResearchQuestion(
            id=row["id"],
            project_id=row["project_id"],
            question=row["question"],
            position=int(row["position"]),
            version=int(row["version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _source_from_row(row: sqlite3.Row) -> ResearchSource:
        return ResearchSource(
            id=row["id"],
            canonical_key=row["canonical_key"],
            source_kind=row["source_kind"],
            title=row["title"],
            authors=tuple(json.loads(row["authors"])),
            year=row["year"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            canonical_url=row["canonical_url"],
            content_sha256=row["content_sha256"],
            indexed_paper_id=row["indexed_paper_id"],
            status=row["status"],
            metadata=json.loads(row["metadata"]),
            locator=json.loads(row["locator"]),
            snapshot_path=row["snapshot_path"],
            snapshot_sha256=row["snapshot_sha256"],
            extracted_text=row["extracted_text"],
            fetched_at=row["fetched_at"],
            version=int(row["version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @classmethod
    def _membership_from_row(
        cls, project_id: str, row: sqlite3.Row
    ) -> ProjectSourceMembership:
        return ProjectSourceMembership(
            project_id=project_id,
            source=cls._source_from_row(row),
            position=int(row["position"]),
            note=row["note"],
            added_at=row["added_at"],
        )

    @staticmethod
    def _identity_from_row(row: sqlite3.Row) -> SourceIdentity:
        return SourceIdentity(
            id=row["id"],
            source_id=row["source_id"],
            identity_kind=row["identity_kind"],
            normalized_value=row["normalized_value"],
            is_primary=bool(row["is_primary"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            id=row["id"],
            source_id=row["source_id"],
            provider=row["provider"],
            provider_record_id=row["provider_record_id"],
            record_url=row["record_url"],
            raw_metadata=json.loads(row["raw_metadata"]),
            retrieved_at=row["retrieved_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> ResearchArtifact:
        return ResearchArtifact(
            id=row["id"],
            project_id=row["project_id"],
            source_id=row["source_id"],
            artifact_type=row["artifact_type"],
            title=row["title"],
            status=row["status"],
            current_revision_number=int(row["current_revision_number"]),
            version=int(row["version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _revision_from_row(row: sqlite3.Row) -> ArtifactRevision:
        return ArtifactRevision(
            id=row["id"],
            artifact_id=row["artifact_id"],
            revision_number=int(row["revision_number"]),
            parent_revision_id=row["parent_revision_id"],
            content=json.loads(row["content"]),
            created_by=row["created_by"],
            source_fingerprint=row["source_fingerprint"],
            model=row["model"],
            usage=None if row["usage"] is None else json.loads(row["usage"]),
            finish_reason=row["finish_reason"],
            prompt_version=row["prompt_version"],
            schema_version=row["schema_version"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _artifact_evidence_from_row(row: sqlite3.Row) -> ArtifactEvidenceLink:
        return ArtifactEvidenceLink(
            artifact_revision_id=row["artifact_revision_id"],
            evidence_id=row["evidence_id"],
            field_path=row["field_path"],
            ordinal=int(row["ordinal"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _note_from_row(row: sqlite3.Row) -> ResearchNote:
        return ResearchNote(
            id=row["id"],
            project_id=row["project_id"],
            scope_kind=row["scope_kind"],
            source_id=row["source_id"],
            evidence_id=row["evidence_id"],
            title=row["title"],
            content_markdown=row["content_markdown"],
            version=int(row["version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _required_text(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} 不能为空")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} 不能为空")
    return text


def _validate_choice(value: str, allowed: frozenset[str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} 必须是以下值之一：{', '.join(sorted(allowed))}")


def _validate_page(limit: int, offset: int) -> tuple[int, int]:
    if not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit 必须是 1–200 的整数")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset 必须是非负整数")
    return limit, offset


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("值必须可序列化为 JSON") from exc


def _validate_note_scope(
    scope_kind: str,
    source_id: Optional[str],
    evidence_id: Optional[str],
) -> None:
    valid = (
        (scope_kind == "project" and source_id is None and evidence_id is None)
        or (scope_kind == "source" and source_id is not None and evidence_id is None)
        or (scope_kind == "evidence" and source_id is None and evidence_id is not None)
    )
    if not valid:
        raise ValueError("note scope 与 source_id/evidence_id 不一致")
