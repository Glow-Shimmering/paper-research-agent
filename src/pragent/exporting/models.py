"""Immutable inputs and outputs for deterministic artifact rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pragent.models import (
    ArtifactEvidenceLink,
    ArtifactFreshness,
    ArtifactRevision,
    Evidence,
    ResearchArtifact,
    ResearchProject,
    ResearchSource,
    SourceIdentity,
    SourceRecord,
)

EXPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenSource:
    source: ResearchSource
    identities: tuple[SourceIdentity, ...]
    records: tuple[SourceRecord, ...]


@dataclass(frozen=True)
class FrozenEvidence:
    link: ArtifactEvidenceLink
    evidence: Evidence


@dataclass(frozen=True)
class FrozenArtifactExport:
    project: ResearchProject
    artifact: ResearchArtifact
    revision: ArtifactRevision
    freshness: ArtifactFreshness
    sources: tuple[FrozenSource, ...]
    evidence: tuple[FrozenEvidence, ...]

    @property
    def citation_style(self) -> str:
        return self.project.citation_style

    def source_by_id(self) -> dict[str, ResearchSource]:
        return {item.source.id: item.source for item in self.sources}


@dataclass(frozen=True)
class RenderedExport:
    suffix: str
    media_type: str
    data: bytes


@dataclass(frozen=True)
class ExportedFile:
    format: str
    media_type: str
    path: Path
    size: int


class ExportEnvelope(BaseModel):
    """Stable JSON export contract; artifact content stays schema-versioned inside."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=EXPORT_SCHEMA_VERSION, ge=1)
    project: dict[str, Any]
    artifact: dict[str, Any]
    revision: dict[str, Any]
    freshness: dict[str, Any]
    citation_style: str = Field(min_length=1)
    sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
