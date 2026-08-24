"""PRAgent 持久化基础设施。"""

from ._repository import RecordVersionConflictError
from .job_repository import (
    JobIdempotencyConflictError,
    JobRepository,
    JobStateConflictError,
)
from .migrations import (
    LATEST_SCHEMA_VERSION,
    FutureSchemaVersionError,
    InvalidSchemaVersionError,
    MigrationReport,
    SchemaMigrationError,
    migrate_schema,
)
from .research_repository import (
    ArtifactValidationError,
    ResearchRepository,
    SourceIdentityConflictError,
)

__all__ = [
    "ArtifactValidationError",
    "LATEST_SCHEMA_VERSION",
    "JobIdempotencyConflictError",
    "JobRepository",
    "JobStateConflictError",
    "FutureSchemaVersionError",
    "InvalidSchemaVersionError",
    "MigrationReport",
    "RecordVersionConflictError",
    "ResearchRepository",
    "SchemaMigrationError",
    "SourceIdentityConflictError",
    "migrate_schema",
]
