"""FastAPI UI/API route registration."""

from .artifacts import register_artifact_routes
from .discovery import register_discovery_routes
from .projects import register_project_routes

__all__ = [
    "register_artifact_routes",
    "register_discovery_routes",
    "register_project_routes",
]
