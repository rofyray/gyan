"""Smoke tests for the initial GYAN package scaffolding."""

from pathlib import Path  # path type for checking configured directories

import gyan  # package import that Stage 1 Task 1.1 requires to work
from gyan import config  # project constants and paths live in this module


def test_package_import_exposes_version() -> None:
    """Confirm the installed package exposes its version string."""
    assert gyan.__version__ == "0.1.0"  # starter package version from src/gyan/__init__.py


def test_project_root_points_to_repository_root() -> None:
    """Confirm config.PROJECT_ROOT resolves from src/gyan/config.py to the repo root."""
    assert isinstance(config.PROJECT_ROOT, Path)  # path constants should be pathlib objects
    assert (config.PROJECT_ROOT / "README.md").exists()  # root should contain the README
    assert (config.PROJECT_ROOT / "PRD" / "CONVENTIONS.md").exists()  # PRDs were relocated
