"""Public metadata and deployment boundaries for the unreleased package."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_metadata_keeps_pypi_disabled_and_points_to_docs() -> None:
    """Pages availability must not weaken the no-publication boundary."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "Private :: Do Not Upload" in project["classifiers"]
    assert project["urls"]["Documentation"] == "https://accidda.github.io/op_infer/"


def test_op_extra_uses_released_interface_floors() -> None:
    """A lowest-dependency install must contain both required OP interfaces."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["optional-dependencies"]["op"] == [
        "op_system>=0.10.0",
        "op_engine>=0.9.0",
    ]


def test_pages_workflow_deploys_docs_without_package_publication() -> None:
    """The Pages workflow may deploy documentation, never a distribution."""
    workflow = (ROOT / ".github/workflows/docs-pages.yml").read_text()
    assert "mkdocs build --strict" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "pages: 'write'" in workflow
    assert "pypi" not in workflow.lower()
