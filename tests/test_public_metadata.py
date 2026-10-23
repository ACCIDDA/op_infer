"""Public metadata and deployment boundaries for the unreleased package."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_metadata_keeps_pypi_disabled_and_points_to_docs() -> None:
    """Pages availability must not weaken the no-publication boundary."""
    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    provider = tomllib.loads((ROOT / "flepimop2-op_infer/pyproject.toml").read_text())[
        "project"
    ]
    assert "Private :: Do Not Upload" in core["classifiers"]
    assert "Private :: Do Not Upload" in provider["classifiers"]
    assert core["urls"]["Documentation"] == "https://accidda.github.io/op_infer/"


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
    assert "workflow_dispatch" not in workflow
    assert re.search(r"push:\s+branches:\s+- 'main'", workflow)
    assert "contents: 'read'" in workflow
    assert "mkdocs build --strict" in workflow
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "pages: 'write'" in workflow
    assert "id-token: 'write'" in workflow
    forbidden = ("pypi", "uv publish", "twine", "gh release")
    assert not any(command in workflow.lower() for command in forbidden)
