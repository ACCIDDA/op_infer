"""Real flepimop2 registration and core SBC/report execution without CLI shells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from flepimop2.process.abc import build
from flepimop2.process.op_infer import InferProcess, SBCWorkflow
from pydantic import ValidationError
from scipy.stats import norm

from op_infer.compare import UnitPredictiveScores
from op_infer.design import (
    DesignDefinition,
    DesignFit,
    DesignGate,
    DesignObservation,
    DesignStudy,
)
from op_infer.posterior import ArtifactReference, Provenance
from op_infer.sbc import SBCEstimand, SBCFit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

    from op_infer.rng import RandomKey


def provenance() -> Provenance:
    """Read a complete unqualified synthetic identity fixture, not consumer data."""
    return Provenance.model_validate_json(
        Path(__file__).with_name("provenance.json").read_text()
    )


def workflow_fixture() -> SBCWorkflow:
    """Return an exact normal-normal posterior consumer for the provider."""

    def prior(key: RandomKey) -> Mapping[str, ArrayLike]:
        return {"theta": key.numpy_generator().normal()}

    def simulate(parameters: Mapping[str, ArrayLike], key: RandomKey) -> ArrayLike:
        return np.asarray(
            [float(np.asarray(parameters["theta"])) + key.numpy_generator().normal()]
        )

    def fit(data: ArrayLike, key: RandomKey) -> SBCFit:
        return SBCFit(
            {
                "theta": key.numpy_generator().normal(
                    np.asarray(data)[0] / 2.0, np.sqrt(0.5), 100
                )
            },
            {"method": "exact-normal"},
        )

    return SBCWorkflow(
        prior,
        simulate,
        fit,
        {"theta": SBCEstimand(lambda p: p["theta"], 1.0)},
        np.array([123.0]),
        provenance(),
    )


def _config(directory: Path) -> dict[str, Any]:
    return {
        "module": "op_infer",
        "operation": "sbc",
        "directory": directory,
        "replicate_ids": [0, 1, 2],
        "seed": 39,
        "factory": "test_provider_scaffold:workflow_fixture",
    }


def test_native_registration_dry_run_and_sbc_resume(tmp_path: Path) -> None:
    """Native discovery executes the actual core runner and preserves artifacts."""
    step = build(_config(tmp_path / "ranks"))
    assert isinstance(step, InferProcess)
    step.execute(dry_run=True)
    assert not (tmp_path / "ranks").exists()
    step.execute()
    records = list((tmp_path / "ranks").glob("*.json"))
    assert len(records) == 3
    before = {p.name: p.read_bytes() for p in records}
    step.execute()
    assert before == {p.name: p.read_bytes() for p in records}
    assert all(not json.loads(p.read_text()).get("_failed", False) for p in records)


def test_calibration_report_keeps_underpowered_label_and_contract(
    tmp_path: Path,
) -> None:
    """A small real bank is underpowered and missing trials cannot vanish."""
    ranks = tmp_path / "ranks"
    build(_config(ranks)).execute()
    report = tmp_path / "report.json"
    config = {
        "module": "op_infer",
        "operation": "calibrate",
        "directory": ranks,
        "replicate_ids": [0, 1, 2],
        "seed": 39,
        "report": report,
        "provenance": provenance(),
    }
    step = build(config)
    step.execute()
    payload = json.loads(report.read_text())
    assert payload["result"]["n_total"] == 3
    assert payload["gate"]["underpowered"] is True
    assert payload["contract"].startswith("sha256:")
    next(ranks.glob("*.json")).unlink()
    with pytest.raises(ValueError, match="complete prespecified"):
        step.execute()
    with pytest.raises(ValidationError, match="outside"):
        build(dict(config, report=ranks / "bad.json"))


def test_dry_run_never_imports_project_factory(tmp_path: Path) -> None:
    """Planning a valid configuration must not run project simulation code."""
    config = _config(tmp_path)
    config["factory"] = "does_not_exist:unavailable"
    build(config).execute(dry_run=True)
    with pytest.raises(ModuleNotFoundError):
        build(config).execute()


def design_fixture() -> object:
    """Return a core design workflow with separated normal training/heldout units."""
    target = ArtifactReference.from_bytes(b"normal held-out", kind="fixture-v1")
    estimator = ArtifactReference.from_bytes(b"conjugate fit", kind="fixture-v1")

    def generate(model: str, parameters: RandomKey, simulation: RandomKey) -> object:
        del parameters
        return (0.0 if model == "a" else 5.0) + simulation.numpy_generator().normal(
            size=8
        )

    def observe(data: object) -> DesignObservation:
        values = np.asarray(data)
        return DesignObservation(values[:2], values[2:], target)

    def fit(model: str, design: str, data: object, key: RandomKey) -> DesignFit:
        del design, key
        mean = 0.0 if model == "a" else 5.0
        return DesignFit(
            (100 * mean + np.asarray(data).sum()) / 102,
            estimator,
            adequate=True,
            diagnostics={},
        )

    def score(
        model: str, fitted: DesignFit, data: object, key: RandomKey
    ) -> UnitPredictiveScores:
        del model, key
        return UnitPredictiveScores(
            norm.logpdf(
                np.asarray(data), loc=fitted.posterior, scale=np.sqrt(1 + 1 / 102)
            ),
            tuple(str(i) for i in range(6)),
            target,
            "independent_unit",
        )

    return DesignStudy(
        ("a", "b"),
        {"small": DesignDefinition(observe, 1)},
        generate,
        fit,
        score,
        {
            ("small", m): DesignGate("small", m, estimator, target, "pass")
            for m in ("a", "b")
        },
        provenance(),
    )


def test_design_process_executes_native_core_workflow(tmp_path: Path) -> None:
    """A registered process writes per-candidate design artifacts through the core."""
    config = _config(tmp_path)
    config["operation"] = "design"
    config["factory"] = "test_provider_scaffold:design_fixture"
    build(config).execute()
    records = list(tmp_path.glob("*.json"))
    assert len(records) == 6
    assert all(json.loads(p.read_text())["outcome"] != "failed" for p in records)
