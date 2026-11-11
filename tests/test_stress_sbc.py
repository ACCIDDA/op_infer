"""Stress SBC: an unchanged estimator under a declared perturbed generator."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_infer.sbc import SBCEstimand, SBCFit, run_sbc
from op_infer.stress import StressSpec, stress_verdict
from tests.test_provenance import provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from numpy.typing import ArrayLike

    from op_infer.rng import RandomKey

pytest.importorskip("trade_study")


def _prior(key: RandomKey) -> Mapping[str, ArrayLike]:
    return {"theta": key.numpy_generator().normal()}


def _simulator(
    scale: float,
) -> Callable[[Mapping[str, ArrayLike], RandomKey], ArrayLike]:
    def simulate(truth: Mapping[str, ArrayLike], key: RandomKey) -> ArrayLike:
        noise = scale * key.numpy_generator().normal()
        return np.array([float(np.asarray(truth["theta"])) + noise])

    return simulate


def _fit(data: ArrayLike, key: RandomKey) -> SBCFit:
    """Exact posterior for prior N(0, 1) and unit noise: N(x / 2, 1 / 2)."""
    draws = key.numpy_generator().normal(
        float(np.asarray(data)[0]) / 2.0, np.sqrt(0.5), 1000
    )
    return SBCFit({"theta": draws}, {"adapter": "analytic-normal"})


def _bank(directory: Path, scale: float, n: int, stress: StressSpec | None) -> Any:  # noqa: ANN401
    return run_sbc(
        _prior,
        _simulator(scale),
        _fit,
        estimands={"theta": SBCEstimand(lambda p: p["theta"], prior_sd=1.0)},
        observed=np.array([1000.0]),
        directory=directory,
        replicate_ids=list(range(n)),
        seed=7,
        provenance=provenance(),
        stress=stress,
    )


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _analytic_coverage(scale: float, z: float = 1.6448536269514722) -> float:
    """Coverage of the central 90% interval when the true noise SD is ``scale``.

    Error theta - x / 2 has variance (1 + scale**2) / 4 while the interval half
    width is z * sqrt(1 / 2).
    """
    return 2.0 * _phi(z * math.sqrt(2.0) / math.sqrt(1.0 + scale**2)) - 1.0


def test_known_variance_inflation_is_detected_against_the_paired_control(
    tmp_path: Path,
) -> None:
    """Coverage falls to the analytic value under doubled noise; control holds."""
    control, stress = tmp_path / "control", tmp_path / "stress"
    n = 800
    _bank(control, 1.0, n, None)
    _bank(stress, 2.0, n, StressSpec("noise_x2", "noise doubled", {"scale": 2.0}))
    verdict = stress_verdict(stress, control, ["theta"], level=0.9, floor=0.8, drop=0.1)
    row = verdict["estimands"]["theta"]
    se = math.sqrt(0.9 * 0.1 / n)
    assert abs(row["control"]["estimate"] - 0.9) < 4 * se
    assert abs(row["stress"]["estimate"] - _analytic_coverage(2.0)) < 4 * se
    assert row["stress"]["high"] < 0.8
    assert verdict["passes"] is False
    assert verdict["n_paired"] == n
    assert "not calibration" in verdict["reading"]


def test_unchanged_generator_passes_and_bank_contracts_are_separate(
    tmp_path: Path,
) -> None:
    """A null stress passes; stress and control cannot share a directory."""
    control, same = tmp_path / "control", tmp_path / "same"
    _bank(control, 1.0, 300, None)
    _bank(same, 1.0, 300, StressSpec("noop", "no change"))
    verdict = stress_verdict(same, control, ["theta"], level=0.9, floor=0.8, drop=0.1)
    assert verdict["passes"] is True
    with pytest.raises(ValueError, match="mismatch"):
        _bank(control, 1.0, 300, StressSpec("noop", "no change"))


def _write(directory: Path, name: str, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record))


def test_failures_and_unpaired_replicates_are_counted_not_dropped(
    tmp_path: Path,
) -> None:
    """Pairing is by replicate ID; failed or missing partners are reported."""
    s, c = tmp_path / "s", tmp_path / "c"
    for i in range(10):
        _write(s, f"rep_{i:020d}.json", {"theta": {"pit": 0.5}})
        _write(c, f"rep_{i:020d}.json", {"theta": {"pit": 0.5}})
    _write(s, f"rep_{10:020d}.json", {"theta": {"pit": 0.5}})
    _write(c, f"rep_{3:020d}.json", {"_failed": True})
    v = stress_verdict(s, c, ["theta"], level=0.9, floor=0.8, drop=0.1)
    assert (v["n_paired"], v["n_unpaired"], v["n_failed_control"]) == (9, 1, 1)


def test_randomised_pits_of_a_point_mass_estimand_read_as_calibrated(
    tmp_path: Path,
) -> None:
    """Uniform randomised PITs (an atom) give nominal coverage in both banks."""
    rng = np.random.default_rng(3)
    s, c = tmp_path / "s", tmp_path / "c"
    for i in range(1000):
        _write(s, f"rep_{i:020d}.json", {"atom": {"pit": float(rng.uniform())}})
        _write(c, f"rep_{i:020d}.json", {"atom": {"pit": float(rng.uniform())}})
    v = stress_verdict(s, c, ["atom"], level=0.9, floor=0.8, drop=0.1)
    assert v["passes"] is True
    assert abs(v["estimands"]["atom"]["stress"]["estimate"] - 0.9) < 0.05


def test_reading_rule_has_no_defaults_and_validates_settings(tmp_path: Path) -> None:
    """Thresholds are the caller's explicit choice; bad settings fail early."""
    _write(tmp_path, "rep_0.json", {"t": {"pit": 0.5}})
    with pytest.raises(TypeError):
        stress_verdict(tmp_path, tmp_path, ["t"], level=0.9)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="level"):
        stress_verdict(tmp_path, tmp_path, ["t"], level=1.0, floor=0.8, drop=0.1)
    with pytest.raises(ValueError, match="portable"):
        StressSpec("bad name", "x").record()
