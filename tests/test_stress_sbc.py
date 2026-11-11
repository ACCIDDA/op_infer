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

_NOISE2 = StressSpec("noise_x2", "noise doubled", {"scale": 2.0})


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


def _estimands() -> dict[str, SBCEstimand]:
    return {
        "theta": SBCEstimand(lambda p: p["theta"], prior_sd=1.0),
        # A point mass at zero: an exact atom estimand, randomised PITs.
        "positive_part": SBCEstimand(
            lambda p: np.maximum(p["theta"], 0.0), prior_sd=0.6, atoms=True
        ),
    }


def _bank(
    directory: Path, scale: float, n: int, stress: StressSpec | None, seed: int = 7
) -> Any:  # noqa: ANN401
    return run_sbc(
        _prior,
        _simulator(scale),
        _fit,
        estimands=_estimands(),
        observed=np.array([1000.0]),
        directory=directory,
        replicate_ids=list(range(n)),
        seed=seed,
        provenance=provenance(),
        stress=stress,
    )


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _analytic_coverage(scale: float, z: float = 1.6448536269514722) -> float:
    """Coverage of the central 90% interval when the true noise SD is ``scale``.

    The error theta - x / 2 has variance (1 + scale**2) / 4 while the interval
    half width is z * sqrt(1 / 2).
    """
    return 2.0 * _phi(z * math.sqrt(2.0) / math.sqrt(1.0 + scale**2)) - 1.0


def test_known_variance_inflation_is_detected_against_the_paired_control(
    tmp_path: Path,
) -> None:
    """Coverage falls to the analytic value under doubled noise; control holds."""
    control, stress = tmp_path / "control", tmp_path / "stress"
    n = 800
    _bank(control, 1.0, n, None)
    _bank(stress, 2.0, n, _NOISE2)
    verdict = stress_verdict(stress, control, ["theta"], level=0.9, floor=0.8, drop=0.1)
    row = verdict["estimands"]["theta"]
    truth = _analytic_coverage(2.0)
    assert abs(row["control"]["estimate"] - 0.9) < 4 * math.sqrt(0.9 * 0.1 / n)
    assert abs(row["stress"]["estimate"] - truth) < 4 * math.sqrt(
        truth * (1 - truth) / n
    )
    assert row["stress"]["high"] < 0.8
    assert verdict["passes"] is False
    assert verdict["n_paired"] == n
    assert "not calibration" in verdict["reading"]


def test_atom_estimand_is_read_with_randomised_pits_and_a_null_stress_passes(
    tmp_path: Path,
) -> None:
    """An exact posterior for a clipped estimand is calibrated at its atom."""
    control, same = tmp_path / "control", tmp_path / "same"
    _bank(control, 1.0, 600, None)
    _bank(same, 1.0, 600, StressSpec("noop", "no change"))
    verdict = stress_verdict(
        same, control, ["theta", "positive_part"], level=0.9, floor=0.8, drop=0.1
    )
    assert verdict["passes"] is True
    atom = verdict["estimands"]["positive_part"]
    assert abs(atom["control"]["estimate"] - 0.9) < 4 * math.sqrt(0.9 * 0.1 / 600)
    assert "stress" not in json.loads((control / "sbc-manifest.meta").read_text())
    assert (
        json.loads((same / "sbc-manifest.meta").read_text())["stress"]["name"] == "noop"
    )


def test_banks_must_be_a_stress_and_control_pair(tmp_path: Path) -> None:
    """Different seeds, a missing stress record or the same bank are refused."""
    control, other, stress = (tmp_path / x for x in ("control", "other", "stress"))
    _bank(control, 1.0, 20, None)
    _bank(other, 1.0, 20, None, seed=8)
    _bank(stress, 2.0, 20, _NOISE2)
    args = {"level": 0.9, "floor": 0.8, "drop": 0.1}
    with pytest.raises(ValueError, match="must carry a stress record"):
        stress_verdict(control, control, ["theta"], **args)
    with pytest.raises(ValueError, match="must carry a stress record"):
        stress_verdict(other, control, ["theta"], **args)
    with pytest.raises(ValueError, match="more than the stress record"):
        stress_verdict(stress, other, ["theta"], **args)
    with pytest.raises(ValueError, match="mismatch"):
        _bank(control, 1.0, 20, _NOISE2)


def _copy_bank(source: Path, target: Path, drop: set[int], fail: set[int]) -> None:
    target.mkdir()
    (target / "sbc-manifest.meta").write_bytes(
        (source / "sbc-manifest.meta").read_bytes()
    )
    for file in source.glob("rep_*.json"):
        i = int(file.stem.split("_")[1])
        if i in drop:
            continue
        data = json.loads(file.read_text())
        (target / file.name).write_text(
            json.dumps({"_failed": True} if i in fail else data)
        )


def test_failures_and_unpaired_replicates_are_counted_not_dropped(
    tmp_path: Path,
) -> None:
    """Pairing is by replicate ID; failed or missing partners are reported."""
    stress, control = tmp_path / "stress", tmp_path / "control"
    _bank(stress, 2.0, 12, _NOISE2)
    _bank(tmp_path / "full_control", 1.0, 12, None)
    _copy_bank(tmp_path / "full_control", control, drop={11}, fail={3})
    v = stress_verdict(stress, control, ["theta"], level=0.9, floor=0.0, drop=1.0)
    assert (v["n_paired"], v["n_unpaired"], v["n_failed_control"]) == (10, 1, 1)
    assert (v["n_stress"], v["n_control"]) == (12, 11)


def test_reading_rule_has_no_defaults_and_validates_inputs(tmp_path: Path) -> None:
    """Thresholds are the caller's explicit choice; bad inputs fail early."""
    with pytest.raises(TypeError):
        stress_verdict(tmp_path, tmp_path, ["t"], level=0.9)  # type: ignore[call-arg]
    good = {"level": 0.9, "floor": 0.8, "drop": 0.1}
    for key, bad in (
        ("level", 1.0),
        ("floor", True),
        ("drop", -0.1),
        ("confidence", 0.0),
    ):
        with pytest.raises(ValueError, match=key):
            stress_verdict(tmp_path, tmp_path, ["t"], **{**good, key: bad})
    with pytest.raises(TypeError, match="not a string"):
        stress_verdict(tmp_path, tmp_path, "theta", **good)
    with pytest.raises(ValueError, match="distinct"):
        stress_verdict(tmp_path, tmp_path, ["t", "t"], **good)
    with pytest.raises(ValueError, match="portable"):
        StressSpec("bad name", "x")


def test_missing_estimand_names_the_record(tmp_path: Path) -> None:
    """A wrong estimand name is a ValueError that names the file."""
    control, stress = tmp_path / "control", tmp_path / "stress"
    _bank(control, 1.0, 5, None)
    _bank(stress, 2.0, 5, _NOISE2)
    with pytest.raises(ValueError, match=r"rep_0+0\.json: missing estimand 'nope'"):
        stress_verdict(stress, control, ["nope"], level=0.9, floor=0.8, drop=0.1)
