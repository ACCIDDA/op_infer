"""Actual TARP reference calls, failed rows, explicit coordinates and RNG isolation."""

from __future__ import annotations

import importlib
import json
import pickle
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_infer.adapters.tarp import tarp_coverage
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytest.importorskip("tarp")
tarp = importlib.import_module("tarp")
DEFINITION = reference(
    "normal-normal; ordered reported coordinates; x-dependent references"
)


def inputs() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate exact conjugate posterior draws and explicit data-dependent points."""
    rng = np.random.default_rng(824)
    truth = rng.normal(size=(600, 2))
    observed = truth + rng.normal(size=truth.shape)
    samples = rng.normal(observed / 2, np.sqrt(0.5), size=(256, *truth.shape))
    points = observed + rng.normal(scale=2, size=truth.shape)
    return samples, truth, points


def report(  # noqa: PLR0913 - exercise independent public metadata guards
    samples: NDArray[np.generic],
    truth: NDArray[np.generic],
    points: NDArray[np.generic],
    *,
    unavailable: dict[int, str] | None = None,
    num_bins: int = 20,
    replicate_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Use complete ordered row IDs and quantities at the public boundary."""
    return tarp_coverage(
        samples,
        truth,
        references=points,
        replicate_ids=tuple(range(truth.shape[0], 0, -1))
        if replicate_ids is None
        else replicate_ids,
        quantities=("z", "a"),
        definition=DEFINITION,
        provenance=provenance(),
        num_bins=num_bins,
        unavailable=unavailable,
    )


def test_named_actual_reference_and_input_identity() -> None:
    """Public reference values agree with all original coordinates retained."""
    samples, truth, points = inputs()
    result = report(samples, truth, points)
    state = np.random.get_state(legacy=False)  # noqa: NPY002 - reference mutates this API
    try:
        ecp, alpha = tarp.get_tarp_coverage(
            samples,
            truth,
            references=points,
            num_alpha_bins=20,
            norm=False,
            bootstrap=False,
            seed=0,
        )
    finally:
        np.random.set_state(state)  # noqa: NPY002 - restore caller's state
    curve = result["curve"]
    assert isinstance(curve, dict)
    np.testing.assert_array_equal(curve["expected_coverage"], ecp)
    np.testing.assert_array_equal(curve["credibility"], alpha)
    assert result["quantities"] == ["z", "a"]
    rows = result["replicates"]
    assert isinstance(rows, list)
    assert [r["replicate_id"] for r in rows] == list(range(600, 0, -1))
    assert result["n_total"] == result["n_usable"] == 600
    assert result["n_failed"] == 0
    assert result["adequacy_status"] == "not_assessed"
    assert result["definition"] == DEFINITION.model_dump(mode="json")
    assert result["provenance"] == provenance().model_dump(mode="json")
    assert result["implementation"] == "tarp.get_tarp_coverage"
    assert result["backend_versions"] is not None
    before = result["inputs"]
    samples[:] += 3
    assert report(samples, truth, points)["inputs"] != before
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_calibrated_and_biased_reference_curves() -> None:
    """The conjugate fixture stays near the diagonal; mean bias changes its curve."""
    samples, truth, points = inputs()
    good, bad = report(samples, truth, points), report(samples + 2, truth, points)
    good_curve, bad_curve = good["curve"], bad["curve"]
    assert isinstance(good_curve, dict)
    assert isinstance(bad_curve, dict)
    # Regression bounds on this fixture, not a confidence band or scientific gate.
    assert good_curve["max_abs_deviation"] < 0.08
    assert bad_curve["max_abs_deviation"] > 0.2


def test_observed_rank_histogram_edges_are_preserved() -> None:
    """The released reference need not return grid endpoints of zero and one."""
    samples = np.broadcast_to(np.arange(1.0, 11)[:, None, None], (10, 2, 1))
    result = tarp_coverage(
        samples,
        np.array([[2.5], [7.5]]),
        references=np.zeros((2, 1)),
        replicate_ids=(7, 1),
        quantities=("theta",),
        definition=DEFINITION,
        provenance=provenance(),
        num_bins=4,
    )
    assert result["status"] == "available"
    assert result["curve"]["credibility"][0] == pytest.approx(0.2)
    assert result["curve"]["credibility"][-1] == pytest.approx(0.7)


def test_nonfinite_and_external_failures_retain_planned_denominator() -> None:
    """Every failed/inadequate row remains identified in the conditional report."""
    samples, truth, points = inputs()
    samples[0, 1, 0] = np.nan
    truth[3, 1] = np.inf
    points[5, 0] = -np.inf
    result = report(samples, truth, points, unavailable={593: "sampler_inadequate"})
    assert result["status"] == "available"
    assert (result["n_total"], result["n_usable"], result["n_failed"]) == (600, 596, 4)
    assert result["failed_fraction"] == pytest.approx(4 / 600)
    rows = result["replicates"]
    assert isinstance(rows, list)
    failed = [r for r in rows if r["status"] == "unavailable"]
    assert [r["replicate_id"] for r in failed] == [599, 597, 595, 593]
    assert [r["nonfinite_values"] for r in failed] == [1, 1, 1, 0]
    assert failed[-1]["reasons"] == ["sampler_inadequate"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    samples[:] = np.nan
    all_failed = report(samples, truth, points)
    assert all_failed["n_total"] == all_failed["n_failed"] == 600
    assert all_failed["n_usable"] == 0
    assert all_failed["status"] == "unavailable"
    assert all_failed["curve"] is None
    assert all_failed["reason"] == "no_usable_replicates"


def test_reference_call_preserves_complete_legacy_rng_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend seeding cannot consume or replace the caller's Gaussian-cache state."""
    original = np.random.get_state(legacy=False)  # noqa: NPY002 - test legacy boundary
    try:
        np.random.seed(827)  # noqa: NPY002 - prime a caller's legacy stream
        np.random.normal()  # noqa: NPY002 - leave a Gaussian value cached
        before = pickle.dumps(np.random.get_state(legacy=False))  # noqa: NPY002
        samples, truth, points = inputs()
        assert report(samples, truth, points)["status"] == "available"
        assert pickle.dumps(np.random.get_state(legacy=False)) == before  # noqa: NPY002

        def broken(*_args: object, **_kwargs: object) -> object:
            np.random.seed(9)  # noqa: NPY002 - emulate the reference failure path
            msg = "synthetic reference failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(tarp, "get_tarp_coverage", broken)
        result = report(samples, truth, points)
        assert result["curve"] is None
        assert "reference_error:RuntimeError" in str(result["reason"])
        assert result["n_total"] == result["n_usable"] == 600
        assert pickle.dumps(np.random.get_state(legacy=False)) == before  # noqa: NPY002
    finally:
        np.random.set_state(original)  # noqa: NPY002 - restore the caller


def test_strict_distance_ties_and_overflow_cannot_supply_a_curve() -> None:
    """The continuous-rank reference cannot silently claim atom-aware behavior."""
    samples, truth, points = inputs()
    samples[:, 0] = truth[0]
    result = report(samples, truth, points)
    assert result["curve"] is None
    assert "contain ties" in str(result["reason"])
    samples, truth, points = inputs()
    samples[0, 0, 0] = 1e308
    result = report(samples, truth, points)
    assert result["curve"] is None
    assert "FloatingPointError" in str(result["reason"])
    assert result["n_total"] == 600


@pytest.mark.parametrize("dtype", ["float16", "int64", "complex128"])
def test_unsupported_host_types_are_explicit(dtype: str) -> None:
    """No precision promotion or unsupported dtype coercion is implicit."""
    samples, truth, points = inputs()
    with pytest.raises(TypeError, match="NumPy float32/float64"):
        report(samples.astype(dtype), truth, points)


def test_shape_coordinate_exclusion_and_bin_guards() -> None:
    """Layouts and exclusion reasons are checked before the reference can run."""
    samples, truth, points = inputs()
    for selected in ((), ("z", "z"), ("z",), "za"):
        with pytest.raises(ValueError, match="original sample shapes"):
            tarp_coverage(
                samples,
                truth,
                references=points,
                replicate_ids=tuple(range(600)),
                quantities=selected,
                definition=DEFINITION,
                provenance=provenance(),
                num_bins=20,
            )
    exclusions: tuple[dict[int, str], ...] = (
        {999: "unknown"},
        {1: ""},
        {True: "boolean ID"},
    )
    for excluded in exclusions:
        with pytest.raises(ValueError, match="declared IDs"):
            report(samples, truth, points, unavailable=excluded)
    with pytest.raises(ValueError, match="original sample shapes"):
        report(samples.transpose(1, 0, 2), truth, points)
    for ids in ((0,) * 600, tuple(range(-600, 0)), (True, *range(2, 601))):
        with pytest.raises(ValueError, match="original sample shapes"):
            report(samples, truth, points, replicate_ids=ids)


@pytest.mark.parametrize("bins", [0, -1, True, 2.5])
def test_bin_count_must_be_a_positive_integer(bins: object) -> None:
    """Invalid grids cannot silently select a reference default."""
    samples, truth, points = inputs()
    with pytest.raises(ValueError, match="num_bins"):
        report(samples, truth, points, num_bins=cast("int", bins))


def test_invalid_reference_curve_stays_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed/nonfinite reference output cannot become a plotted diagnostic."""
    samples, truth, points = inputs()

    def invalid(*_args: object, **_kwargs: object) -> object:
        return np.linspace(-1, 1, 21), np.linspace(0, 1, 21)

    monkeypatch.setattr(tarp, "get_tarp_coverage", invalid)
    result = report(samples, truth, points)
    assert result["status"] == "unavailable"
    assert "invalid coverage curve" in str(result["reason"])
