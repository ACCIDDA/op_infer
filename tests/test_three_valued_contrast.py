"""Three-valued paired contrast: readings, refusals and simulated ties."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_infer.compare import UnitPredictiveScores, three_valued_contrast
from op_infer.posterior import ArtifactReference

if TYPE_CHECKING:
    from numpy.typing import NDArray

TARGET = ArtifactReference.from_bytes(b"units", kind="test-v1")


def _scores(values: list[float] | NDArray[np.float64]) -> UnitPredictiveScores:
    ids = tuple(str(i) for i in range(len(values)))
    return UnitPredictiveScores(list(values), ids, TARGET, "independent_unit")


def test_identical_scores_are_tied_at_zero_floor() -> None:
    """Identical scores (a point mass at zero) read as tied."""
    side = _scores([-1.0, -2.0, -3.0, -4.0])
    result = three_valued_contrast(side, side, floor=0.0, z_multiplicity=2.5)
    assert result.reading == "tied"
    assert result.favours is None
    assert result.tied_bound == 0.0


def test_borderline_band_lies_between_the_two_bounds() -> None:
    """A difference between the two bounds reads as borderline."""
    # d = 1, se = 0.378: tied bound 0.74, separated bound 1.13.
    delta = np.array([0.0, 2.0] * 4)
    result = three_valued_contrast(
        _scores(delta), _scores(np.zeros(8)), floor=0.1, z_multiplicity=3.0
    )
    assert result.se_delta == pytest.approx(math.sqrt(8 / 7) / math.sqrt(8))
    assert result.reading == "borderline"
    assert result.favours == "generator"
    assert result.tied_bound is not None
    assert result.separated_bound is not None
    assert result.tied_bound < 1.0 < result.separated_bound


def test_floor_dominates_and_alternative_is_favoured() -> None:
    """The floor sets both bounds when it exceeds the standard-error terms."""
    gen = _scores([0.0, 0.0, 0.0, 0.0])
    alt = _scores([3.0, 3.1, 2.9, 3.0])
    assert (
        three_valued_contrast(gen, alt, floor=0.1, z_multiplicity=2.5).favours
        == "alternative"
    )
    tied = three_valued_contrast(gen, alt, floor=4.0, z_multiplicity=2.5)
    assert tied.reading == "tied"
    assert tied.tied_bound == tied.separated_bound == 4.0


@pytest.mark.parametrize(
    ("gen", "stable", "reason"),
    [
        ([1.0, math.nan, 2.0], None, "nonfinite units"),
        ([1.0], None, "fewer than two score units"),
        ([5.0, 6.0, 5.5], False, "Monte Carlo stability check failed"),
    ],
)
def test_failures_are_not_assessable_never_tied(
    gen: list[float], *, stable: bool | None, reason: str
) -> None:
    """Non-finite, too few or unstable contrasts are not assessable."""
    result = three_valued_contrast(
        _scores(gen),
        _scores([0.0] * len(gen)),
        floor=0.1,
        z_multiplicity=2.5,
        stable=stable,
    )
    assert result.reading == "not_assessable"
    assert result.reason == reason
    assert result.tied_bound is None
    assert result.audited is (stable is not None)


def test_unstable_identical_scores_are_not_tied() -> None:
    """A failed stability check outranks an apparent tie."""
    side = _scores([1.0, 2.0, 3.0])
    result = three_valued_contrast(
        side, side, floor=0.0, z_multiplicity=2.5, stable=False
    )
    assert result.reading == "not_assessable"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"floor": -0.1, "z_multiplicity": 2.5},
        {"floor": math.nan, "z_multiplicity": 2.5},
        {"floor": 0.1, "z_multiplicity": 1.5},
        {"floor": 0.1, "z_multiplicity": math.inf},
        {"floor": 0.1, "z_multiplicity": 2.5, "z_base": 0.0},
        {"floor": True, "z_multiplicity": 2.5},
    ],
)
def test_invalid_rule_is_refused(kwargs: dict[str, float]) -> None:
    """Invalid floors and quantiles are refused."""
    side = _scores([1.0, 2.0])
    with pytest.raises(ValueError, match=r"floor|z_"):
        three_valued_contrast(side, side, **kwargs)  # type: ignore[arg-type]


def test_simulated_ties_are_rarely_separated() -> None:
    """Under an exact tie the separated rate stays near its nominal level."""
    rng = np.random.default_rng(20261111)
    n_rep, n_units, z_k = 4000, 60, 2.5
    readings = []
    for _ in range(n_rep):
        shared = rng.normal(-2.0, 1.0, n_units)
        gen = shared + rng.normal(0.0, 0.3, n_units)
        alt = shared + rng.normal(0.0, 0.3, n_units)
        readings.append(
            three_valued_contrast(
                _scores(gen), _scores(alt), floor=0.0, z_multiplicity=z_k
            ).reading
        )
    counts = {r: readings.count(r) for r in ("separated", "borderline", "tied")}
    nominal = 2 * (1 - 0.5 * math.erfc(-z_k / math.sqrt(2)))
    rate = counts["separated"] / n_rep
    # Four binomial standard deviations above the nominal rate.
    assert rate <= nominal + 4 * math.sqrt(nominal * (1 - nominal) / n_rep)
    assert counts["tied"] / n_rep > 0.9


@pytest.mark.parametrize(
    "stable", ["stable", "not_assessable", {"label": "not_assessable"}, 0, 1]
)
def test_invalid_stability_result_is_refused(stable: object) -> None:
    """Audit records and truthy labels cannot bypass the stability check."""
    side = _scores([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="stable must be a bool or None"):
        three_valued_contrast(
            side,
            side,
            floor=0.0,
            z_multiplicity=2.5,
            stable=stable,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("label", ["stable", "not_assessable"])
def test_audit_label_is_explicitly_converted(label: str) -> None:
    """The documented Boolean conversion preserves a failed audit."""
    audit_result = {"label": label}
    side = _scores([1.0, 2.0, 3.0])
    result = three_valued_contrast(
        side,
        side,
        floor=0.0,
        z_multiplicity=2.5,
        stable=(audit_result["label"] == "stable"),
    )
    assert result.audited
    assert result.reading == ("tied" if label == "stable" else "not_assessable")
