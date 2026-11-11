"""Unit-level predictive comparison.

Implemented: a bounded paired held-out contrast for DesignStudy.
LOO/PSIS recovery and stratified comparisons remain planned.

Scope: Exact leave-one-unit-out (re-conditioning or refit); unit-level PSIS-LOO with
explicit adequacy and recovery records; paired contrasts with failure accounting;
stratification; three-way outcomes and pre-declared reading rules. Scoring rules and
paired bootstraps come from trade_study.


Predictive comparisons follow ``docs/design/likelihood-targets.md``. Raw importance
sampling after failed PSIS is diagnostic only until independently qualified; it cannot
erase the failure. Future comparison artifacts retain both target identities, excluded
and non-finite unit counts, adequacy failures and the conditional paired estimand with
its actual denominator. The held-out API below does not implement LOO/PSIS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from op_infer.posterior import ArtifactReference

__all__ = [
    "PredictiveContrast",
    "ThreeValuedContrast",
    "UnitPredictiveScores",
    "paired_predictive_contrast",
    "three_valued_contrast",
]


@dataclass(frozen=True)
class UnitPredictiveScores:
    """Held-out log predictive densities on declared independent units/clusters.

    Values come from a proper-scoring callback, not from this module. The target
    identifies the same held-out observable/window across models and designs.
    Shared latent parameters and numerical/Monte Carlo approximations belong in
    the surrounding provenance and qualification.
    """

    values: ArrayLike
    unit_ids: tuple[str, ...]
    target: ArtifactReference
    unit_kind: Literal["independent_unit", "independent_cluster"]


@dataclass(frozen=True)
class PredictiveContrast:
    """Prespecified paired reading rule, never a certified hypothesis test."""

    outcome: Literal["generator", "alternative", "indistinguishable", "failed"]
    mean_delta: float | None
    se_delta: float | None
    n_units: int
    nonfinite_units: int
    reason: str | None = None


def paired_predictive_contrast(
    generator: UnitPredictiveScores,
    alternative: UnitPredictiveScores,
    *,
    threshold_se: float = 2.0,
) -> PredictiveContrast:
    """Apply a fixed paired ELPD reading rule with explicit failed-unit accounting.

    No unit is discarded. A nonfinite unit makes the whole
    comparison unavailable. The threshold is a decision rule, not a claim of
    exact testing level; repeated independent cohorts qualify its recovery rate.

    Returns:
        Generator/alternative/indistinguishable, or a failed comparison.

    Raises:
        ValueError: Units, targets, independence assumptions or controls disagree.

    Examples:
        >>> from op_infer.posterior import ArtifactReference
        >>> target = ArtifactReference.from_bytes(
        ...     b"held-out units", kind="example-v1"
        ... )
        >>> generator = UnitPredictiveScores(
        ...     [-1.0, -2.0, -3.0], ("a", "b", "c"), target, "independent_unit"
        ... )
        >>> alternative = UnitPredictiveScores(
        ...     [-5.0, -3.0, -4.0], ("c", "a", "b"), target, "independent_unit"
        ... )
        >>> contrast = paired_predictive_contrast(generator, alternative)
        >>> contrast.outcome, contrast.mean_delta, contrast.se_delta
        ('generator', 2.0, 0.0)
        >>> paired_predictive_contrast(generator, generator).outcome
        'indistinguishable'
    """
    if not np.isfinite(threshold_se) or threshold_se <= 0:
        msg = "threshold_se must be finite and positive"
        raise ValueError(msg)
    if (
        generator.target != alternative.target
        or generator.unit_kind != alternative.unit_kind
    ):
        msg = (
            "Predictive comparisons require identical targets and score-unit semantics"
        )
        raise ValueError(msg)
    if generator.unit_kind not in {"independent_unit", "independent_cluster"}:
        msg = "Score-unit independence must be explicitly declared"
        raise ValueError(msg)
    ids = generator.unit_ids
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(ids) != set(alternative.unit_ids)
        or len(set(alternative.unit_ids)) != len(alternative.unit_ids)
    ):
        msg = "Predictive score units must be distinct and match across models"
        raise ValueError(msg)
    left = np.asarray(generator.values, dtype=float)
    right = np.asarray(alternative.values, dtype=float)
    if left.shape != (len(ids),) or right.shape != (len(ids),):
        msg = "One held-out score is required per declared unit"
        raise ValueError(msg)
    positions = {name: i for i, name in enumerate(alternative.unit_ids)}
    right = right[[positions[name] for name in ids]]
    bad = int(np.sum(~np.isfinite(left) | ~np.isfinite(right)))
    if bad or len(ids) < 2:
        return PredictiveContrast(
            "failed",
            None,
            None,
            len(ids),
            bad,
            "nonfinite units" if bad else "fewer than two score units",
        )
    with np.errstate(over="ignore", invalid="ignore"):
        delta = left - right
        mean = float(delta.mean())
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    if not np.isfinite(mean) or not np.isfinite(se):
        return PredictiveContrast(
            "failed", None, None, len(ids), 0, "nonfinite contrast arithmetic"
        )
    outcome: Literal["generator", "alternative", "indistinguishable"] = (
        "generator"
        if mean > threshold_se * se
        else "alternative"
        if mean < -threshold_se * se
        else "indistinguishable"
    )
    return PredictiveContrast(outcome, mean, se, len(ids), 0)


@dataclass(frozen=True)
class ThreeValuedContrast:
    """A paired contrast read as separated, borderline, tied or not assessable.

    ``tied_bound`` and ``separated_bound`` are the two fixed bounds on the
    absolute mean difference: at or below ``tied_bound`` the contrast is tied,
    above ``separated_bound`` it is separated, and between them it is
    borderline. Both are ``None`` when the contrast is not assessable.
    ``favours`` names the side with the larger mean score for separated and
    borderline readings.
    """

    reading: Literal["separated", "borderline", "tied", "not_assessable"]
    favours: Literal["generator", "alternative"] | None
    mean_delta: float | None
    se_delta: float | None
    n_units: int
    nonfinite_units: int
    tied_bound: float | None
    separated_bound: float | None
    audited: bool
    reason: str | None = None


def three_valued_contrast(  # noqa: PLR0913 - explicit reading rule
    generator: UnitPredictiveScores,
    alternative: UnitPredictiveScores,
    *,
    floor: float,
    z_multiplicity: float,
    z_base: float = 1.96,
    stable: bool | None = None,
) -> ThreeValuedContrast:
    """Read a paired held-out contrast with a floor and a multiplicity bound.

    The mean paired difference ``d`` and its standard error ``se`` come from
    `paired_predictive_contrast`. The reading rule is fixed before the contrast
    is seen: the contrast is **separated** when ``|d| > max(floor,
    z_multiplicity * se)``, **tied** when ``|d| <= max(floor, z_base * se)``,
    and **borderline** in between. ``z_multiplicity`` is the caller's quantile
    for the number of registered contrasts (for example the two-sided
    ``1 - 0.05 / (2 K)`` normal quantile for ``K`` contrasts), so it must not
    be smaller than ``z_base``. When ``floor`` dominates both products the
    borderline band is empty.

    The normal approximation to a predictive-score difference is poorly
    calibrated for similar predictions, under misspecification with outliers
    and for small numbers of units (Sivula, Magnusson, Matamoros and Vehtari
    2025, arXiv:2008.10296). Two consequences are built in. A contrast whose
    Monte Carlo stability check failed (``stable=False``, for example from
    `op_infer.audit.stability_label`), or whose scores are non-finite, or that
    has fewer than two units, is **not assessable**: it is never reported as
    tied, because a failed check carries no evidence of equivalence. Tied means
    only that the difference is within the fixed bounds, not that the models
    are equivalent.

    Args:
        generator: Held-out unit scores of the generating-model side.
        alternative: Held-out unit scores of the alternative on the same units.
        floor: Smallest absolute mean difference worth reading, in score units
            (nonnegative).
        z_multiplicity: Multiplicity-adjusted standard-error multiple that
            defines the separated bound.
        z_base: Unadjusted standard-error multiple that defines the tied bound.
        stable: Result of a caller-run Monte Carlo stability check, or ``None``
            when no audit was run (recorded in ``audited``).

    Returns:
        The reading with both bounds and the unit accounting.

    Raises:
        ValueError: Units or targets disagree, or ``floor``, ``z_base`` or
            ``z_multiplicity`` is invalid.

    Examples:
        >>> from op_infer.posterior import ArtifactReference
        >>> target = ArtifactReference.from_bytes(b"units", kind="example-v1")
        >>> ids = tuple(str(i) for i in range(8))
        >>> left = UnitPredictiveScores(
        ...     [1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5],
        ...     ids,
        ...     target,
        ...     "independent_unit",
        ... )
        >>> right = UnitPredictiveScores([0.0] * 8, ids, target, "independent_unit")
        >>> three_valued_contrast(
        ...     left, right, floor=0.1, z_multiplicity=2.5
        ... ).reading
        'separated'
        >>> three_valued_contrast(
        ...     left, right, floor=5.0, z_multiplicity=2.5
        ... ).reading
        'tied'
        >>> three_valued_contrast(
        ...     left, right, floor=0.1, z_multiplicity=2.5, stable=False
        ... ).reading
        'not_assessable'
    """
    if isinstance(floor, bool) or not math.isfinite(floor) or floor < 0:
        msg = "floor must be a finite nonnegative number"
        raise ValueError(msg)
    if isinstance(z_base, bool) or not math.isfinite(z_base) or z_base <= 0:
        msg = "z_base must be finite and positive"
        raise ValueError(msg)
    if (
        isinstance(z_multiplicity, bool)
        or not math.isfinite(z_multiplicity)
        or z_multiplicity < z_base
    ):
        msg = "z_multiplicity must be finite and at least z_base"
        raise ValueError(msg)
    base = paired_predictive_contrast(generator, alternative)
    audited = stable is not None
    if base.outcome == "failed" or base.mean_delta is None or base.se_delta is None:
        return ThreeValuedContrast(
            "not_assessable",
            None,
            None,
            None,
            base.n_units,
            base.nonfinite_units,
            None,
            None,
            audited,
            base.reason,
        )
    if stable is False:
        return ThreeValuedContrast(
            "not_assessable",
            None,
            base.mean_delta,
            base.se_delta,
            base.n_units,
            0,
            None,
            None,
            audited,
            "Monte Carlo stability check failed",
        )
    tied_bound = max(floor, z_base * base.se_delta)
    separated_bound = max(floor, z_multiplicity * base.se_delta)
    size = abs(base.mean_delta)
    reading: Literal["separated", "borderline", "tied"] = (
        "separated"
        if size > separated_bound
        else "tied"
        if size <= tied_bound
        else "borderline"
    )
    favours: Literal["generator", "alternative"] | None = (
        None
        if reading == "tied"
        else "generator"
        if base.mean_delta > 0
        else "alternative"
    )
    return ThreeValuedContrast(
        reading,
        favours,
        base.mean_delta,
        base.se_delta,
        base.n_units,
        0,
        tied_bound,
        separated_bound,
        audited,
    )
