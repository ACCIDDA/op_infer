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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from op_infer.posterior import ArtifactReference

__all__ = ["PredictiveContrast", "UnitPredictiveScores", "paired_predictive_contrast"]


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
