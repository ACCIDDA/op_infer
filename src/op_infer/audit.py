"""Blind stratified Monte Carlo stability audit for sampled per-unit scores.

A contrast between two models is a sum over units of per-unit predictive score
differences, each estimated by Monte Carlo (often importance sampling with
heavy-tailed weights). Before the contrast is read, its reproducibility can be
checked on a sample of units: choose the sample from tail diagnostics only,
re-score it with independent random keys and larger Monte Carlo settings, and
estimate the shift and the Monte Carlo error of the total over all units (see
Vehtari, Gelman and Gabry 2017, Statistics and Computing 27:1413, and Sivula,
Magnusson, Matamoros and Vehtari 2025, Bayesian Analysis, on the uncertainty of
elpd differences).

The sampler takes diagnostics only; it has no argument through which a score
could enter (enforced by its signature). Thresholds of the stability rule are the
caller's reading-rule choices and have no defaults.

The estimates assume finite-variance, independent per-unit Monte Carlo errors
and independent random keys in the two scorings. Correlated errors, shared keys
(the noise cancels in the difference) and heavy-tailed importance-sampling errors
(Pareto shape above 0.5, where sample variances are unstable and usually too
small) can make ``mc_sd`` understate the noise. A stable label means the total is
reproducible against an independent rerun, not that it is accurate: a bias that
both scorings share is invisible to the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


def stratify(diagnostic: ArrayLike, edges: Sequence[float]) -> NDArray[np.intp]:
    """Assign each unit to a stratum by a tail diagnostic.

    Stratum ``k`` holds values from ``edges[k-1]`` (inclusive) up to ``edges[k]``
    (exclusive); the last stratum is open above. A non-finite diagnostic is
    assigned to the **top** stratum, never dropped.

    Raises:
        ValueError: ``edges`` is empty, not finite or not strictly increasing.

    Examples:
        >>> stratify([0.1, 0.5, 0.69, 0.7, 1.2, float("nan")], (0.5, 0.7)).tolist()
        [0, 1, 1, 2, 2, 2]
    """
    cuts = np.asarray(edges, dtype=float)
    if cuts.ndim != 1 or cuts.size == 0 or not np.all(np.isfinite(cuts)):
        msg = "edges must be a nonempty finite sequence"
        raise ValueError(msg)
    if np.any(np.diff(cuts) <= 0):
        msg = "edges must be strictly increasing"
        raise ValueError(msg)
    values = np.asarray(diagnostic, dtype=float)
    labels = np.searchsorted(cuts, values, side="right").astype(np.intp)
    labels[~np.isfinite(values)] = cuts.size
    return labels


@dataclass(frozen=True)
class AuditSample:
    """The audited units, the stratum of each, and the stratum population sizes.

    Examples:
        >>> s = audit_sample(
        ...     [0.1, 0.2, 0.6, 0.9, 1.1, 1.3], edges=(0.5,), per_stratum=1, seed=3
        ... )
        >>> s.sizes, s.taken
        ((2, 4), (1, 1))
    """

    units: tuple[int, ...]
    strata: tuple[int, ...]
    sizes: tuple[int, ...]
    taken: tuple[int, ...]
    seed: int
    edges: tuple[float, ...]


def audit_sample(
    diagnostic: ArrayLike,
    *,
    edges: Sequence[float],
    per_stratum: int | np.integer[Any],
    seed: int,
) -> AuditSample:
    """Draw a stratified sample of units from a tail diagnostic only.

    Each stratum contributes ``min(per_stratum, size)`` units, chosen uniformly
    without replacement; a stratum smaller than ``per_stratum`` is taken in full.

    Raises:
        ValueError: ``per_stratum`` is not a positive integer or the diagnostic
            is empty.

    Examples:
        >>> s = audit_sample(
        ...     np.linspace(0, 2, 50), edges=(0.5, 1.0), per_stratum=5, seed=1
        ... )
        >>> s.taken, len(s.units) == len(set(s.units))
        ((5, 5, 5), True)
    """
    if isinstance(per_stratum, bool) or not isinstance(per_stratum, Integral):
        msg = "per_stratum must be a positive integer"
        raise ValueError(msg)  # noqa: TRY004 - value contract
    if per_stratum < 1:
        msg = "per_stratum must be a positive integer"
        raise ValueError(msg)
    values = np.asarray(diagnostic, dtype=float)
    if values.ndim != 1 or values.size == 0:
        msg = "the diagnostic must be a nonempty one-dimensional array"
        raise ValueError(msg)
    per_stratum = int(per_stratum)
    labels = stratify(values, edges)
    rng = np.random.default_rng(seed)
    units: list[int] = []
    strata: list[int] = []
    sizes: list[int] = []
    taken: list[int] = []
    for k in range(len(tuple(edges)) + 1):
        members = np.flatnonzero(labels == k)
        n = min(per_stratum, members.size)
        pick = np.sort(rng.permutation(members)[:n])
        units += [int(i) for i in pick]
        strata += [k] * n
        sizes.append(int(members.size))
        taken.append(int(n))
    return AuditSample(
        tuple(units),
        tuple(strata),
        tuple(sizes),
        tuple(taken),
        seed,
        tuple(float(e) for e in edges),
    )


@dataclass(frozen=True)
class StabilityEstimate:
    """Monte Carlo noise of a total and the shift between the two scorings.

    ``mc_sd`` is the conservative standard deviation of the Monte Carlo noise of
    the total over all units; ``shift`` is the estimated difference between the
    audit total and the original total, with its sampling standard error
    ``shift_se`` (finite population correction included). If a nonempty stratum
    has fewer than two usable audited units its noise cannot be estimated:
    ``insufficient`` is true and ``mc_sd`` and ``shift_se`` are NaN.
    """

    shift: float
    shift_se: float
    mc_sd: float
    n_failed: int
    n_sampled: int
    insufficient: bool = False


def _check_sample(sample: AuditSample) -> None:
    k = len(sample.sizes)
    strata = np.asarray(sample.strata, dtype=int)
    counts = np.bincount(strata, minlength=k) if strata.size else np.zeros(k, int)
    if (
        len(sample.taken) != k
        or strata.size != len(sample.units)
        or (strata.size and (strata.min() < 0 or strata.max() >= k))
        or tuple(int(c) for c in counts) != tuple(sample.taken)
        or any(t > n for t, n in zip(sample.taken, sample.sizes, strict=True))
        or len(set(sample.units)) != len(sample.units)
    ):
        msg = "inconsistent AuditSample (strata, taken, sizes or duplicate units)"
        raise ValueError(msg)


def stability_estimate(
    original: ArrayLike, audit: ArrayLike, sample: AuditSample
) -> StabilityEstimate:
    """Estimate the Monte Carlo noise and the shift of a contrast total.

    ``original`` and ``audit`` are per-unit contrast values (for example paired
    elpd differences) for the sampled units, aligned with ``sample.units``. With
    ``e_i = audit_i - original_i``, strata ``s`` of ``N_s`` units, ``n_s``
    audited, ``ebar_s`` and ``v_s`` the stratum mean and sample variance:

    - ``shift = sum_s N_s ebar_s``, the estimated difference of the totals (audit
      minus original), with ``shift_se = sqrt(sum_s N_s^2 (1 - n_s/N_s) v_s /
      n_s)`` the sampling error of that extrapolation from a sample (zero for a
      census stratum);
    - ``mc_sd = sqrt(sum_s N_s v_s)``, the noise of a total over all ``N_s``
      units if the per-unit errors are independent with variance ``v_s``.
      ``var(e)`` holds the noise of **both** runs and any between-unit spread of
      the shift, so under independent errors and keys this overstates the noise
      of the original run.

    A sampled unit with a non-finite value in either run is counted in
    ``n_failed`` and left out of the moments; the shift assumes failures are
    missing at random within a stratum, and ``stability_label`` treats any
    failure as unassessable. A nonempty stratum with fewer than two usable
    audited units makes the estimate ``insufficient`` instead of raising.

    Raises:
        ValueError: Lengths do not match the sample, or the sample is inconsistent.

    Examples:
        >>> s = AuditSample((0, 1, 2, 3), (0, 0, 1, 1), (10, 10), (2, 2), 0, (0.5,))
        >>> est = stability_estimate([0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 1.0, 3.0], s)
        >>> est.shift, round(est.mc_sd, 3), round(est.shift_se, 3), est.n_failed
        (40.0, 6.325, 12.649, 0)
    """
    _check_sample(sample)
    a = np.asarray(original, dtype=float)
    b = np.asarray(audit, dtype=float)
    if a.shape != b.shape or a.shape != (len(sample.units),):
        msg = "original and audit must align with the sampled units"
        raise ValueError(msg)
    e = b - a
    usable = np.isfinite(e)
    strata = np.asarray(sample.strata)
    shift = 0.0
    var_shift = 0.0
    var_noise = 0.0
    insufficient = False
    for k, size in enumerate(sample.sizes):
        x = e[usable & (strata == k)]
        if size == 0:
            continue
        if x.size < 2:  # a variance needs two values
            insufficient = True
            continue
        v = float(x.var(ddof=1))
        shift += size * float(x.mean())
        var_shift += size**2 * (1.0 - x.size / size) * v / x.size
        var_noise += size * v
    n_failed, n_sampled = int(np.sum(~usable)), int(e.size)
    if insufficient:
        nan = float("nan")
        return StabilityEstimate(nan, nan, nan, n_failed, n_sampled, insufficient=True)
    return StabilityEstimate(
        shift,
        float(np.sqrt(max(var_shift, 0.0))),
        float(np.sqrt(var_noise)),
        n_failed,
        n_sampled,
    )


def stability_label(
    estimate: StabilityEstimate,
    se: float,
    *,
    max_mc_ratio: float,
    max_shift_ratio: float,
    z: float = 1.645,
) -> dict[str, object]:
    """Label a contrast ``stable`` or ``not_assessable`` (never ``tied``).

    Stable requires all of: no failed audited unit and no ``insufficient``
    stratum; ``mc_sd <= max_mc_ratio * se`` with ``se`` the standard error of the
    original total; and the shift small **including its sampling uncertainty**,
    ``|shift| + z * shift_se <= max_shift_ratio * se`` (one-sided, ``z`` =
    1.645). An audit too small to bound the shift is therefore not assessable. The
    ratios are the caller's
    reading-rule choices, fixed before the audit is read, and have no defaults;
    only ``z`` has a default. Stable means reproducible against an independent
    rerun, not accurate.

    Raises:
        ValueError: ``se``, a ratio or ``z`` is not a positive finite number.

    Examples:
        >>> est = StabilityEstimate(1.0, 0.5, 0.5, n_failed=0, n_sampled=90)
        >>> stability_label(est, 4.0, max_mc_ratio=0.25, max_shift_ratio=0.5)[
        ...     "label"
        ... ]
        'stable'
    """
    for name, value in (
        ("se", se),
        ("max_mc_ratio", max_mc_ratio),
        ("max_shift_ratio", max_shift_ratio),
        ("z", z),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            msg = f"{name} must be a positive finite number"
            raise ValueError(msg)
    mc_ratio = estimate.mc_sd / se
    shift_bound = (abs(estimate.shift) + z * estimate.shift_se) / se
    stable = (
        not estimate.insufficient
        and estimate.n_failed == 0
        and mc_ratio <= max_mc_ratio
        and shift_bound <= max_shift_ratio
    )
    label: Literal["stable", "not_assessable"] = (
        "stable" if stable else "not_assessable"
    )
    return {
        "label": label,
        "mc_ratio": float(mc_ratio),
        "shift_bound_ratio": float(shift_bound),
        "n_failed": estimate.n_failed,
        "n_sampled": estimate.n_sampled,
        "insufficient": estimate.insufficient,
    }
