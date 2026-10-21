r"""Rank-based simulation-based calibration (SBC): PITs, aggregation, gate.

Scope: the rank half of the calibration layer. Ordinary and randomized-rank PIT,
Benjamini-Hochberg multiplicity, power floors, failed-replicate accounting and
contraction, aggregated from one JSON file per replicate. The study extra adds
central-interval coverage summaries with Wilson intervals. Fixed coverage-rule
planning is re-exported from op_infer.coverage; optional reference TARP curves live
in op_infer.adapters.tarp. The module is estimator independent: any method that
can draw a posterior for a simulated truth (NUTS, SVI, NPE, Laplace) writes the
files. The bounded runner lives in op_infer.sbc.
The masked-unit runner and raw width comparisons live in op_infer.masked_sbc.
Prespecified scalar ``Gate``/``GateSet`` records live in op_infer.gates and are
re-exported here.
There is no plotting layer;
the ledger carries the data for rank histograms.

Estimator adapters use the same replicate schema, so one aggregator can read
their calibration records.

Replicate JSON schema (one ``*.json`` file per replicate)::

    {
      "<estimand>": {"pit": 0.37, "contraction": 0.41},
      "<estimand 2>": {"pit": 0.92},
      "_fit_diag": {"rhat_max": 1.01, "ess_min": 850}
    }

``contraction`` (posterior SD over prior SD) is optional. A key that starts with
``_`` is a diagnostic, not an estimand; ``_fit_diag`` values are collected per key.
A replicate that failed (non-convergence, timeout, solver error) is written as
``{"_failed": true}``. Failures are counted, kept in the denominator and reported;
more than 10% failed fails the gate. A replicate that lacks an estimand without
``_failed`` is an error, and so is a non-finite PIT: nothing is dropped silently.

Rules encoded here (see ``docs/calibration.md``):

* a pass below ``power_floor`` replicates is an "underpowered pass" and is not a
  calibration claim; a failure counts at any number of replicates;
* quantities with atoms (peak week, point masses at zero) need the randomized PIT
  (``atoms=True``);
* flagging uses the Benjamini-Hochberg q-value across the estimand set, never the
  raw p-value;
* before reading any rank, assert that the simulated data differ from the observed
  data (``assert_simulated_differs``);
* report contraction beside the ranks, because a posterior equal to the prior is
  uniform.

Examples:
    >>> import numpy as np
    >>> pit([1.0, 2.0, 2.0, 3.0], 2.0)
    0.5
    >>> rng = np.random.default_rng(0)
    >>> 0.25 <= pit([1.0, 2.0, 2.0, 3.0], 2.0, rng=rng, atoms=True) <= 0.75
    True
    >>> benjamini_hochberg([0.01, 0.04, 0.03, 0.005]).round(3).tolist()
    [0.02, 0.04, 0.04, 0.02]

Command line (exit status 1 on a gate failure)::

    python -m op_infer.calibrate aggregate DIR --out LEDGER.json \
        [--max-flagged 1] [--must-pass a,b]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chisquare

from op_infer.coverage import (
    CoverageRule,
    coverage_operating_characteristics,
    plan_coverage_replicates,
)
from op_infer.gates import Gate, GateAmendment, GateOutcome, GateRecord, GateSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MAX_FAILED_FRACTION",
    "POWER_FLOOR",
    "CoverageRule",
    "Gate",
    "GateAmendment",
    "GateOutcome",
    "GateRecord",
    "GateSet",
    "aggregate",
    "assert_simulated_differs",
    "benjamini_hochberg",
    "coverage_operating_characteristics",
    "coverage_summary",
    "format_table",
    "gate",
    "load_replicates",
    "main",
    "pit",
    "plan_coverage_replicates",
    "posterior_contraction",
    "write_failed_replicate",
    "write_ledger",
    "write_replicate",
]

MAX_FAILED_FRACTION = 0.10
POWER_FLOOR = 200


def _count_setting(value: int, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)
    return int(value)


def _probability_setting(value: float, name: str, *, closed: bool) -> float:
    valid_type = not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
    valid_range = valid_type and (0 <= value <= 1 if closed else 0 < value < 1)
    if not valid_range:
        msg = f"{name} must be finite and in {'[0, 1]' if closed else '(0, 1)'}"
        raise ValueError(msg)
    result = float(value)
    if not math.isfinite(result) or (not closed and not 0 < result < 1):
        msg = f"{name} must remain a finite supported probability"
        raise ValueError(msg)
    return result


def _rank_settings(bins: int, power_floor: int, alpha: float) -> tuple[int, int, float]:
    return (
        _count_setting(bins, "bins", 2),
        _count_setting(power_floor, "power_floor", 1),
        _probability_setting(alpha, "alpha", closed=False),
    )


def pit(
    draws: ArrayLike,
    truth: float,
    *,
    rng: np.random.Generator | None = None,
    atoms: bool = False,
) -> float:
    """Return the probability integral transform of ``truth`` among ``draws``.

    For a continuous quantity the result is ``(#below + 0.5 #equal) / n``. With
    ``atoms=True`` the randomized form ``p_below + U p_equal`` with
    ``U ~ Uniform(0, 1)`` is used. It is uniform under calibration for discrete
    quantities and quantities with point masses; ordinary and mid-ranks are not.

    Args:
        draws: Posterior draws of the quantity.
        truth: The simulated true value of the quantity.
        rng: Random generator for the randomization; required when ``atoms`` is
            true. Seed it so the gate is reproducible.
        atoms: Use the randomized PIT for quantities with atoms.

    Returns:
        The PIT in [0, 1].

    Raises:
        ValueError: If there are no draws, a draw or the truth is not finite, or
            ``atoms`` is set without ``rng``.

    Examples:
        >>> pit([1.0, 2.0, 3.0], 2.5)
        0.6666666666666666
        >>> key = np.random.default_rng(3)
        >>> 0.25 <= pit([0, 1, 1, 2], 1, atoms=True, rng=key) <= 0.75
        True
    """
    x = np.asarray(draws, dtype=float).ravel()
    if x.size == 0:
        msg = "no posterior draws"
        raise ValueError(msg)
    if not (np.all(np.isfinite(x)) and np.isfinite(truth)):
        msg = "posterior draws and truth must be finite"
        raise ValueError(msg)
    p_below = float(np.mean(x < truth))
    p_equal = float(np.mean(x == truth))
    if atoms:
        if rng is None:
            msg = "atoms=True needs a numpy Generator in rng"
            raise ValueError(msg)
        return p_below + float(rng.uniform()) * p_equal
    return p_below + 0.5 * p_equal


def posterior_contraction(draws: ArrayLike, prior_sd: float) -> float:
    """Return the posterior SD over the prior SD of a quantity.

    Args:
        draws: Posterior draws of the quantity.
        prior_sd: Standard deviation of the prior (positive and finite).

    Returns:
        ``std(draws) / prior_sd``; values near 1 mean the data did not narrow the
        prior.

    Raises:
        ValueError: If there are no finite draws or ``prior_sd`` is not positive
            and finite.
    """
    x = np.asarray(draws, dtype=float).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        msg = "posterior draws must be non-empty and finite"
        raise ValueError(msg)
    if not (np.isfinite(prior_sd) and prior_sd > 0.0):
        msg = "prior_sd must be positive and finite"
        raise ValueError(msg)
    return float(np.std(x, ddof=1) / prior_sd) if x.size > 1 else 0.0


def assert_simulated_differs(
    simulated: ArrayLike, observed: ArrayLike, mask: ArrayLike
) -> int:
    """Check that simulated data are not the real data; count the equal cells.

    A probabilistic program with an observed site can return the observed array
    from its predictive. Every replicate then refits the real data against an
    unrelated prior draw and the ranks look like a result.

    Args:
        simulated: Simulated data of the replicate.
        observed: The real data.
        mask: Boolean array, true where the real cell is observed.

    Returns:
        The number of observed cells where the simulated value equals the real
        value (record it).

    Raises:
        ValueError: If the shapes differ, no cell is observed, or all observed
            cells are equal (the replicate refits the real data).
    """
    sim = np.asarray(simulated)
    obs = np.asarray(observed)
    keep = np.asarray(mask, dtype=bool)
    if not sim.shape == obs.shape == keep.shape:
        msg = (
            f"shape mismatch: simulated {sim.shape}, observed {obs.shape}, "
            f"mask {keep.shape}"
        )
        raise ValueError(msg)
    n_obs = int(keep.sum())
    if n_obs == 0:
        msg = "mask has no observed cell"
        raise ValueError(msg)
    n_equal = int(np.sum((sim == obs) & keep))
    if n_equal == n_obs:
        msg = (
            f"all {n_obs} observed cells equal the real data: "
            "the replicate refits the real data"
        )
        raise ValueError(msg)
    return n_equal


def benjamini_hochberg(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Return Benjamini-Hochberg step-up q-values in the input order.

    For ascending p-values ``p_(1..m)`` the q-value is
    ``min(1, min_{j >= i} p_(j) m / j)``.

    Args:
        pvalues: Raw p-values in [0, 1].

    Returns:
        The q-values, same length and order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q


def _as_pits(name: str, values: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        msg = f"estimand {name!r} has no PITs"
        raise ValueError(msg)
    if not np.all(np.isfinite(x)) or x.min() < 0.0 or x.max() > 1.0:
        msg = f"estimand {name!r} has PITs that are not finite values in [0, 1]"
        raise ValueError(msg)
    return x


def _estimand_row(
    x: NDArray[np.float64],
    p: float,
    q: float,
    c: NDArray[np.float64] | None,
    alpha: float,
) -> dict[str, Any]:
    return {
        "n": int(x.size),
        "p_chisq": p,
        "q_bh": q,
        "mean_pit": float(x.mean()),
        "median_pit": float(np.median(x)),
        "coverage50": float(np.mean((x >= 0.25) & (x <= 0.75))),
        "coverage90": float(np.mean((x >= 0.05) & (x <= 0.95))),
        "mean_contraction": None if c is None or c.size == 0 else float(c.mean()),
        "flagged": bool(q <= alpha),
    }


def _failed_replicate_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        msg = "n_failed must be a nonnegative integer"
        raise ValueError(msg)
    return int(value)


def coverage_summary(
    pits: Mapping[str, ArrayLike],
    *,
    levels: Sequence[float] = (0.5, 0.9),
    confidence: float = 0.95,
    n_failed: int = 0,
) -> dict[str, Any]:
    """Summarize central posterior coverage with marginal Wilson intervals.

    The binomial calculation uses trade-study.
    Requires the study extra. Each PIT must come from one independent SBC
    replicate of the same estimator and simulation design, at a fixed size.
    For atoms use randomized PITs. Finite posterior draws can discretize PITs.

    Coverage is conditional on successful fits. Failures remain explicit in
    the total attempt count and failure fraction; they are not covered draws.
    An all-failed bank has unavailable coverage, not a zero-width interval.
    The same successful replicate bank must supply every estimand. Intervals
    are marginal, without multiple-testing or optional-stopping guarantees.
    An interval containing its nominal level does not certify calibration.

    Args:
        pits: Estimand names to finite PIT vectors in [0, 1].
        levels: Distinct nominal central coverage levels in (0, 1).
        confidence: Wilson interval confidence in (0, 1).
        n_failed: Nonnegative integer count of failed attempts in the bank.

    Returns:
        JSON-compatible counts, failure fraction and per-estimand coverage
        rows. Each row retains its nominal level and complete interval record.

    Raises:
        ValueError: Counts, levels, PITs or the shared bank are invalid.
        ImportError: The study extra is not installed.

    Examples:
        Requires the study extra; the required study job executes this example.

        >>> ranks = (np.arange(1000) + 0.5) / 1000
        >>> report = coverage_summary(
        ...     {"theta": ranks}, n_failed=5
        ... )  # doctest: +SKIP
        >>> report["n_successful"], report["n_total"]  # doctest: +SKIP
        (1000, 1005)
        >>> report["estimands"]["theta"][0]["estimate"]  # doctest: +SKIP
        0.5
    """
    from trade_study import proportion_interval  # noqa: PLC0415 - optional dependency

    n_failed = _failed_replicate_count(n_failed)
    nominal = tuple(levels)
    if (
        not nominal
        or any(
            isinstance(x, bool) or not np.isfinite(x) or not 0 < x < 1 for x in nominal
        )
        or len(set(nominal)) != len(nominal)
    ):
        msg = "levels must be distinct finite values in (0, 1)"
        raise ValueError(msg)
    # Delegate confidence validation even when the entire bank failed.
    proportion_interval(0, 1, confidence=confidence)
    arrays = {name: _as_pits(name, values) for name, values in pits.items()}
    counts = {x.size for x in arrays.values()}
    if len(counts) > 1 or (not arrays and n_failed == 0):
        msg = "coverage requires one shared nonempty replicate bank"
        raise ValueError(msg)
    successful = int(next(iter(counts), 0))
    total = successful + n_failed
    table = {}
    for name, x in arrays.items():
        rows = []
        for level in nominal:
            tail = (1 - level) / 2
            k = int(np.count_nonzero((x >= tail) & (x <= 1 - tail)))
            interval = proportion_interval(k, successful, confidence=confidence)
            rows.append({"nominal": float(level), **asdict(interval)})
        table[name] = rows
    return {
        "estimands": table,
        "available": bool(successful),
        "n_successful": successful,
        "n_failed": n_failed,
        "n_total": total,
        "failed_fraction": n_failed / total,
        "conditioning": "successful fits",
        "multiplicity": "marginal",
        "sampling": "fixed independent replicate bank",
        "settings": {
            "levels": list(nominal),
            "confidence": confidence,
            "method": "wilson",
        },
    }


def aggregate(  # noqa: PLR0913 -- one keyword per documented setting
    pits: Mapping[str, ArrayLike],
    contraction: Mapping[str, ArrayLike] | None = None,
    *,
    bins: int = 20,
    power_floor: int = POWER_FLOOR,
    alpha: float = 0.05,
    n_failed: int = 0,
) -> dict[str, Any]:
    """Summarize the PITs of every estimand and flag the non-uniform ones.

    Args:
        pits: Estimand name to the per-replicate PITs in [0, 1].
        contraction: Optional estimand name to per-replicate posterior SD over
            prior SD.
        bins: Number of equal-width bins of the chi-square uniformity test.
        power_floor: Replicate count below which the result is underpowered.
        alpha: Flagging level on the Benjamini-Hochberg q-value.
        n_failed: Nonnegative integer count of failed replicates; kept as failures (see
            ``load_replicates``).

    Returns:
        The per-estimand table under ``estimands`` (n, chi-square p, BH q, mean
        and median PIT, coverage50, coverage90, mean contraction or None,
        flagged) and the top-level n_replicates (the minimum over estimands),
        underpowered, n_flagged, flagged, underpowered_banner (None when
        powered), n_failed, n_total, failed_fraction and the settings.

    Raises:
        ValueError: Replicates/PITs are invalid, failed counts are not nonnegative
            integers, bins are not integers >= 2, the power floor is not a
            positive integer, alpha is not finite/in (0, 1), or a test is nonfinite.

    Examples:
        >>> ranks = np.tile((np.arange(20) + 0.5) / 20, 50)
        >>> report = aggregate({"theta": ranks}, n_failed=5)
        >>> report["n_replicates"], report["n_total"], report["underpowered"]
        (1000, 1005, False)
        >>> report["n_flagged"]
        0
    """
    bins, power_floor, alpha = _rank_settings(bins, power_floor, alpha)
    n_failed = _failed_replicate_count(n_failed)
    if not pits and n_failed == 0:
        msg = "no replicates to aggregate"
        raise ValueError(msg)
    arrays = {name: _as_pits(name, values) for name, values in pits.items()}
    pvals = [
        float(chisquare(np.histogram(x, bins=bins, range=(0.0, 1.0))[0]).pvalue)
        for x in arrays.values()
    ]
    if not np.all(np.isfinite(pvals)):
        msg = "Rank uniformity returned nonfinite p-values; no gate can be read"
        raise ValueError(msg)
    qvals = benjamini_hochberg(pvals)

    table: dict[str, dict[str, Any]] = {}
    for (name, x), p, q in zip(arrays.items(), pvals, qvals, strict=True):
        c = None
        if contraction is not None and name in contraction:
            c = np.asarray(contraction[name], dtype=float)
        table[name] = _estimand_row(x, p, float(q), c, alpha)
    n_rep = min((row["n"] for row in table.values()), default=0)
    underpowered = n_rep < power_floor
    flagged = [name for name, row in table.items() if row["flagged"]]
    n_total = n_rep + n_failed
    banner = (
        f"UNDERPOWERED: {n_rep} replicates < power floor {power_floor}. "
        "A pass means 'not detected', not 'calibrated'; failures count at any n."
    )
    return {
        "estimands": table,
        "n_replicates": n_rep,
        "underpowered": underpowered,
        "n_flagged": len(flagged),
        "flagged": flagged,
        "underpowered_banner": banner if underpowered else None,
        "n_failed": int(n_failed),
        "n_total": n_total,
        "failed_fraction": n_failed / n_total if n_total else 0.0,
        "settings": {"bins": bins, "power_floor": power_floor, "alpha": alpha},
    }


def gate(
    result: Mapping[str, Any],
    *,
    max_flagged: int,
    must_pass: Sequence[str] = (),
    max_failed_fraction: float = MAX_FAILED_FRACTION,
) -> dict[str, Any]:
    """Apply the pass rule to an aggregated result.

    Pass iff at most ``max_flagged`` estimands are flagged, none of ``must_pass``
    is flagged, and no more than ``max_failed_fraction`` of the replicates
    failed. A pass with an underpowered result is labelled "underpowered pass",
    which is not a claim of calibration; a failure counts at any number of
    replicates.

    Args:
        result: Output of ``aggregate``.
        max_flagged: Largest number of flagged estimands that still passes.
        must_pass: Estimands that must not be flagged.
        max_failed_fraction: Largest fraction of failed replicates that passes.

    Returns:
        ``passed``, ``underpowered``, ``label`` ("pass", "fail" or "underpowered
        pass"), ``reasons`` (why it failed), ``n_failed`` and
        ``failed_fraction``.

    Raises:
        ValueError: A must-pass name is unknown, settings are missing/invalid,
            max_flagged is not a nonnegative integer or max_failed_fraction is
            not a finite probability in [0, 1].

    Examples:
        >>> ranks = np.tile((np.arange(20) + 0.5) / 20, 5)
        >>> report = aggregate({"theta": ranks})
        >>> verdict = gate(report, max_flagged=0)
        >>> verdict["label"]
        'underpowered pass'
        >>> verdict["underpowered"]
        True
    """
    max_flagged = _count_setting(max_flagged, "max_flagged", 0)
    max_failed_fraction = _probability_setting(
        max_failed_fraction, "max_failed_fraction", closed=True
    )
    settings = result.get("settings")
    if (
        not isinstance(settings, Mapping)
        or not {"bins", "power_floor", "alpha"} <= settings.keys()
    ):
        msg = "Gate requires the settings from its aggregate result"
        raise ValueError(msg)
    _rank_settings(settings["bins"], settings["power_floor"], settings["alpha"])
    estimands: Mapping[str, Any] = result["estimands"]
    unknown = [name for name in must_pass if name not in estimands]
    if unknown:
        msg = f"must_pass names not in the result: {unknown}"
        raise ValueError(msg)
    reasons: list[str] = []
    if result["n_flagged"] > max_flagged:
        reasons.append(
            f"{result['n_flagged']} estimands flagged (> {max_flagged}): "
            + ", ".join(result["flagged"])
        )
    reasons += [
        f"must-pass estimand {name!r} is flagged"
        for name in must_pass
        if estimands[name]["flagged"]
    ]
    n_failed = int(result.get("n_failed", 0))
    failed_fraction = float(result.get("failed_fraction", 0.0))
    if failed_fraction > max_failed_fraction:
        reasons.append(
            f"{n_failed} of {result.get('n_total', n_failed)} replicates failed "
            f"({failed_fraction:.1%} > {max_failed_fraction:.0%})"
        )
    passed = not reasons
    underpowered = bool(result["underpowered"])
    label = "fail" if not passed else ("underpowered pass" if underpowered else "pass")
    return {
        "passed": passed,
        "underpowered": underpowered,
        "label": label,
        "reasons": reasons,
        "n_failed": n_failed,
        "failed_fraction": failed_fraction,
    }


def _write_json_atomically(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.with_name(out.name + ".tmp")
    scratch.write_text(json.dumps(payload, indent=2) + "\n")
    scratch.replace(out)


def write_replicate(
    path: str | Path,
    pits: Mapping[str, float],
    *,
    contraction: Mapping[str, float] | None = None,
    fit_diag: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write one replicate in the shared schema, atomically.

    The file appears complete or not at all, so a runner can resume by skipping
    existing files and array tasks can write into one directory.

    Args:
        path: Output ``*.json`` file; parent directories are created.
        pits: Estimand name to the replicate's PIT in [0, 1].
        contraction: Optional estimand name to posterior SD over prior SD.
        fit_diag: Optional per-replicate fit diagnostics (R-hat, ESS,
            divergences), stored as ``_fit_diag``.
        metadata: Optional runner/provenance metadata, stored as ``_meta``.

    Raises:
        ValueError: If a PIT is not a finite value in [0, 1] or an estimand name
            starts with an underscore.
    """
    payload: dict[str, Any] = {}
    for name, value in pits.items():
        if name.startswith("_"):
            msg = f"estimand name {name!r} must not start with an underscore"
            raise ValueError(msg)
        _as_pits(name, [value])
        entry: dict[str, float] = {"pit": float(value)}
        if contraction is not None and name in contraction:
            entry["contraction"] = float(contraction[name])
        payload[name] = entry
    if fit_diag is not None:
        payload["_fit_diag"] = dict(fit_diag)
    if metadata is not None:
        payload["_meta"] = dict(metadata)
    _write_json_atomically(path, payload)


def write_failed_replicate(
    path: str | Path,
    reason: str | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a failed replicate (``{"_failed": true}``) so it stays counted.

    Args:
        path: Output ``*.json`` file; parent directories are created.
        reason: Optional short cause (timeout, non-convergence), stored as
            ``_reason``.
        metadata: Optional runner/provenance metadata, stored as ``_meta``.
    """
    payload: dict[str, Any] = {"_failed": True}
    if reason is not None:
        payload["_reason"] = reason
    if metadata is not None:
        payload["_meta"] = dict(metadata)
    _write_json_atomically(path, payload)


def _read_replicate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        msg = f"{path.name}: expected a JSON object"
        raise ValueError(msg)  # noqa: TRY004 -- a schema violation, not a type bug
    return data


def _pit_entry(file: Path, data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    entry = data.get(name)
    if not isinstance(entry, dict) or "pit" not in entry:
        msg = (
            f"{file.name}: missing estimand {name!r} "
            "(mark a failed replicate with _failed: true)"
        )
        raise ValueError(msg)
    if not np.isfinite(float(entry["pit"])):
        msg = (
            f"{file.name}: non-finite PIT for {name!r} "
            "(mark a failed replicate with _failed)"
        )
        raise ValueError(msg)
    return entry


def load_replicates(
    directory: str | Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
    """Read the replicate JSON files of a directory.

    Args:
        directory: Directory with one ``*.json`` file per replicate (schema in
            the module docstring).

    Returns:
        ``(pits, contraction, diag)``: PITs and contractions per estimand over
        the successful replicates, and ``diag`` with ``n_failed``, ``n_total``,
        ``failed_files`` and ``fit_diag`` (the ``_fit_diag`` values per key).

    Raises:
        ValueError: If there is no file, a file is malformed, a PIT is not
            finite, or a replicate that is not marked ``_failed`` lacks an
            estimand that another has.
    """
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        msg = f"no *.json replicate files in {directory}"
        raise ValueError(msg)
    records = [(f, _read_replicate(f)) for f in files]
    failed = [f.name for f, data in records if data.get("_failed") is True]
    ok = [(f, data) for f, data in records if data.get("_failed") is not True]

    names = list(
        dict.fromkeys(k for _, data in ok for k in data if not k.startswith("_"))
    )
    pits: dict[str, list[float]] = {name: [] for name in names}
    contraction: dict[str, list[float]] = {}
    fit_diag: dict[str, list[Any]] = {}
    for f, data in ok:
        for name in names:
            entry = _pit_entry(f, data, name)
            pits[name].append(float(entry["pit"]))
            if entry.get("contraction") is not None:
                contraction.setdefault(name, []).append(float(entry["contraction"]))
        for key, value in (data.get("_fit_diag") or {}).items():
            fit_diag.setdefault(key, []).append(value)
    diag = {
        "n_failed": len(failed),
        "n_total": len(files),
        "failed_files": failed,
        "fit_diag": fit_diag,
    }
    return pits, contraction, diag


def _git_sha() -> str | None:
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git on PATH
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd(),
        )
        return out.stdout.strip() or None
    return None


def write_ledger(
    path: str | Path, result: Mapping[str, Any], *, meta: Mapping[str, Any]
) -> None:
    """Write the SBC result and its provenance as a JSON ledger.

    Args:
        path: Output file; parent directories are created.
        result: The aggregated result (``aggregate``, optionally with the gate
            under ``gate``).
        meta: Provenance such as the script name and settings; ``git_sha`` (of
            the current working directory) and ``timestamp_utc`` are added
            unless given.
    """
    full_meta = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **meta,
    }
    _write_json_atomically(path, {"meta": full_meta, "result": result})


def format_table(result: Mapping[str, Any]) -> str:
    """Return the per-estimand table as text.

    Args:
        result: Output of ``aggregate``.

    Returns:
        The table, one row per estimand.
    """
    head = (
        f"{'estimand':<28}{'n':>6}{'chi2 p':>10}{'q_BH':>8}{'meanPIT':>9}"
        f"{'cov50':>7}{'cov90':>7}{'contr.':>8}  flagged"
    )
    lines = [head]
    for name, r in result["estimands"].items():
        contr = "-" if r["mean_contraction"] is None else f"{r['mean_contraction']:.2f}"
        flag = "FLAGGED" if r["flagged"] else "ok"
        lines.append(
            f"{name:<28}{r['n']:>6}{r['p_chisq']:>10.3g}{r['q_bh']:>8.3f}"
            f"{r['mean_pit']:>9.3f}{r['coverage50']:>7.2f}{r['coverage90']:>7.2f}"
            f"{contr:>8}  {flag}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``aggregate`` command line.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        The exit status: 0 on a pass or an underpowered pass, 1 on a gate
        failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m op_infer.calibrate",
        description="Rank-based SBC: aggregate replicate JSON files, apply the gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    agg = sub.add_parser("aggregate", help="aggregate a directory of replicates")
    agg.add_argument("directory", type=Path)
    agg.add_argument("--out", type=Path, required=True, help="ledger JSON to write")
    agg.add_argument("--max-flagged", type=int, default=1, help="gate: flagged allowed")
    agg.add_argument("--must-pass", default="", help="gate: comma-separated estimands")
    agg.add_argument("--bins", type=int, default=20)
    agg.add_argument("--power-floor", type=int, default=POWER_FLOOR)
    agg.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    pits, contraction, diag = load_replicates(args.directory)
    result = aggregate(
        pits,
        contraction,
        bins=args.bins,
        power_floor=args.power_floor,
        alpha=args.alpha,
        n_failed=diag["n_failed"],
    )
    must_pass = [name for name in args.must_pass.split(",") if name]
    verdict = gate(result, max_flagged=args.max_flagged, must_pass=must_pass)

    lines = [
        (
            f"# SBC aggregate: {result['n_replicates']} replicates "
            f"({diag['n_failed']} failed) from {args.directory}"
        )
    ]
    if result["underpowered_banner"]:
        lines.append(f"# *** {result['underpowered_banner']} ***")
    lines += [
        format_table(result),
        (
            f"\ngate: {verdict['label']} (max flagged {args.max_flagged}, "
            f"must pass {must_pass or 'none'})"
        ),
        *[f"  - {reason}" for reason in verdict["reasons"]],
    ]
    sys.stdout.write("\n".join(lines) + "\n")

    full = {
        **result,
        "gate": verdict,
        "fit_diag": diag["fit_diag"],
        "failed_files": diag["failed_files"],
    }
    meta = {
        "script": "op_infer.calibrate aggregate",
        "directory": str(args.directory),
        "settings": {
            "bins": args.bins,
            "power_floor": args.power_floor,
            "alpha": args.alpha,
            "max_flagged": args.max_flagged,
            "must_pass": must_pass,
        },
    }
    write_ledger(args.out, full, meta=meta)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
