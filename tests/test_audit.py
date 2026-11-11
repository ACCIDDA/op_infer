"""Stability audit: a diagnostics-only sample and known Monte Carlo noise."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from op_infer.audit import (
    AuditSample,
    StabilityEstimate,
    audit_sample,
    stability_estimate,
    stability_label,
    stratify,
)


def test_sampler_has_no_score_argument_and_is_reproducible() -> None:
    """The sample depends on the diagnostic and seed only."""
    names = set(inspect.signature(audit_sample).parameters)
    assert names == {"diagnostic", "edges", "per_stratum", "seed"}
    d = np.random.default_rng(0).uniform(0, 1.5, 500)
    a = audit_sample(d, edges=(0.5, 0.7), per_stratum=20, seed=4)
    b = audit_sample(d, edges=(0.5, 0.7), per_stratum=20, seed=4)
    assert a == b
    assert audit_sample(d, edges=(0.5, 0.7), per_stratum=20, seed=5) != a
    assert sum(a.sizes) == 500
    assert a.taken == tuple(min(20, s) for s in a.sizes)
    assert len(set(a.units)) == len(a.units)
    assert all(
        stratify(d, (0.5, 0.7))[u] == k for u, k in zip(a.units, a.strata, strict=True)
    )


def test_nonfinite_goes_to_top_stratum_and_small_strata_are_complete() -> None:
    """Nothing leaves the frame; a stratum smaller than the target is taken whole."""
    d = np.array([0.1, 0.2, np.nan, np.inf, 0.9, 1.0])
    s = audit_sample(d, edges=(0.5,), per_stratum=3, seed=1)
    assert s.sizes == (2, 4)
    assert s.taken == (2, 3)
    assert stratify(d, (0.5,)).tolist() == [0, 0, 1, 1, 1, 1]
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="per_stratum"):
            audit_sample(d, edges=(0.5,), per_stratum=bad, seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly increasing"):
        stratify(d, (0.7, 0.5))


def test_known_noise_and_bias_are_recovered_over_repeated_audits() -> None:
    """Shift and its sampling error match the simulated truth; noise is conservative."""
    rng = np.random.default_rng(11)
    n_units = 2000
    diag = rng.uniform(0.0, 1.5, n_units)
    truth = rng.normal(0.0, 1.0, n_units)
    sigma_o, sigma_a, bias = 0.5, 0.25, -0.02
    original = truth + bias + sigma_o * rng.normal(size=n_units)
    audit_all = truth + sigma_a * rng.normal(size=n_units)
    sample0 = audit_sample(diag, edges=(0.5, 0.7), per_stratum=120, seed=0)
    shifts, ses, mcs = [], [], []
    for seed in range(300):
        s = audit_sample(diag, edges=(0.5, 0.7), per_stratum=120, seed=seed)
        u = np.array(s.units)
        est = stability_estimate(original[u], audit_all[u], s)
        shifts.append(est.shift)
        ses.append(est.shift_se)
        mcs.append(est.mc_sd)
    full = float(np.sum(audit_all - original))
    assert sample0.sizes == tuple(int(x) for x in sample0.sizes)
    assert abs(np.mean(shifts) - full) < 4 * np.std(shifts) / np.sqrt(300)
    assert 0.8 < np.mean(ses) / np.std(shifts) < 1.25
    assert 0.9 < np.mean(mcs) / np.sqrt(n_units * (sigma_o**2 + sigma_a**2)) < 1.1


def test_label_requires_small_noise_small_shift_and_no_failures() -> None:
    """Stable only if every condition holds; ratios have no defaults."""
    good = StabilityEstimate(
        shift=1.0, shift_se=0.5, mc_sd=0.5, n_failed=0, n_sampled=90
    )
    kw = {"max_mc_ratio": 0.25, "max_shift_ratio": 0.5}
    assert stability_label(good, 4.0, **kw)["label"] == "stable"
    noisy = StabilityEstimate(1.0, 0.5, 2.0, 0, 90)
    assert stability_label(noisy, 4.0, **kw)["label"] == "not_assessable"
    unbounded = StabilityEstimate(1.0, 5.0, 0.5, 0, 90)
    assert stability_label(unbounded, 4.0, **kw)["label"] == "not_assessable"
    failed = StabilityEstimate(1.0, 0.5, 0.5, 1, 90)
    assert stability_label(failed, 4.0, **kw)["label"] == "not_assessable"
    with pytest.raises(TypeError):
        stability_label(good, 4.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="se"):
        stability_label(good, 0.0, **kw)
    with pytest.raises(ValueError, match="max_mc_ratio"):
        stability_label(good, 4.0, max_mc_ratio=True, max_shift_ratio=0.5)


def test_failed_units_are_counted_and_short_strata_refused() -> None:
    """A non-finite value is counted, not silently averaged away."""
    s = AuditSample((0, 1, 2, 3, 4, 5), (0, 0, 0, 1, 1, 1), (10, 10), (3, 3), 0, (0.5,))
    est = stability_estimate([0.0] * 6, [1.0, 3.0, 2.0, np.nan, 2.0, 4.0], s)
    assert est.n_failed == 1
    with pytest.raises(ValueError, match="fewer than two"):
        stability_estimate([0.0] * 6, [1.0, 3.0, 2.0, np.nan, np.nan, 4.0], s)
    with pytest.raises(ValueError, match="align"):
        stability_estimate([0.0] * 5, [1.0] * 5, s)
