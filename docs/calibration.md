# Rank-based calibration (SBC)

`op_infer.calibrate` turns a directory of simulation-based calibration (SBC)
replicates into a per-estimand table, a gate label and a ledger. It is estimator
independent: NUTS, SVI, amortized NPE or a Laplace fit all write the same
one-file-per-replicate schema, so one aggregator scores them all. It also reports central-interval coverage
with Wilson intervals when the study extra is installed. The bounded runner is described in
the [SBC execution guide](sbc-runner.md). [Coverage planning and TARP](coverage-planning.md)
provides fixed-coverage family sizing and optional reference curves.
[Masked-unit SBC](masked-sbc.md) adds complete planned banks, stratum readings and
posterior-width comparisons. [Saved gate records](gates.md)
provide prespecified scalar criteria separately from the legacy rank gate. There is no plotting layer here: the
ledger carries the data for rank histograms.

The following requirements support reproducible calibration checks.

Rank aggregation requires an integer `bins >= 2`, a positive integer
`power_floor`, and finite `0 < alpha < 1`. Gate controls require a nonnegative
integer `max_flagged` and finite `max_failed_fraction` in `[0, 1]`. Boolean values
are rejected; supported NumPy scalar settings are normalized to Python numbers.
Invalid settings raise an error before statistics or a gate label, including
when an older aggregate is passed back to `gate`. A nonfinite reference test
cannot become an unflagged result. These checks do not qualify a gate's power
or select a scientific threshold.

## Rules

1. **Assert that the simulated data differ from the observed data, before any rank
   is read.** A probabilistic program with an observed site can hand back the
   observed array from its predictive, so every replicate refits the real data
   against an unrelated prior draw, invalidating the calibration experiment.
   `assert_simulated_differs(simulated, observed, mask)` raises when every
   observed cell is equal and returns the number of equal cells, which the runner
   should record.
2. **Power floor.** Below `power_floor` replicates (default 200) the aggregate
   carries `underpowered: true` and a banner, and a pass is labelled
   `"underpowered pass"`: *not detected*, not *calibrated*. Failures are
   evidence of a discrepancy; an underpowered pass is not calibration evidence.
3. **Failed replicates are counted and stay in the denominator.** Write a failed
   or timed-out replicate as `{"_failed": true}` (`write_failed_replicate`). More
   than `max_failed_fraction` (default 10%) failed fails the gate. A successful replicate with a missing estimand or a non-finite
   PIT is an error, never a silent drop. Do not filter the simulated truth on
   plausibility unless the fitted prior has the same truncation.
4. **Randomized-rank PIT for quantities with atoms.** For discrete quantities
   (a peak week) and point masses (zero), use
   `pit(draws, truth, rng=rng, atoms=True)`, which returns
   `p_below + U * p_equal`. Ordinary and mid-rank PITs reject a posterior that is
   calibrated there; see
   `test_discrete_quantity_needs_randomized_pit`. Seed the generator, so the
   gate replays.
5. **Multiplicity.** Each estimand gets a chi-square uniformity test on 20
   equal-width bins; an estimand is flagged on its Benjamini-Hochberg q-value
   across the estimand set. The gate
   passes with at most `max_flagged` flagged estimands and none of `must_pass`.
6. **Contraction beside ranks.** A posterior equal to the prior is uniform, so
   ranks cannot see an uninformative fit. Store `contraction` (posterior SD over
   prior SD, `posterior_contraction(draws, prior_sd)`) and per-replicate fit
   diagnostics (`_fit_diag`: R-hat, ESS, divergences) with each replicate.
7. **Test the gate on a posterior that is calibrated by construction**, and on a
   biased or overconfident one, and size the replicate count so
   the false-fail rate under calibration is small.
   `tests/test_calibrate.py` does this with a conjugate normal and a discrete
   toy.

## Replicate schema

Write one `*.json` file per replicate using the following schema:

```json
{
  "theta": {"pit": 0.37, "contraction": 0.41},
  "peak_week": {"pit": 0.92},
  "_fit_diag": {"rhat_max": 1.01, "ess_min": 850}
}
```

`write_replicate` writes through a scratch file and renames, so a replicate
appears complete or not at all: a runner can resume by skipping existing files,
and array tasks can share a directory.

## Use

```python
import numpy as np

from op_infer.calibrate import (
    aggregate,
    gate,
    load_replicates,
    pit,
    posterior_contraction,
    write_ledger,
    write_replicate,
)

rng = np.random.default_rng(7)  # seeded, separate from simulation
draws, truth = posterior_draws, simulated_truth
write_replicate(
    "sbc/rep_0001.json",
    {
        "theta": pit(draws, truth),
        "peak_week": pit(peak_draws, peak, rng=rng, atoms=True),
    },
    contraction={"theta": posterior_contraction(draws, prior_sd)},
    fit_diag={"rhat_max": 1.003},
)

pits, contraction, diag = load_replicates("sbc")
result = aggregate(pits, contraction, n_failed=diag["n_failed"])
verdict = gate(result, max_flagged=1, must_pass=["theta"])
write_ledger("sbc/ledger.json", {**result, "gate": verdict}, meta={"script": "..."})
```

or `python -m op_infer.calibrate aggregate sbc --out sbc/ledger.json`, which
exits with status 1 on a gate failure and 0 on a pass or an underpowered pass.

## Central-interval coverage

Install `op_infer[study]` and pass the same successful replicate bank to
`coverage_summary`. Failures remain in the attempt count and failure fraction.
Coverage intervals describe successful fits; they do not silently treat failed
fits as posterior draws. A bank with no successes has unavailable coverage.

```python
from op_infer.calibrate import coverage_summary

coverage = coverage_summary(pits, n_failed=diag["n_failed"], levels=(0.5, 0.9))
write_ledger("sbc/coverage.json", coverage, meta={"seed": 7, "stage": "validation"})
```

Every row includes nominal coverage, successes, usable trials, the empirical
proportion, interval bounds, confidence and method. The intervals are marginal
Wilson intervals supplied by trade-study. Use independent replicates from a
fixed simulation design, and randomize PITs for quantities with atoms. The
complete bank must supply every estimand. Finite posterior draws can discretize
PITs, so Monte Carlo resolution belongs in the validation design.

Plan interval precision with `trade_study.replicates_for_half_width`. At 95%
confidence, a worst-case Wilson half-width of 0.05 requires 381 replicates.
This sizes marginal interval precision. For a fixed family that rejects on any
empirical-coverage violation, use the [coverage planner](coverage-planning.md)
to evaluate binomial rejection probabilities and a conservative family bound.
The same guide covers optional reference TARP curves. Failure policies, sampler
diagnostics and other combined criteria still require separate qualification
under calibrated and misspecified posteriors. Repeatedly checking a fixed-sample
interval until it passes does not preserve its nominal coverage.

Saved scalar criteria and amendments are described in [gate records](gates.md).
They do not select or qualify statistical thresholds automatically.
