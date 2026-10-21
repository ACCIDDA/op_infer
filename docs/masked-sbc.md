# Masked-unit SBC and width comparisons

`op_infer.masked_sbc` tests a reporting estimator on simulated cohorts with one
unit's observations removed. It reuses `run_sbc` and trade-study scheduling.
The full prespecified bank retains every planned ID; disjoint shards share one
contract. Summaries report overall and per-stratum calibration and posterior
widths, with separate counts for unavailable cross-method comparisons.

```python
from op_infer.masked_sbc import (
    MaskedSBCPlan,
    MaskedUnit,
    run_masked_sbc,
    summarize_masked_sbc,
)

plan = MaskedSBCPlan(
    units=(
        MaskedUnit(name="u1", stratum="A"),
        MaskedUnit(name="u2", stratum="A"),
        MaskedUnit(name="u3", stratum="B"),
    ),
    replicate_ids=tuple(range(1000)),
    seed=2026,
)
# Supply project callbacks, estimands, observations and complete provenance.
execution = run_masked_sbc(
    prior_sampler,
    simulator,
    masked_estimator,
    plan=plan,
    estimands=estimands,
    observed=observed,
    directory="masked-bank",
    provenance=provenance,
)
summary = summarize_masked_sbc("masked-bank", expected=execution.contract)
```

The example budget above is an explicit input, not a universal qualification
budget. Choose it for the full reporting rule and each required stratum. Use the
[coverage planner](coverage-planning.md) for a fixed coverage family; other
criteria and combined gates need their own operating-characteristic checks.

## Masks, units and fixed bank identities

Unit order must match the observation array's first axis. Provide the observed-cell
mask explicitly with `observed_mask`, or use the finite observed cells. At least
two units must have observed cells. Removing a unit sets all its mask entries to
false and preserves other missingness. The optional `plan.stratum` restricts which
units can be removed. Units without observed cells are not eligible for removal.

The stable replicate ID modulo the eligible pool length selects a unit. The
selection uses no simulated truths or fit outcomes and does not depend on workers,
shard order or completed tasks. Arbitrary ID sets may yield unequal allocations;
the report retains actual planned counts for every stratum, including zero-plan
strata. The overall reading describes that declared allocation, not an inferred
population weighting.

The simulator generates the full cohort with the same prior/support/acceptance
rule used for training and reporting. The estimator callback has signature
`(data, mask, key) -> SBCFit`. It receives full original-shaped data and an explicit
boolean cell mask and must honor the mask. The runner supplies copies; it does
not choose NaNs, delete/reorder rows or adapt a native inference model. Project
callbacks own any transfer, feature encoding or native observation masking. Data
identical to the actual observations on the retained cells are refused before
fitting, even if the removed unit differs.

The ordinary runner's parameter, simulation, reporting-estimator and rank streams
are retained. MCMC fits still require explicit adequacy and finite ESS; widths and
ranks use the same retained/thinned draws. No additional diagnostic threshold or
sampler implementation is introduced. Use `atoms=True` for estimands with point
masses and a positive prior SD on their reported scale.

To run a shard, pass `replicate_ids` as a subset of `plan.replicate_ids`. Keep the
complete plan, seed and directory unchanged across shards. Existing successful
and failed records are reused; baseline failures are not silently retried.
`n_jobs` delegates parallel work within each unit group to trade-study.

## Cross-method baselines and informativeness

Supply both `baseline` and `baseline_provenance` for a comparison estimator. Its
callback uses the same `(data, mask, key)` signature and receives an independent
`estimator.baseline` stream. Baseline prior, design, simulator and intended target
must match the reporting context. Different algorithm/numerical settings remain
recorded in the separate complete provenance. The project must establish that
the methods, priors, scales and masked training supports are scientifically
comparable; metadata checks cannot establish that on their own.

Each successful reporting fit retains per-estimand posterior SD and prior SD.
Available baselines add their SD and supplied fit diagnostics, including MCMC
adequacy/ESS and the number of retained draws. An exception, inadequate fit or
unserializable baseline diagnostic leaves an unavailable comparison with a reason;
the reporting fit's calibration result remains available. Reporting failures stay
failed and keep their planned comparison denominator.

Summaries show mean/min/max posterior-to-prior SD ratios, posterior-to-baseline SD
ratios and paired SD differences. They retain planned, available and unavailable
counts. A zero baseline SD retains its actual width and difference while its ratio
is undefined; nonfinite ratios are also counted separately. Without a baseline,
all baseline comparisons are explicitly `not_requested`. These are raw width
summaries. Projects choose any narrower-than-prior or no-wider-than-baseline
criteria before seeing results and can express scalar readings through
[saved gate records](gates.md). Average widths do not establish calibration.

## Guarded summaries and incomplete groups

Retain `execution.contract` independently. It identifies the complete plan,
observed values/masks, reporting and baseline provenance, estimands, installed
versions and runner source identities. The root manifest uses a `.meta` file;
per-unit numbered directories contain the ordinary guarded SBC manifests and
replicate files. Unit design references retain the parent context and declared
mask. Caller provenance must identify the entire masked reporting procedure,
including callbacks, transformations and acceptance rules.

The summary checks the expected root identity, per-unit contracts, record IDs,
estimand/width metadata and contraction consistency. It always reads the complete
planned bank. Missing records remain unfinished; groups with missing tasks have
no rank reading. All-failed groups retain full failure counts and unavailable
calibration. Zero-plan strata remain visible. Raw width summaries from available
rows retain their unavailable denominators.

Rank aggregation and its existing power-floor/BH settings are reused within each
group. Overall and stratum summaries overlap, so these adjustments do not qualify
a joint gate across groups. `adequacy_status` stays `not_assessed`. Self-recorded
hashes establish context/integrity against the supplied reference; they do not
authenticate artifact producers, prove that a plan preceded results or establish
scientific validity. Retain referenced payloads and external registration evidence
as appropriate.

## Runnable analytic example

```bash
uv sync --dev --extra study
just example-masked /tmp/masked-normal-sbc
just qualify-masked
```

The [complete example](https://github.com/ACCIDDA/op_infer/blob/main/examples/masked_normal_sbc.py)
uses `theta ~ Normal(0, 1)` with four independent observations whose SDs are
`(1, 1, 2, 2)`. The exact masked posterior has variance
`1 / (1 + sum(1 / sigma_i**2))` over retained cells and mean
`variance * sum(y_i / sigma_i**2)`. It also reports `max(theta, 0)`, with randomized
PITs for its atom. The baseline draws independently from the same exact posterior.
Its table includes all groups/estimands and comparison counts, and its 64-replicate
bank resumes unchanged. It is deliberately underpowered.

The required current/minimum-study jobs run the example and real scheduling tests.
Checks cover analytic width/rank references, strata, atom-aware PITs, parallel/shard
replay, missing and failed records, baseline exceptions/inadequacy/serialization,
zero baseline widths and refused contract/metadata mismatches. Missing optional
dependencies fail the qualification recipe rather than count as skipped success.
