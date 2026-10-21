# Implemented capabilities and executable examples

`op_infer` is pre-alpha and unreleased, with PyPI publication on hold. The architecture note describes the
intended statistical layer; the table below describes the implementation.
Adapters wrap native libraries and the `study` extra delegates orchestration to
trade-study. No sampler, scoring-rule library or executor is reimplemented here.

| Capability | Available implementation | Boundary |
| --- | --- | --- |
| Provenance and forward artifacts | `ArtifactReference`, `Provenance`, target/numerical references, `ForwardRunReference` | Content identity and guarded loading; metadata alone is not numerical qualification. |
| Identified forward models | `ForwardModel`, `CallableForward`, `ForwardDesign`, optional `adapters.op.compiled_forward` | [Forward models](forward-models.md); schema-2 structural provenance, qualified fixed explicit CPU subset and observation-only retention. Legacy schema-1 identities remain unchanged. |
| Statistical priors and observations | `JointPrior`, five scalar prior families, `IIDObservation`, `StatisticalModel` | [Shared statistical models](statistical-models.md); independent fixed-shaped priors, seven iid observation laws, Normal clipping/censoring, NumPy/JAX/Torch scoring and [scoped CUDA qualification](cuda-qualification.md). General transforms/hierarchy/dependence remain planned. |
| Fixed-forward execution baseline | Grouped observation-interval scans and a fresh-process CPU pipeline harness | [Pipeline measurements](pipeline-performance.md); 60 successful grouped workers, separate compilation/warm/process/XLA memory, retained baseline censoring; no training/GPU claim. |
| Native posterior containers | Named dimensions/coordinates, separate chain/seed semantics | [Posteriors](posteriors.md); optional `arviz` adds guarded ArviZ 1.x/netCDF snapshots. |
| Random streams | `RandomKey`, `SimulationStreams`, native NumPy and optional JAX/Torch lowering | [Streams](random-streams.md); task addresses separate parameter and simulation randomness. |
| Native NUTS | `adapters.numpyro.fit_nuts`, `NutsSettings` | `numpyro` extra; bounded CPU scope and raw diagnostics, not automatic adequacy certification. |
| Named MCMC diagnostics | `adapters.arviz.diagnose_mcmc`, `MCMCDiagnostics` | `diagnostics` extra; [rank R-hat and absolute bulk/tail ESS](mcmc-diagnostics.md), original axes and unavailable elements retained; no adequacy thresholds. |
| Masked-cohort NPE | `adapters.sbi`, `NpeSettings`, `NpeEnsemble`, guarded native save/load | `sbi` extra; CPU, declared masks/counts, full real-vector support; see [NPE](sbi-npe.md). |
| Large NPE banks and training resume | `fit_npe` borrowed-bank/checkpoint options | Qualified sbi 0.27.0 CPU boundary; removes full observation copies and preserves native training state, with exact bank/runtime guards. |
| Seeded criteria | `gates.Gate`, `GateSet`, `GateRecord` (also in `calibrate`) | [Saved gate records](gates.md): adequacy versus decision criteria, context checks, seeded callbacks and amendment disclosures. |
| Coverage summaries | `calibrate.coverage_summary` | `study` extra; central-interval coverage, marginal Wilson intervals and explicit failed-attempt counts. |
| Coverage-rule planning | `CoverageRule`, `coverage_operating_characteristics`, `plan_coverage_replicates` | [Coverage planning](coverage-planning.md); fixed successful bank, inclusive bounds and a conservative union bound over the full supplied family. |
| TARP curves | `adapters.tarp.tarp_coverage` | `tarp` extra; [reference curves](coverage-planning.md) with explicit points/coordinates, retained row failures and no automatic gate reading. |
| Rank calibration | `calibrate.pit`, `aggregate`, `gate`, rank artifacts | [Calibration](calibration.md): atom-aware ranks, multiplicity, power floors, failure accounting and contraction. |
| Fixed-bank SBC execution | `sbc.run_sbc`, `SBCEstimand`, `SBCFit` | `study` extra; [runner](sbc-runner.md), stable IDs, guarded resume and retained failures. |
| Masked-unit SBC | `masked_sbc.MaskedSBCPlan`, `run_masked_sbc`, `summarize_masked_sbc` | `study` extra; [masked banks](masked-sbc.md), complete planned IDs, strata and independent-baseline width comparisons. No automatic informativeness verdict. |
| Held-out pipeline recovery | `design.DesignStudy` and `compare.paired_predictive_contrast` | `study` extra; [design studies](design-studies.md), real refits, aligned units, supplied gate evidence and fixed independent cohorts. |
| Numerical likelihood/gradient checks | Provenance records and optional synthetic reference tests | [Numerical provenance](design/numerical-provenance.md); limited to declared references/backends and tolerances. |
| flepimop2 workflows | Separate GPL `flepimop2-op_infer` process provider | Executes calibration/SBC/design plans; the MIT core never imports flepimop2. |

General parameter/transform catalogs, correlated or hierarchical priors,
AR(1)/general interval-censored observations, integrated unit likelihood
algorithms, LOO/PSIS, automatic general gate qualification, Fisher/profile
or structural identifiability, other estimator adapters and EVSI remain planned.
The presence of an extra or a placeholder module is not an implemented interface.

## Run an analytic SBC bank

From a checkout:

```bash
uv sync --dev --extra study
just example-sbc /tmp/normal-sbc
just doctest
```

[`examples/normal_sbc.py`](https://github.com/ACCIDDA/op_infer/blob/main/examples/normal_sbc.py)
uses the exact posterior for `theta ~ Normal(0, 1)` with four unit-variance
observations: mean `sum(y)/5`, variance `1/5`. It draws through NumPy's native
normal generator, schedules 64 independently addressed replicates via trade-study,
records full source/estimator provenance, resumes the same bank, and aggregates
its ranks. Artifacts stay in the supplied directory. Scientific revisions require
a new directory; resume refuses changed provenance.

The result is explicitly underpowered. An unflagged demonstration is not
calibration evidence. The runner does not choose adequate replicate counts or
certify its own gate's false-failure rate. A fixed Wilson precision calculation
in trade-study addresses marginal precision, not gate power.

For native sampling, run `just example-numpyro` after installing `numpyro` and
follow the [NumPyro guide](numpyro.md). Design recovery has an independent,
executable normal-normal refit and proper held-out log-score fixture in
[`tests/test_design_study.py`](https://github.com/ACCIDDA/op_infer/blob/main/tests/test_design_study.py).
It checks separation of training/held-out data, three-way outcomes, failures and
shard replay. Its synthetic gate fixtures are not consumer calibration evidence.

For a table of named raw MCMC diagnostics, install `diagnostics` and run
`python examples/mcmc_diagnostics.py`. The [diagnostic guide](mcmc-diagnostics.md)
explains the deliberately shifted, constant and invalid quantities in that example.

## What runs automatically

`just ci` includes lint, strict typing, library doctests, tests and a strict docs
build. Public docstrings cover random keys/provenance, rank/gate bookkeeping,
estimand transformation, paired unit contrasts and adapter settings. Default
core checks do not require heavy optional fitting dependencies.

`just doctest` runs the core examples. `just qualify-study` requires trade-study
and executes the standalone SBC example test plus the runner/design tests.
CI runs that qualification at current and minimum trade-study versions. Separate
required jobs perform real NumPyro and sbi CPU fitting, native JAX/Torch checks,
ArviZ I/O and process-provider wheel tests. A skipped backend is not qualification.
`just qualify-diagnostics` requires ArviZ Stats and NumPyro and compares named
modern diagnostics with the reference library on synthetic and actual native draws.
`just qualify-coverage` requires the TARP reference, tests fixed-coverage family
planning and actual curves, and checks the generated table/figure inputs and exports.

Keep examples executable and check an independent mathematical result. Retain
provenance, failed attempts, masks, independent units and stage names in examples
as in production workflows. Run `just ci` before a PR.
