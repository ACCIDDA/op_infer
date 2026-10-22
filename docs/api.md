# API

`posterior` implements immutable provenance/target/numerical references and a guarded provenance JSON
loader, plus coordinate-aware native posterior samples. `forward` attaches that shared provenance to
immutable output references. The optional ArviZ adapter implements guarded DataTree/netCDF snapshots.
`rng` implements versioned task stream addresses and default parameter/simulation separation;
its optional adapters lower to native JAX keys and validated Torch CPU generator banks.
The optional NumPyro adapter implements a bounded native NUTS fit with shared posterior
records and raw diagnostic outcomes. `calibrate` implements rank-based SBC aggregation (ordinary and
randomized-rank PIT, Benjamini-Hochberg multiplicity, power floors, failed-replicate accounting and
contraction), plus study-extra central-coverage summaries with marginal Wilson intervals (see [Rank-based calibration](calibration.md)). The native sbi adapter adds masked-cohort NPE and seed ensembles. Prespecified scalar gate criteria and records are available from `gates`
and re-exported by `calibrate`. Other statistical modules/adapters remain planned. `sbc` schedules replayable fixed banks and
`design` composes held-out refits with aligned unit-score contrasts from `compare`.
See [implemented capabilities](capabilities.md) for the current scope and extras.

::: op_infer.params

::: op_infer.priors

::: op_infer.observe

::: op_infer.forward

::: op_infer.adapters.op

::: op_infer.likelihood

::: op_infer.estimators

::: op_infer.posterior

::: op_infer.adapters.arviz

::: op_infer.adapters.rng

::: op_infer.adapters.numpyro

::: op_infer.calibrate

::: op_infer.coverage

::: op_infer.adapters.tarp

::: op_infer.compare

::: op_infer.identify

::: op_infer.adapters.op_identify

::: op_infer.design

::: op_infer.rng

::: op_infer.registry


## SBC execution

::: op_infer.sbc

::: op_infer.masked_sbc

## Native masked NPE

::: op_infer.adapters.sbi

## Saved gate records

::: op_infer.gates

## Shared statistical models

::: op_infer.priors

::: op_infer.observe

::: op_infer.simulation
