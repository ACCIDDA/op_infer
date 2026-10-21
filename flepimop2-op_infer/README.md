# flepimop2-op_infer

This GPL-3.0-or-later provider registers `flepimop2.process.op_infer.InferProcess`.
The MIT `op_infer` core imports no flepimop2 code. The provider is
unreleased alongside the core; install both from the same checkout/revision.
Provider tests cover public flepimop2 0.3.0 and a pinned upstream revision.

| Operation | Project-supplied input | Provider/core output |
| --- | --- | --- |
| `sbc` | An `SBCWorkflow` factory, observations, provenance and explicit replicate IDs | Guarded rank records and execution/failure ledger |
| `design` | A complete core `DesignStudy` factory | Named study artifacts with cohort and failure accounting |
| `calibrate` | Completed rank directory, exact IDs and independently supplied provenance | Aggregate/gate report outside the replicate directory |

The [illustrated exact-normal example](https://github.com/ACCIDDA/op_infer/blob/main/docs/sbc-runner.md#illustrated-analytic-fixture)
shows the core SBC ledger and rank inputs behind these operations. Its
64-replicate demonstration remains underpowered; provider execution does not
change that statistical interpretation.

The process runs `sbc`, `design`, or `calibrate`. SBC/design factories are ordinary
importable project functions returning `SBCWorkflow` or core `DesignStudy`.
Factories receive configured `parameters` and own model construction, observation
support, acceptance rules, `SystemProtocol.bind()` / `EngineProtocol.run()` and
reporting-estimator callbacks. Project factories construct models and callbacks;
the provider reads configuration and manages artifacts. It implements no
sampler, scoring rule, generic model DSL or scheduler.

```yaml
process:
  calibration_bank:
    module: op_infer
    operation: sbc
    factory: my_project.workflows:calibration
    parameters: {scenario: baseline}
    directory: artifacts/calibration-bank
    replicate_ids: [0, 1, 2, 3]
    seed: 2026
    n_jobs: 1
```

`SBCWorkflow` contains `prior_sampler`, `simulator`, `estimator`, `estimands`,
`observed`, `provenance` and optional `mask`, which map directly to core `run_sbc`.
For design studies change `operation` to `design` and return a complete core
`DesignStudy` from the factory. Shared cohorts, held-out scoring, gate evidence,
failed trial accounting and resume remain core responsibilities. Named cohorts
and seeds are explicit; no success-only filtering or automatic retries occur.

To aggregate SBC, use `operation: calibrate` with the same `directory`, exact
complete `replicate_ids`, `seed`, an independently supplied full `provenance`
record, and a `report` path outside the replicate directory. Omit `factory`.
The provider refuses missing/extra rank files and mismatched manifest/replicate
contracts. It delegates ranks, multiplicity, failure accounting and the verdict
to core `calibrate`, preserving `underpowered pass`. Set `bins`, `power_floor`,
`alpha`, `max_flagged`, `must_pass` and `max_failed_fraction` explicitly when the
prespecified analysis needs different settings. The report includes the manifest,
contract, IDs, aggregate and gate result; its existence is not calibration proof.

Native process discovery uses `flepimop2.process.abc.build`. Dry runs validate
configuration and display the plan without importing/calling project factories or
writing artifacts. Normal repeated execution delegates to core guarded resume;
existing failed records remain failed. Scheduling is delegated to trade-study.
Future prior-backed parameters and broader inference DSLs remain separate issues.
