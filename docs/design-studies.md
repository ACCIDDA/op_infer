# Fixed-bank design recovery

`DesignStudy` provides fixed-bank simulate/refit/predict comparisons, scheduled by
`trade_study.run_grid`. Install `op_infer[study]` (trade-study >=0.4). It returns
trade-study `ResultsTable` summaries, suitable for its native `extract_front`.
The core defines no scoring rule, optimizer, sampler or executor.

Declare candidate `models`, named `DesignDefinition(observe, cost)` entries, a
`generate(model, parameter_key, simulation_key)` callback, a design-aware
`fit(model, design, training_data, key)` callback and a proper predictive
`score(model, fit, held_out, key)` callback. A generator makes one full cohort;
each design projects that same cohort into `DesignObservation(training,
held_out, target)`. Parameter and simulation keys are independent; the full
cohort is shared across designs for common random numbers. Preserve the training
acceptance rule and likelihood support in these callbacks and their provenance.
The workflow introduces no additional filtering.

The caller must ensure the fitting and held-out observations are separated,
that every alternative is genuinely refitted, and that design covariates/counts
are available to each estimator. `DesignFit` retains the full posterior (including
any inferred end-of-window state), its estimator reference, diagnostics and an
explicit adequacy verdict. Each design/model needs a `DesignGate` with exact
estimator and calibration artifact references. Only `pass` permits execution;
`underpowered pass` remains unavailable. These are caller-supplied evidence
references, not automatically verified calibration or the general Gate API.
A changed estimator requires new evidence/provenance and a new output directory.

Scores are `UnitPredictiveScores(values, unit_ids, target, unit_kind)` with one
held-out log predictive density per declared independent unit or independent
cluster. Integrating shared parameters/latent states and qualifying numerical
or Monte Carlo approximations belongs to the scoring callback. Unit identities,
score semantics and held-out targets must match across models. The paired reading
rule uses the mean score difference and its sample standard error; above +2 SE
favors the generator, below -2 SE favors the alternative, and the middle is
indistinguishable. `threshold_se` is prespecified. This is a reading rule, with
operating characteristics measured by repeated independent cohorts; it does not
claim an exact test level. Nonfinite scores/arithmetic and fewer than two units
produce a failed contrast. No bad unit is discarded.

`study.run(directory, replicate_ids=[...], seed=..., n_jobs=...)` writes one
record per design/generating-model/cohort, including per-alternative contrasts,
unit scores and fit diagnostics. Separate processes can run disjoint ID shards.
Stable addresses and an immutable manifest support replay independent of bank
order or shard size. Duplicate workers are refused by per-record locks; a crashed
worker's stale lock requires investigation before deliberate removal. Failures
are complete records and are not retried on resume. The caller's full Provenance,
including code/callback revisions, must identify callback changes.

`study.summarize(directory, replicate_ids=planned_ids, seed=...)` includes every
planned ID, counting missing records as failed. It reports favor-generator,
favor-alternative, indistinguishable and failed fractions plus design cost.
The joint rule favors the generator only if every alternative loses, favors an
alternative if at least one wins, and otherwise is indistinguishable; any failed
contrast makes the joint trial failed. Marginal Wilson intervals come from
trade-study. The probability is over all prespecified pipeline trials, including
failures; it is not conditioned on successful fitting. One independent generated
cohort is one replicate. Bootstrap resamples are not extra independent cohorts.
Intervals are marginal; the joint rule does not certify multiplicity control or
gate power. Per-alternative results remain in the underlying records.

Qualify the full pipeline on the intended model/design range before interpreting
recovery. The tests exercise exact normal-normal refits and SciPy predictive
scores, three-way outcomes, aligned units, nonfinite failures, powered-bank
bookkeeping, shared-cohort replay, guarded resume and underpowered gates. LOO/PSIS
recovery, automatic gate-driven retraining, replicate sizing, paired
information-ordering diagnostics, Fisher screens and EVSI remain open scope.

Plan the fixed cohort bank with native trade-study
`replicates_for_half_width(half_width, confidence=...)`, then pass the complete
`range(n)` to execution and summary. This plans worst-case marginal Wilson
interval precision; it is not statistical power or a sequential stopping rule.
Use the same independent cohort IDs for every paired design.
