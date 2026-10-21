# Changelog

## 0.0.1 (unreleased)

### Documentation

- Record forward/provenance, shared prior/observation, backend and scoped
  identifiability interfaces, with explicit downstream statistical-method gates.

### Fixed

- Reject invalid rank/gate configuration before labels or statistics.
  Single-bin tests and NaN thresholds previously allowed false passes. Validate
  integer bin/power/flag counts and finite probabilities, check settings when
  reading older aggregates, and refuse nonfinite reference p-values. Share rank
  setting checks with masked summaries and normalize supported NumPy scalars.
  Valid settings, scientific defaults and inclusive failure limits are unchanged.

- Reject negative, fractional and boolean failed-replicate counts before rank
  aggregation. Rank and coverage summaries share count validation and
  normalize integer scalar counts, keeping total attempts and failure fractions
  consistent. Valid runner-produced counts and all-failed banks are unchanged.

### Added

- Identified forward protocol/callable adapter and an optional compiled fixed-step
  adapter using existing engine kernels and latent observations. Retain current
  state/FSAL plus selected outputs, with bounded setup, explicit native scan/JIT
  controls and NumPy/JAX/CPU Torch value/gradient qualification.
- Provenance schema 2 carries independently expected structural model content
  through existing banks, native fit/checkpoint/posterior and calibration/selection
  records. Preserve legacy schema-1 serialization and complete identities; refuse
  unknown schemas and model mismatches without a second artifact hierarchy.

- Native NPE training checkpoints and borrowed CPU float32 observation banks
  Artifacts retain their explicit qualification boundary. Preserve optimizer, early-stop,
  split, loss and RNG state before native best-weight restoration, with complete
  bank/runtime/provenance guards and atomic writes. Keep native optimization and
  parameter standardization; the latter still materializes training theta rows.
  Qualify the private sbi 0.27.0 boundary with exact uninterrupted/resumed native
  comparisons, interruption, early stopping, memory-map storage and separate
  process examples on current/minimum Torch. No scientific defaults change.

- Masked-unit SBC on the existing runner: fixed complete ID banks, explicit
  masks and strata, guarded parallel/shard resume, missing/failed denominators and
  raw posterior/prior/baseline width comparisons. Baselines use separate provenance
  and estimator streams; unavailable comparisons preserve the reporting fit's
  calibration result. Retain native adequacy evidence and undefined width ratios.
  Add an underpowered analytic table example and required current/minimum-study
  qualification. Scientific informativeness thresholds remain project policies.

- Fixed-coverage family planning through SciPy binomial tails: explicit
  inclusive rules, marginal rejection probabilities, family bounds without an
  independence assumption, and exhaustive sizing with an explicit false-failure
  target/budget. Qualify Gaussian and randomized point-mass references and retain
  unavailable plans. No universal replicate budget or general gate certification.
- Optional TARP reference curves, with explicit coordinate/reference-point
  definitions, complete planned row IDs, counted nonfinite/inadequate inputs,
  unavailable ties/errors, provenance and input/source/version identities. Restore
  the reference's NumPy global RNG side effect and declare its missing imports.
  Add executable tables/vector figures and required current/minimum qualification.

- Optional named rank-normalized R-hat and absolute bulk/tail ESS through ArviZ
  Stats. Preserve original chain/draw/event coordinates, explicit reported
  quantity definitions, all unavailable/undefined elements, dependency versions,
  adapter source and host-input digests. Add a tabulated analytic example and
  required current/minimum-reference and actual native-posterior qualification.
  No scientific thresholds or adequacy decisions are introduced; native NumPyro
  classic diagnostics remain unchanged.

- Immutable scalar Gate/GateSet/GateRecord contracts, also exported from
  `calibrate`: criteria saved before a seeded callback, checked evidence context,
  separate adequacy and decision outcomes, unavailable values, guarded JSON
  replay and prospective amendments retaining prior identity and seen evidence.
  Existing rank-gate behavior and consumer thresholds remain unchanged.

- Central-interval coverage summaries with marginal Wilson intervals delegated
  to trade-study. Reports retain successful and total attempt counts,
  failures and conditioning. The required study job checks Gaussian, biased,
  overconfident and point-mass posterior references. TARP and calibration-gate
  power qualification remain planned.

### Documentation

- Replace issue and pull-request breadcrumbs in guides and API docstrings with
  current behavior, named interfaces and explicit limitations.

- Scope distribution summaries to implemented fitting, calibration and design
  interfaces. Retain the unreleased pre-alpha development stage.

- Edit package and provider introductions for plain language. Label development
  principles separately from implemented features and remove claims that the
  planned statistical interfaces are already complete.

- Illustrate the exact-normal SBC fixture with posterior/reference and rank
  figures, a generated failure/resume ledger and retained plotted inputs.
  Keep the 64-replicate bank explicitly underpowered and check source/data/export
  identity in the required study job, with plotting outside the core.

- Document implemented versus planned capabilities; add a runnable exact-normal
  SBC/resume example, public API doctests and required optional example checks.
  These changes do not alter runtime APIs or dependencies.

- Replace the GPL inference-provider scaffold with registered `op_infer` process
  steps for SBC, fixed-bank design execution and provenance-checked calibration
  reports. Project factories own system/engine bindings and callbacks.
  The study extra installs trade-study's native parallel dependency for `n_jobs`.
  Dry runs do not import factories or write artifacts; calibration requires the
  complete prespecified ID bank. Qualify source and clean installed wheels on
  public flepimop2 0.3.0 and immutable merged upstream main. Nonserializable design
  diagnostics now produce counted failed records instead of aborting a bank.

- Add a bounded DesignStudy execution/summary API over optional trade-study:
  shared full cohorts across nested designs, separate parameter/simulation keys,
  design-aware refits, declared held-out targets, per-unit paired contrasts and a
  fixed three-way reading rule. Complete planned cohort denominators include
  failed/missing/gated trials; results carry cost and marginal Wilson intervals.
  Immutable provenance/gate contracts guard shard replay. Raises study extra to
  trade-study >=0.4 for its interval implementation. General LOO/PSIS, automatic
  gate certification, precision sizing and EVSI remain planned.

- Add a bounded SBC execution runner on optional trade-study: stable
  replicate identities and separate parameter/simulation/reporting-estimator/rank
  streams, immutable provenance/observed-data contracts, guarded shard resume,
  counted simulation/fit failures, contraction and retained fit diagnostics.
  MCMC fits require declared adequacy and ESS thinning; this does not implement
  an adequacy diagnostic. Existing rank-file readers remain compatible with
  optional `_meta` records. Required current/minimum-study CI exercises a
  calibrated-by-construction bank, replay, failure accounting and ESS bookkeeping.

- Add a bounded optional native sbi NPE adapter: mean/count set embeddings
  with per-cohort all-NaN absence and training-only removal, no dropped simulation
  rows, guarded equal-weight seed ensembles and named latent/parameter draws.
  Single-round CPU float32 fitting requires full-real parameter support for direct
  estimator sampling. Tensor-only atomic snapshots check independent provenance,
  installed versions, architecture/settings and weight integrity. Native current/
  minimum sbi/Torch CI exercises real fits, count/gradient semantics, global RNG
  restoration and exact snapshot replay. Calibration gates remain separate.

- Add rank-based SBC aggregation in `op_infer.calibrate`: ordinary and
  randomized-rank PIT, Benjamini-Hochberg multiplicity across estimands, a power
  floor with an "underpowered pass" label, failed-replicate accounting, contraction,
  the simulated-differs-from-observed assertion, atomic per-replicate JSON writing
  and a `python -m op_infer.calibrate aggregate` gate. The replicate schema is shared
  across estimator adapters. Adds `scipy` as a core dependency. Coverage/Wilson, TARP and general `Gate` records remain planned at this increment.
- Add a bounded optional native NumPyro NUTS fitting adapter: shared
  native posteriors, complete coordinate layouts, settings/RNG/version/precision
  provenance checks and retained native fit objects. Keep sampled diagnostics,
  per-fit classic split R-hat/ESS, divergence/nonfinite counts and unavailable
  outcomes distinct from adequacy gates. Add an analytic Gaussian example and
  current/minimum CPU qualification. SVI and predictive/report integration remain open.
- Keep strict typing compatible with the declared NumPy 2.0 floor, using
  portable masked-array checks, typed quadrature boundaries and full dictionary RNG
  state snapshots. Add a minimum-core CI job without relaxing strict type checks.
- Add versioned replayable seed/stream/index addresses and default parameter/simulation
  separation. Wrap SeedSequence/PCG64DXSM, lazy explicit typed Threefry JAX keys
  and local Torch CPU generator banks. Refuse effective 32-bit Torch seed collisions;
  qualify native replay, JIT/vmap, bank ordering and untouched global state in CPU CI.
  Priors, observations, estimators, GPU RNG and consumer migrations remain planned.
- Require netCDF4 >=1.7.2 in the optional ArviZ extra and qualify its minimum I/O
  dependencies in CI. The previously permitted 1.6.5 wheel cannot import
  against the package's NumPy 2 core requirement.
- Add native coordinate-aware posterior containers with explicit chain/seed semantics, sampled
  diagnostics and per-unit log likelihoods, reusing complete shared provenance. Add optional
  guarded ArviZ 1.x DataTree/netCDF snapshots, array integrity checks and explicit host transfer.
  Qualify real round trips at both supported Python endpoints and native JAX/Torch graph retention.
  No inference sampler, automatic adequacy certification, authenticated artifact publication or
  shard merger is introduced; remaining artifact contracts stay planned.
- Add immutable, versioned provenance/likelihood-target/numerical execution references and a guarded
  provenance JSON loader, shared by forward output references (the shared provenance foundation). Keep seven
  error/uncertainty sources and derivative/geometry scope distinct. Add a bounded optional op_engine
  RK4 likelihood/gradient refinement qualification on NumPy, JAX JIT and Torch, with a dedicated CPU CI
  job. General adaptive/spatial inference, full artifact loading and shard merging
  remain planned; no universal accuracy or gradient-portability claim is made.
- Define the likelihood-target and adequacy contract before inference adapters: distinguish analytic,
  deterministic, frozen-random and augmented targets; require auxiliary-state qualification, explicit
  approximation provenance, preserved importance-estimate failures and counted unit populations. Add a
  dated design amendment and independent synthetic hierarchical numerical qualification tests. Runtime
  adapters, posterior records and diagnostic policies remain planned.
- Package scaffold: MIT core with planned modules (no public API), GPL flepimop2 provider scaffold, design
  proposal, CI, docs skeleton.
