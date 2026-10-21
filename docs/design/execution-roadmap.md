# Statistical interfaces and execution roadmap

Engineering decisions recorded on 2026-10-21. Consult the
[capability guide](../capabilities.md) for implemented behavior; this plan does not
turn reserved extras or placeholder modules into supported interfaces. The shared
[execution roadmap](https://github.com/ACCIDDA/op_engine/blob/main/docs/development/execution-roadmap.md)
and [baseline protocol](https://github.com/ACCIDDA/op_engine/blob/main/docs/development/execution-baselines.md)
define numerical ownership and measurement boundaries.

## First interfaces

1. The [identified forward protocol](../forward-models.md), callable adapter and
   bounded compiled fixed-step CPU adapter are implemented. Extend other solver
   modes only with their own layout, output, derivative and execution evidence.
2. Schema-2 structural identity now flows through existing guarded provenance;
   legacy schema-1 identities remain unchanged. Keep model structure separate
   from fitted values, numerical settings, prior/design and estimator identity.
3. The [shared statistical models](../statistical-models.md) now supply five
   independent prior families, seven iid observation families, Normal censoring,
   native lowerings and an identified forward-to-measurement composition. General
   transforms, correlated/hierarchical priors, AR(1) and interval censoring remain
   scoped extensions under the existing family issues.
4. Require NumPy/JAX/CPU Torch family-specific support, parameterization, moments,
   log-density and derivative conformance before claiming interchangeability.
5. Qualify complete JAX simulation-to-observation execution, bounded batches and
   explicit device transfers before adding native acceleration implementations.

Compiled latent observables are owned by OP System and numerical observation
evaluation/reduction by OP Engine. Statistical noise, censoring, masks and prior
semantics belong here. Samplers remain native adapters to established libraries;
Rust is not a new inference implementation target. Julia structural-identifiability
integration is optional interoperability, not a required runtime or package port.

Native arrays stay native inside numerical code. Host snapshots and interchange
are explicit adapter/serialization boundaries; DLPack does not by itself preserve
autodifferentiation graphs. Existing bounded CPU fitting contracts do not qualify
GPU fitting or all backend operations.

## Identifiability contracts

Consume the compiler's existing scaling-gauge reports and coordinate maps. Keep
unassessed/unsupported structural cases distinct from a proved absence of gauges.
Finite-design Fisher spectra use measured-output parameter sensitivities in stated
coordinates and noise units. The RHS state Jacobian is not that sensitivity map;
small Fisher eigenvalues alone do not prove a global structural symmetry.

Dimension reports must state whether they count nominal coordinates, a reported
gauge quotient or an independently established identifiable dimension. Do not
automatically identify these counts or turn them into a model-selection penalty.

Prior transformations preserve the induced distribution and record support,
Jacobians and any gauge fixing. Requiring a proper prior to be constant along an
unbounded gauge orbit is not a general conformance rule. Compare matched estimands
under matched prior/observation targets after coordinate changes.

The implemented `posterior_contraction` returns posterior SD divided by prior SD:
values near one indicate little narrowing. A gate on insufficient narrowing must
respect that direction and explicitly declare its threshold and coordinates.
Contraction complements calibration and sampler adequacy; it does not alone
establish a symmetry or certify inference accuracy.

## Downstream integration

Use trade-study for replicate loops and scoring. A Fisher constraint is a scoped
screen with declared nuisance treatment, coordinates and tolerances. Predictive
stacking does not receive an automatic BIC-style dimension penalty. A proposed
alternative requires a separately specified objective and numerical/statistical
qualification.

Decision-theoretic EVSI needs actions, utilities, a future-data law and posterior
updating. Surrogate expected improvement is a different acquisition target.
General surrogates and execution stay with trade-study; statistical updating and
design targets stay here. Forecast artifacts and vintage-aware evaluation remain
subject to OP Forecast's explicit consumer/start gate.

## Memory and runtime qualification

Offline `fit_npe` epochs consume a supplied bank and do not invoke the ODE solver.
Measure training, bank generation, posterior sampling and calibration simulation
as separate phases. Solver settings can affect training data, but that does not
establish solver calls as a cause of slow offline epochs.

`borrow_bank=True` retains contiguous CPU float32 banks, including copy-on-write
maps, with chunked validation and hashing at the qualified sbi 0.27.0 boundary.
Parameter standardization, minibatches, masks, activations and optimizer states
still allocate memory. Native checkpoints retain model/optimizer state, split,
losses, RNG and bank identity; resume requires the same immutable ordered bank.

Use bounded synthetic banks for controlled comparisons. Hold data, seed, split,
fixed-epoch work and dependency versions constant while varying placement,
explicit thread budgets and local/shared storage. Record affinity, NUMA, thread
pools, page faults, memory pressure and seconds per batch/epoch separately from
epochs to stop. Allocated CPUs and the intended training thread budget need
separate records, since memory requirements can affect scheduler allocation.

Report RSS/PSS/USS where available, file-backed versus anonymous memory, cgroup
peaks and checkpoint/atomic-save disk peaks. Compare ordinary, borrowed and mapped
banks and force a resume to check equivalence. Any later coordinate or solver
change requires its own target, accuracy and calibration qualification.

These are measurement requirements, not evidence of a production speedup, GPU
support or a particular cause of runtime variation.
