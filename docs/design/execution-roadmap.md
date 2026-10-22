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
2. The compiler's schema-1 structural identity now flows through schema-2 guarded
   provenance; legacy schema-1 provenance remains unchanged. Keep model structure separate
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

The [scoped implementation](../identifiability.md) now copies existing compiler
scaling reports and consumes the engine's measured forward sensitivities. Native
bounded Gaussian mean information retains physical units and design/source/
initialization/observation/numerical references. Explicit host spectra preserve
rank tolerances. This does not complete general transformed-coordinate Fisher
information, nuisance profiling, posterior curvature or external rank tooling.

The [saved reading policy](../calibration-readings.md) separately records point
and interval criteria, a prospective governing choice, complete replicate banks,
usable floors, failure budgets and must-pass quantities. Its public finite-sample
qualification uses independent synthetic coverage counts. Application policy,
metric assumptions and end-to-end calibration remain consumer responsibilities.

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

## Completed execution evidence and next work

The [CPU pipeline comparison](../pipeline-performance.md) now measures simulation,
likelihood and gradients with complete/censored worker records. Grouped JAX
observation scans remove repeated tracing in those fixed-forward cases. Existing
NumPy/JAX/Torch execution remains the baseline; these measurements do not justify
a new Rust implementation or qualify offline training performance.

The [CUDA qualification](../cuda-qualification.md) covers actual A100 JAX/Torch
statistical families, dtype/device, replay and gradients. Compiler/engine core
CUDA checks additionally cover CuPy fixed-step execution, which does not imply
CuPy distribution or estimator support. GPU fitting and application memory/epoch
claims still need their own evidence.

Reviewed derivative/export/sensitivity/device sources are
[op_system ad3c28a](https://github.com/ACCIDDA/op_system/commit/ad3c28ac63e9af3b1210413a764b44a9c9bd2074)
and [op_engine 2b8d46a](https://github.com/ACCIDDA/op_engine/commit/2b8d46a2b7a6c3ee2645c4d595be7d224c52433a).
These are source qualification pins, not new published package versions.

Next, profile representative structured/coupled consumers before selecting
[optional Numba lowering](https://github.com/ACCIDDA/op_system/issues/327) or a
[Torch sequential/compile adapter](https://github.com/ACCIDDA/op_engine/issues/257).
Retain compilation break-even, memory, native gradients and explicit unsupported
cases. The [Rust experiment](https://github.com/ACCIDDA/op_system/issues/329) stays
behind a measured bottleneck and coarse boundary decision. Broader derivative
structure, coordinate transformations and statistical studies keep their own
acceptance criteria; a local matrix or generic gate is not a shortcut to them.
