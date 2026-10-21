# Identified forward models

The core `ForwardModel` protocol maps `(theta, design, key)` to **named latent
outputs**, with ordered parameter/output declarations and shared `Provenance`.
`CallableForward` adapts a plain function; `ForwardDesign` carries the design's
independently supplied `ArtifactReference` beside borrowed native inputs. The
callback receives `(theta, design.inputs, key)` unchanged. No parameter sampling
or statistical observation noise is implicit. Use `SimulationStreams.parameters`
for parameter draws and `.simulation` for the forward call.

Callables declare their own model-content manifest. The library does not infer
identity from callable names, repr, bytecode or captured values. A reference
is content bookkeeping, not evidence that supplied arrays or callbacks match
their labels. Keep borrowed inputs immutable and verify persisted bytes before
reuse. Changed design identity, missing/extra parameters and changed output names
or order fail explicitly. Native values and graphs are not copied to host.

## Compiler and fixed-engine adapter

`adapters.op.compiled_forward` binds an existing compiled model, an initial-state
factory and selected compiled observables to OP Engine's public functional
`fixed_explicit_step` kernels and namespace loop adapters. No numerical method is
reimplemented. The first supported subset is deterministic fixed Euler, Heun,
RK4 and DOPRI5 in natural coordinates, without operators, history, input tables or
gauge fixing. Those compositions fail explicitly; live adaptive, stochastic,
steady-state and provider adapters need separate qualification.

`FixedForwardSettings` declares static observation times (including the initial
time), maximum internal step, method, float32/float64 state dtype, loop choice and a complete-partition work
limit. Setup rejects oversized partitions before asking the engine to allocate
step tuples. `auto` selects an available namespace loop adapter; `unroll` uses
eager stepping; forced `scan` fails if no adapter is available. `checkpoint=True`
requires forced scan and available rematerialization. The settings, actual
engine-planned mesh and selected latent outputs each have separate reference
helpers; the constructor checks them against independently supplied numerical
provenance, along with the compiler structural identity.
The initial-state dtype must match the declared precision; no implicit cast hides
a precision change under the same numerical identity.

The initial-state callback receives dynamic named parameters and design inputs.
It returns a native real floating flat state in compiler state order. Declare
all RHS, observable-only and initial-state parameters in `parameter_names`.
Only selected named observation series are returned at requested solver nodes;
there is no temporal interpolation or added weighting/noise. The engine's
existing observation adapter retains compiled output shapes, coordinates and
physical measures.

The driver carries current state/FSAL and stores observations. Its scan body has
no stacked state-history output. Storage for results still scales with requested
outputs; reverse-mode AD can retain additional residuals. Checkpointing is an
option, not a guaranteed application memory reduction. The qualified CPU cases
cover NumPy values, JAX JIT/reused bounded `vmap`/AD, and Torch AD in float32 and
float64. CUDA/CuPy and other devices need independent qualification. Batch size
and chunking remain explicit caller choices; no unbounded batching API is added.

## Reproduce the complete example

The compiler identity API is on reviewed op_system main, beyond the archived
0.9.0 tag. Until a release contains it, use the exact qualified source revision:

```shell
uv sync --dev --extra study --extra op --extra jax
uv pip install 'op-system @ git+https://github.com/ACCIDDA/op_system@4c054694f7cfd4774a3e9abfdefc93551d4f7126'
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
just qualify-forward
```

[`examples/compiled_forward.py`](https://github.com/ACCIDDA/op_infer/blob/main/examples/compiled_forward.py)
constructs complete, explicitly unqualified execution metadata, returns
`6*exp(-0.7*t)` within a declared finite tolerance and prints its identities.
It is a forward-only example, with no prior, likelihood fit or posterior.
Application provenance must record its actual prior/target/design, derivative
scope, source revisions and dependencies; this example's source reference is
only the illustrative program. Qualification also exercises an actual
trade-study `Simulator.generate`/`run_grid` call and guarded checkpoint replay.
Supply complete provenance identity as the caller-managed checkpoint revision;
names or class source alone do not identify instance/model state.

## Structural identity and legacy artifacts

Provenance **schema 2** requires `structural_model: ArtifactReference`. Compiled
models reuse OP System's exact manifest digest with kind
`op-system-structural-model-v1`; unknown compiler schema kinds are refused.
Callables can reference a declared implementation/layout manifest instead.
For multiple-model studies, reference an explicit ordered model registry
containing each constituent identity, rather than labeling different models as
one compiled artifact. Model structure is separate from fitted values,
prior/design/statistical observation identity, numerical policies, estimator
stage and runtime/source versions.

`existing.with_structural_model(independently_supplied_model)` explicitly creates
a new validated schema-2 record. It does **not** relabel existing saved artifacts
or establish which model produced them. Legacy schema-1 records remain readable
only against independently expected schema-1 provenance; their serialized shape
and complete identity are preserved, and they carry no structural claim. A legacy
record does not match an expected schema-2 record. Missing model references,
unknown schema versions and meaningful mismatches fail before reuse.

The same shared record flows through forward references, simulation-bank
manifests, native NPE training checkpoints/fits/ensembles, posterior containers,
ArviZ/netCDF snapshots and calibration/selection contracts. Existing loaders and
resume guards compare the independently expected complete provenance. This adds
no second artifact identity system, authenticated publication or general shard
merger; broader merging stays with the existing provenance work.
