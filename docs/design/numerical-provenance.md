# Numerical execution and provenance

`posterior.Provenance` supplies immutable metadata and guarded provenance JSON
reuse, plus a static scalar integration
qualification. [Posterior containers and ArviZ/netCDF snapshots](../posteriors.md) now reuse these
records. Shard ranges/merging, authenticated artifact loading, general numerical adapters and
adaptive/spatial inference remain planned. A bounded [fixed compiled forward adapter](../forward-models.md)
now consumes these records. Current fitting adapters reuse
these same references.

## The implemented records

Import the metadata API from `op_infer.posterior`; `op_infer.forward.ForwardRunReference` associates
output bytes with that same `Provenance`. Core imports do not import op_engine, op_system, JAX, Torch,
ArviZ or flepimop2. No arrays cross backend boundaries in these records.

| Record | Meaning |
|---|---|
| `ArtifactReference` | Semantic kind and SHA-256 of immutable referenced bytes. `from_bytes` computes the digest. |
| `Qualification` | Qualified, failed or unqualified outcome, domain/scope, reason and evidence references. A qualified declaration needs evidence references; their contents still require an audit. |
| `ErrorComponent` | One of time, space/quadrature, nonlinear solve, linear solve, unit integration, Monte Carlo or sampler uncertainty; explicit settings, inapplicability or unqualified status. |
| `DerivativeContract` | Method, ordered coordinates/transforms, adaptive-path treatment, graph/geometry scope and derivative qualification. |
| `NumericalExecution` | Solver, meshes/topology, geometry, replay/refresh policy, physical observations, field/prior interpretation, derivatives and all seven error components. |
| `LikelihoodTargetReference` | Intended/evaluated identities, one of the four [target kinds](likelihood-targets.md) and scoped qualification. |
| `BackendVersion` | Exact dependency name/version; lists are nonempty, unique and sorted by name. |
| `Provenance` | Required schema version 1 (legacy) or 2 (explicit structural model), prior/design/simulator, target, numerical execution or explicit analytic absence, estimator stage/settings, dependencies and code identity. |
| `ForwardRunReference` | Immutable outputs and shared provenance; requires a numerical execution contract. |

Fields are required, including nullable references where absence must be explicit;
the additional `structural_model` is absent in schema 1 and required in schema 2.
Nested records
and tuples are immutable; extra fields, empty labels, malformed digests and unsupported/missing schema
versions are rejected. A spatial mesh requires its topology; learnable geometry requires both. A
frozen-replay derivative requires the replay artifact. Every error source appears exactly once, even
when it is unqualified or inapplicable. Controls are not achieved-error bounds, and no total error
budget is computed by adding incomparable components.

## Record the behavior, not just its name

Referenced artifacts must contain complete settings, including their own format/version. For example:

```python
from op_infer.posterior import ArtifactReference

solver = ArtifactReference.from_bytes(
    b'{"method":"rk4","max_step":0.0625,"dtype":"float64","format":1}',
    kind="solver-settings-v1",
)
```

Writers own serialization of referenced bytes: specify array dtype, shape, ordering, physical units and
coordinate interpretation, and a canonical settings representation. Different bytes conservatively
receive different references. A label such as `fine-grid`, an unresolved path or a seed does not identify
behavior. Store immutable artifacts and verify their bytes before interpreting the references. Hashes
do not authenticate provenance, establish mathematical truth or make unavailable artifacts usable.

`Provenance.identity` hashes a canonical JSON representation of the complete validated metadata. JSON
object key order and formatting do not matter; tuple order is retained. Changed tolerances, meshes,
observations, derivatives, stages, dependencies, target approximations or evidence produce a changed
identity. A semantic equivalence claim needs separate qualification; the guard remains conservative.

Given a constructed expected `Provenance` supplied independently by the caller:

```python
from op_infer.posterior import Provenance

serialized = expected.model_dump_json()
restored = Provenance.load_json(serialized, expected=expected)
restored.require_match(expected)
assert restored.identity == expected.identity
```

This loads only the provenance header, not posterior draws or any referenced payload. Missing fields,
unknown schema versions and mismatches fail before reuse. Do not construct `expected` from the untrusted
header being checked. Pydantic's low-level `model_construct` and unvalidated `model_copy(update=...)`
are not supported validation/loading paths. Full artifact availability/authenticity and shard checks
remain unimplemented. Comparisons retain separate expected identities for each model, not an equality
requirement between intentionally different models.

## Mesh, physical interpretation and derivatives

The physical observation definition and field/prior interpretation have separate references from the
numerical mesh. Refining a mesh must preserve observation times/locations, units, weights/measures,
representation (counts versus density, for example), field basis and prior interpretation, or explicitly
declare a changed inverse problem. Reuse op_system state/measure declarations and named observables,
and op_engine observation diagnostics; metadata alone does not perform those conversions.

`geometry="frozen"` means numerical coordinates are held fixed for the declared execution.
`learnable_fixed_topology` records a different parameterized map: coordinate transforms, cell measures,
observation interpolation and prior basis must be included in the graph scope. Fixed topology is
required by this increment. Neither mode qualifies changing topology or remeshing automatically.

The mesh-policy artifact specifies refresh/reuse, pilot/adaptation inputs, randomness and when changes
are allowed. A sampler cannot silently refresh its mesh and continue claiming the same frozen target.
A deterministic parameter-dependent mesh defines a different map whose regularity and derivatives need
qualification. Frozen accepted-step replay differentiates the replayed discrete map; differentiating
controller decisions or a continuous sensitivity system is a separate claim. Labels remain metadata
until independently checked. Backend transfer, including DLPack, does not promise graph continuity;
record backend-native graph boundaries and any custom derivative rule. Future mesh epochs from
adaptive numerical consumers would extend this contract only after separate qualification.

## Bounded numerical qualification

`tests/test_numerical_reference.py` composes op_engine's public RK4 fixed-step kernel for
`x'=-r x`, `x(0)=1`. Four physical concentration observations at `t=(0.5,1,1.5,2)` use fixed data
`(0.83,0.55,0.28,0.18)` and Gaussian observation standard deviation `0.2`.
The rate prior is defined on the physical scalar rate, `Uniform(0.1,2)`, and is unchanged by the
partition; the qualification compares likelihoods rather than sampling a posterior.

At rates `r=0.4,1.1`, compare internal maximum steps `0.25,0.125,0.0625` with the independent analytic
prediction `exp(-rt)` and its derivative `-t exp(-rt)`. The original observation times stay fixed.
NumPy uses central parameter differences with step `1e-5`; JAX uses JIT `value_and_grad`, and Torch
uses native autograd. Every backend uses float64. For **this fixture only**, both successive error ratios
must exceed 12; final absolute log-likelihood error must be below `3e-6`, and gradient error below
`2e-5`. These are local qualification tolerances, not posterior-distance bounds or universal defaults.
The functional step receives the native state explicitly; this fixture does not qualify full ModelCore
history storage or provider execution. No solver or sampler is reimplemented here.
Spatial/learnable geometry, dense/off-grid reconstruction,
implicit derivatives and general adaptive-mesh likelihoods remain unqualified by this fixture.

Install the `op`, `jax` and `torch` extras, then run `just qualify-numerical`. The recipe requires all
optional backend imports and runs six cases, so missing dependencies fail rather than creating a green
qualification from skips. Core `just ci` can skip these optional integration cases. A dedicated CPU CI
job installs all backends and runs the recipe, while ordinary CI covers metadata and doctests on
Python 3.12–3.14. The separate GPL provider now executes fixed-bank SBC/design and calibration
reports; these numerical-reference tests do not qualify provider execution.
