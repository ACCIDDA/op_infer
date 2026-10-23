# Random streams

`op_infer.rng` provides explicit task addresses around native random generators.
Estimator adapters, the implemented bounded catalog of five independent scalar or
fixed-shaped IID prior families, the seven IID observation-law families and the
[SBC runner](sbc-runner.md) consume these addresses. Transformed, hierarchical and
dependent prior or observation structures remain planned. Stable task identities
preserve the separation of parameter and simulation streams.

## Default parameter and simulation separation

```python
from op_infer.rng import SimulationStreams

streams = SimulationStreams(seed=2026, index=8)
parameter_rng = streams.parameters.numpy_generator()
simulation_rng = streams.simulation.numpy_generator()

parameter = parameter_rng.normal()
measurement = parameter + simulation_rng.normal()
saved_address = streams.model_dump_json()
replayed = SimulationStreams.model_validate_json(saved_address)
assert replayed == streams
```

An address consists of schema version 1, an explicit unsigned 128-bit root seed, a
case-sensitive named stream and an unsigned 64-bit task index. Names are ASCII:
start with a letter, followed by letters, digits, `_`, `.`, `:`, `/` or `-`. Boolean,
fractional, negative and overflowing integers, unknown fields and unknown versions
are rejected. Metadata is immutable; native generators are mutable and caller-owned.

`SimulationStreams` supplies the distinct names `parameters` and `simulation` for one
task. Consuming more parameter draws cannot advance the simulation generator. A retry
needs a new recorded index or namespace. The API never silently retries, advances a
counter, changes an acceptance rule or drops a failed task.

## Address tasks rather than workers

```python
from op_infer.rng import RandomKey

key = RandomKey(seed=2026, stream="calibration/simulation", index=19)
draws = key.numpy_generator().normal(size=10)
# The same task can run later, on another worker or in a differently sized batch.
replay = RandomKey.model_validate_json(key.model_dump_json())
assert (draws == replay.numpy_generator().normal(size=10)).all()
```

Use task/attempt identities that remain stable when execution is reordered, extended
or resumed. Different experiment stages need different recorded namespaces or root
seeds: the same `(seed, stream, index)` deliberately means replay. Generate task
addresses directly; no batch size or worker number enters their derivation. This
contract does not depend on whether a particular JAX version's `split` output changes
when its requested count changes.

Lowering an address again restarts its native stream. Retain a generator to continue
consuming it; record native generator state separately if resuming inside a task.
Recreating a generator at each draw would replay the same value. Common random numbers
can deliberately reuse an address across designs, but those design results are then
dependent; this does not create independent replicates.

## Optional native adapters

Install `op_infer[jax]` or `op_infer[torch]` for these adapters. Importing the core or
the adapter module does not import either backend. Missing extras fail when requested.

```python
from op_infer.adapters.rng import jax_key, torch_generators

# Host address -> native typed key. Pass this key into a JIT/vmap kernel.
native_key = jax_key(streams.simulation)

# Validate the COMPLETE independent task bank before dispatching/chunking it.
parameter_generator, simulation_generator = torch_generators(
    (streams.parameters, streams.simulation)
)
```

JAX uses explicitly typed `threefry2x32` keys, independent of its global PRNG default.
Host address parsing/SeedSequence construction happens before tracing. Sampling then
uses ordinary native JAX operations; dynamically traced task counters are not accepted
by the host metadata API. The CPU tests qualify native JIT/vmap key inputs and replay.

Torch lowering defaults to **CPU generators**, preserving existing streams. Pass
`device="cuda:0"` for device-local CUDA generators. Observation sampling also
accepts a device-matched native generator; a mismatched device fails explicitly.
The [CUDA qualification](cuda-qualification.md) covers actual A100 execution.
Its native CPU MT19937
initialization has 32 effective seed bits: a qualification reproduction with Torch
2.14.1 shows `manual_seed(123)` and `manual_seed(123 + 2**32)` generate identical draws.
The adapter therefore uses an explicitly 32-bit seed and checks its complete supplied
bank for effective collisions, including repeated addresses, before creating any
generator. This is a property of native seed lowering, not evidence that the host
addresses or full SeedSequence states are identical.

The regression fixture contains a real collision: root seed `2026`,
`simulation/21297` and `parameters/68078` both lower to CPU seed `1998271142`.
`torch_generators` refuses the pair. A user must choose and record a new root seed or
namespace for the complete bank before evaluating it; the library does not silently
remap a collided task. Separate calls cannot detect collisions with previous banks.
For a study that needs all tasks independent, validate them together before splitting
work across workers. Version 1 conservatively retains the same 32-bit effective
seed mapping and collision check on CUDA; it does not use CUDA's wider seed space
to change existing task addresses. Large/distributed bank management remains
separate. A deterministic finite test is not a statistical independence
certificate, and finite backend key spaces cannot offer unlimited distinct streams.

## Version 1 derivation and reproducibility

1. Hash the ASCII stream bytes with SHA-256. Interpret its 32 bytes as eight
   little-endian uint32 words.
2. Create NumPy `SeedSequence(seed, pool_size=4)` with explicit spawn path
   `(0x4F504931, *stream_words, index_low32, index_high32)`.
3. NumPy uses the named `PCG64DXSM` bit generator. JAX wraps the first two generated
   uint32 state words as a typed Threefry key. Torch uses the first uint32 word,
   subject to the complete-bank collision check.

Both halves of the index and all root-seed bits participate before backend key-width
reduction. The function uses no Python randomized hash, mutable spawning history,
process-global RNG or arithmetic `seed + worker` convention. Tests pin the framing,
state words and raw NumPy generator output against independently evaluated vectors.

Record the address alongside the model/prior/design/stage identities in the existing
run/provenance artifacts. Also record backend and NumPy versions, lowering algorithm,
device, distribution calls/shapes/dtypes and task/retry allocation. An address is only
the randomness part of an experiment identity, not a replacement provenance schema.
Native sampling algorithms, transforms and execution may change across versions or
devices. No bitwise equivalence across NumPy/JAX/Torch or platform/version guarantee is
claimed. These tools are not cryptographic RNGs or sampler-adequacy checks.

The algorithm choice follows [NumPy's parallel RNG guidance](https://numpy.org/doc/stable/reference/random/parallel.html)
and [SeedSequence API](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html),
[JAX's typed-key guidance](https://docs.jax.dev/en/latest/jax.random.html) and
[key-data conversion API](https://docs.jax.dev/en/latest/_autosummary/jax.random.wrap_key_data.html),
and the local-generator interface in [Torch's Generator API](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html).

## Qualification

`just ci` exercises host validation, serialization, independent derivation vectors,
NumPy replay, untouched global state and fail-closed Torch banks. After installing
both backend extras, `just qualify-rng` requires real JAX and Torch imports and runs
the CPU replay/JIT/vmap/gradient checks. The numerical CPU CI job runs that recipe;
missing backends cannot turn that qualification into a green skipped run.
