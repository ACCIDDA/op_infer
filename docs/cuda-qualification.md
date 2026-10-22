# Scoped CUDA qualification

Actual NVIDIA A100-SXM4-80GB execution qualifies the statistical paths below.
CPU and device checks have separate requirements. CuPy is an array backend;
CUDA is a device/runtime target for several backends.

| Path | Qualified arrays and scope |
| --- | --- |
| NumPy statistical interfaces | CPU; required current and minimum dependency checks. |
| JAX statistical interfaces | CPU plus A100 CUDA float32/64: five prior densities, seven iid observation densities/sampling, native dynamic RNG keys and masked/censored Normal gradients. |
| Torch statistical interfaces | CPU plus A100 CUDA float32/64 for the same families; device-local addressed observation sampling and explicit native generators. |
| CuPy statistical interfaces | Explicitly unsupported; no silent NumPy fallback. |
| Compiler/engine scalar reference | Separately executed A100 JAX, Torch and CuPy float32/64 RHS/derivative values and fixed RK4 trajectories with final-only and stored output. Native AD was checked for JAX/Torch. This does not qualify every operator/solver combination. |
| Native NPE/NUTS fitting and application workflows | Their existing qualification boundaries apply. These statistical device checks do not establish training performance or application calibration. |

The statistical run used JAX 0.12.0 and Torch 2.14.1, with NumPy 2.5.4 and
SciPy 1.18.1. The separate compiler/engine run also used CuPy 14.2.0. The A100
driver was 595.71.05. Older CUDA versions/devices, multi-GPU execution, MPS,
mixed precision and distributed estimator fitting need their own checks.
Minimum CPU dependencies remain covered independently, including Torch 2.6.

## Local generators and native data

```python
from op_infer.adapters.rng import torch_generators
from op_infer.rng import RandomKey

keys = (
    RandomKey(seed=2026, stream="parameters"),
    RandomKey(seed=2026, stream="simulation"),
)
# Requires a CUDA Torch installation and an available CUDA device.
generators = torch_generators(keys, device="cuda:0")
```

`IIDObservation.sample(cuda_latent, key)` lowers a `RandomKey` to a local generator
on that latent array's device. Supplying a native generator requires the same
device. CPU streams retain their existing derivation and bitwise fixtures;
cross-device/version bitwise equality is not promised. Validate the complete
simultaneously independent task bank for effective seed collisions before
dispatching it. Addressed draws do not seed or consume a global generator.
An explicitly supplied native generator is consumed as supplied.

Latent arrays and returned samples remain native; CUDA observation sampling
accepts float32/64. Prior declarations retain their native NumPyro/Torch sites,
including explicit Torch site devices. General prior transforms, correlated
observation models and estimator device settings remain separate interfaces.

## Reproduce the statistical checks

Use separate JAX CUDA and Torch CUDA environments. The tested JAX/CuPy lane used
CUDA 12 packages, while the tested Torch wheel supplied its own CUDA library
set. Combining both library sets in one environment can obscure runtime errors.
Install the reviewed source checkout and the selected backend's CUDA runtime;
the core package adds no CUDA dependency automatically.

```shell
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  python examples/cuda_statistics.py --backend jax > jax-cuda.json
python examples/cuda_statistics.py --backend torch > torch-cuda.json
```

The program fails if actual GPU execution is unavailable. It checks 24 family/
precision cases per backend, 8,192 samples per observation case, independent
analytic moments, SciPy density references, device/dtype retention and replay.
A separate Normal check compares missing/censored values and gradients with
independent tail formulas. Torch additionally checks CPU/global CUDA RNG states
and rejects a CPU generator for CUDA sampling. RNG keys and thresholds are fixed
in the executable fixture before the run.

Reviewed receipts retain versions, source/fixture hashes, moments and process
peak RSS: [JAX](assets/cuda-2026-10-21/statistics-jax.json) and
[Torch](assets/cuda-2026-10-21/statistics-torch.json). Torch also records device
peak allocated/reserved bytes. Process RSS includes imports and runtime context;
it is distinct from device allocator memory. These are correctness checks on
bounded synthetic arrays, with no timing comparison or production memory claim.

The companion core receipts are [JAX](assets/cuda-2026-10-21/core-jax.json),
[Torch](assets/cuda-2026-10-21/core-torch.json) and
[CuPy](assets/cuda-2026-10-21/core-cupy.json). An [earlier Torch core attempt](assets/cuda-2026-10-21/earlier-torch-failure.json) failed
while stacking CPU constants with CUDA outputs. The corrected compiler path is
[op_system #335](https://github.com/ACCIDDA/op_system/pull/335), and graph/device
state adoption is [op_engine #265](https://github.com/ACCIDDA/op_engine/pull/265).
The successful rerun retains the same numerical fixture. Core receipt sampling
entries describe the earlier primitive check; the separate statistical receipts
qualify the public sampling interface.


## Deferred workstation rerun

Further CuPy, JAX and Torch GPU execution is on hold until the maintainer resumes
it on the GPU workstation. The earlier A100 receipts above remain historical
and do not qualify another device or a changed source build.

The [workstation handoff in op_system PR #339](https://github.com/ACCIDDA/op_system/pull/339)
includes `docs/guides/gpu-workstation.md` and a bounded core fixture. It records
exact engine/inference revisions, the required candidate compiler head, isolated
current/minimum CuPy and separate JAX/Torch CUDA environments, expected counts,
commands and metadata to retain. CuPy requires ten zero-skip compiler tests in
each dependency lane; JAX/Torch require four core cases and the existing 24-case
statistical fixture plus its masked/censored gradient check. The separate Torch
compiled-loop adapter remains a CPU qualification. No estimator training,
consumer adoption or package publication is part of this handoff.
