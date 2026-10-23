# Fixed-forward pipeline performance

The compiled fixed forward now groups consecutive observation intervals with the
same internal-step count into a shared outer scan and inner step scan. Previously
each observation interval traced a separate differentiated driver. The grouped
form reduces repeated JAX compilation while retaining the existing step kernels,
fixed mesh, requested outputs and parameter coordinates.

One bootstrap step establishes DOPRI5's FSAL carry structure. Ragged interval
lengths start another group. Both eager and scanned execution evaluate the same
public engine observation callbacks and check output shape, namespace and dtype.
Time-dependent/vector outputs and initial-state gradients remain supported.
Scans return requested observations, retaining current state/FSAL rather than a
stacked primal state history. Both implementations already avoided that full
history; this change primarily reduces the repeated compiled graph.

## Bounded CPU comparison

The [baseline](assets/pipeline-2026-10-21/baseline.json),
[grouped result](assets/pipeline-2026-10-21/grouped.json),
[derived summary](assets/pipeline-2026-10-21/summary.json) and
[environment](assets/pipeline-2026-10-21/environment.json) retain 20 cases with
three serial fresh-process repetitions each. The small case has four scalar
states, 16 RK4 steps, five output times and batch size one. The larger case has
32 flat scalar states, 64 steps, 17 output times and batch size eight. Both use
float64, a declared Normal rate prior and shared Normal measurement law.
Simulation, fixed-data likelihood and gradients run in separate workers.

All completed workers pass independent analytic decay/likelihood/gradient checks.
Source fingerprints stay constant within each revision. All 54 completed matched
baseline cases retain the same data, mesh, model structure and dependency versions
as their grouped counterparts. Baseline source matches public main at
`ca3e0f34a9cfaabca36ea12dd4a0d1a2e520b219`. Candidate source content hashes are
retained explicitly, including its then-uncommitted adapter change.

| Larger JAX case | Baseline first call | Grouped first call | Baseline/grouped warm medians | Baseline/grouped process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Simulation | 19.65 s | 3.36 s | 1.52 / 0.15 ms | 887 / 489 MiB |
| Fixed-data likelihood | 19.44 s | 3.25 s | 1.77 / 0.28 ms | 883 / 472 MiB |
| Gradient | 3 workers censored | 17.32 s | unavailable / 4.86 ms | unavailable / 663 MiB |
| Checkpointed gradient | 3 workers censored | 12.16 s | unavailable / 3.30 ms | unavailable / 642 MiB |

First-call time includes selected-function JIT and execution; backend/model setup
has separate records. Censoring uses a **120-second whole-worker** bound, which
also includes setup, warm calls and receipt collection. It is not a measured
baseline first-call time or a basis for an exact gradient speedup ratio. The
baseline has **54 successes and six timeouts**; the grouped matrix has **60
successes, no errors and no timeouts**. Small cases and all native CPU results
remain visible in the complete summary.

These measurements use one logical CPU on a shared Intel Core Ultra 7 265U host,
with recorded one-thread native pools. Three repetitions per revision were
collected in separate blocks, with alternating case order. They are descriptive
results, not a controlled-runner timing gate or a portable performance guarantee.
NumPy's gradient probe uses two finite-difference evaluations; JAX/Torch use
reverse mode. Backend RNG bookkeeping and compilation differ, so the table does
not establish a universal backend ranking.

## Interpret memory separately

Process RSS/PSS includes libraries, compilation and allocator state. For the
larger simulation/likelihood cases its median peak falls about 45–47%, while XLA's
estimated temporary buffers slightly increase (13,576 to 15,240 bytes for
simulation; 14,384 to 16,056 for likelihood). These describe different costs.
The compiler-graph reduction is not evidence that previously retained state
history was removed, or that training-bank memory falls by the same amount.

Within the grouped larger gradient case, checkpointing changes the XLA temporary
estimate from 776,968 to 356,424 bytes, about 54% lower. Whole-worker RSS changes
from about 663 to 642 MiB. Reverse-mode residuals and selected observations still
allocate memory; checkpointing is an option with a scoped measurement.

Each worker records live RSS/PSS/USS, anonymous/file-backed RSS components,
visible-cgroup pressure/events, result sizes, affinity and loaded thread pools.
Cgroup metrics may include unrelated processes and are not model-only peaks.
A theoretical full-history size is labeled as an illustration, not an observed
allocation saving.

## Reproduce

Install the optional numerical stack and released OP dependency floors described
in [forward models](forward-models.md), plus the benchmark reporter:

```shell
uv sync --dev --extra op --extra numpyro --extra benchmark
uv pip install 'op-system==0.10.0' 'op-engine==0.9.0'
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
uv run --no-sync python examples/benchmark_statistical_pipeline.py \
  --output /tmp/pipeline-comparison.json --repetitions 3 --warm-repeats 5 --timeout 120
```

The Linux protocol runs fresh workers serially, disables persistent JAX caches,
sets explicit thread limits and preserves failures/timeouts. JAX simulation keys
are dynamic compiled inputs, preventing random draws from being precomputed as
constants. An exclusive worker journal retains completed records if the parent
is interrupted. Existing receipt paths are refused.

The receipt covers flat synthetic fixed-forward workloads. Structured models,
CUDA/CuPy, device transfers, consumer banks/networks and offline NPE epochs remain
separate qualification. This change introduces no numerical method, sampler,
statistical default or PyPI release, and does not migrate frozen applications.
