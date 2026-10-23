# Fixed-work NPE performance pilot

This pilot measures the implemented `fit_npe` boundary under fixed synthetic
work. It separates engineering phases and retains failures; it does not establish
posterior calibration, an application setting, a general CPU/GPU speedup or the
cause of variation between compute nodes.

The executable harness is
[`examples/benchmark_npe_pipeline.py`](https://github.com/ACCIDDA/op_infer/blob/main/examples/benchmark_npe_pipeline.py).
Every condition runs in a fresh process. A skipped or unavailable CUDA worker is
a failure, not qualification.

## Declared local matrix

The default bank has 8,192 rows, two parameters and observations with 256 units
and eight features. Its two `.npy` files contain 64.06 MiB including headers.
The data, missing-unit pattern and file digests are fixed before timing. All
full-budget fits use one host-defined train/validation split (NumPy generator
seed 51,817), the same training key, batch size 128, one transform, four
completed native epochs and one requested CPU thread. The matrix requires the split to agree across
every device, stage and storage path and across all repetitions. It also
requires the final CPU state to agree across ordinary training, full checkpoint
training and resume, in addition to detailed replay-state equality.
At the pinned sbi 0.27.0 boundary, the harness installs these indices before
native loader construction; sbi still owns batching, optimization, validation
and estimator construction.

The serial matrix crosses `cpu` and `cuda:0` with three CPU-bank representations:

- `ordinary` loads resident arrays and permits native sbi copies;
- `borrowed` loads resident contiguous tensors and enables `borrow_bank=True`;
- `mapped` uses copy-on-write `.npy` maps and enables `borrow_bank=True`.

For every CPU representation, a separate uninterrupted four-epoch checkpoint
run is compared with a two-epoch run resumed to four epochs. Final network,
optimizer, split, loss history and Torch RNG states must agree exactly. Measured
epoch durations remain descriptive and are not part of that replay equality.
CUDA training checkpoints remain outside the qualified adapter boundary; CUDA
workers instead exercise guarded inference snapshot save/load and same-device
addressed sampling replay.

Each worker separately records bank loading, bank validation, training and
validation epochs, checkpoint writes and restores, inference snapshot save/load,
one 4,096-draw posterior sample and a small calibration-pipeline timing smoke.
The last phase uses 32 synthetic cases and 128 draws per case. It is deliberately
underpowered and is not calibration evidence.

## Measurement record

Elapsed CUDA phases synchronize the selected device before and after the timed
operation. A 10 ms telemetry thread runs on a separately declared CPU; setup,
the initial scan, teardown and the final scan fall outside the elapsed interval.
Epoch records include executed batch counts, CPU/CUDA RNG-state identities and
retain the first epoch rather than silently treating it as warm-up. The native
train/validation split is also retained. Phase records include:

- sampled RSS, PSS, USS, anonymous/file-backed components and separate artifact
  and checkpoint-directory initial, final and high-water marks;
- minor/major page faults and process block-I/O counters;
- Torch allocated/reserved CUDA peaks;
- the worker's actual cgroup path, memory/CPU pressure and memory counters;
- CPU affinity, visible NUMA nodes, filesystem/mount, thread controls and exact
  Python, NumPy, sbi, Torch, CUDA runtime, device and compute capability;
- start/end `nvidia-smi` temperature, utilization, memory, clock and power-state
  observations; and
- exact bank and source-byte identities plus post-run bank re-verification.

The sampler still perturbs memory, caches and kernel work even on a distinct CPU,
and can miss a shorter temporary peak. Its elapsed values are therefore
instrumented observations, not unobserved operation times or grounds for a
causal CPU/GPU or storage speedup claim. Torch allocator peaks and sampled
process memory have different scopes. Cgroup values may describe unrelated
processes when the recorded cgroup is shared. Start/end GPU observations do not
reconstruct conditions throughout a phase. These fields must remain separate
rather than being combined into one memory, contention or performance number.

## Running the pilot

Install the `sbi` extra with sbi 0.27.0 and a CUDA-capable Torch build. Prepare
the immutable bank outside the timed workers:

```bash
python examples/benchmark_npe_pipeline.py /tmp/op-infer-npe-pilot --prepare
```

Preflight CUDA explicitly, then run five serial repetitions in a fresh output
directory. Select and record distinct compute and telemetry CPUs for the host;
this example uses CPUs 7 and 8 only as placeholders, not portable recommendations:

```bash
python -c "import sbi, torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python examples/benchmark_npe_pipeline.py /tmp/op-infer-npe-pilot \
  --matrix --devices cpu,cuda:0 --repetitions 5 --threads 1 --cpus 7 \
  --monitor-cpu 8
```

The directory receives one JSON record per worker, an append-only, synchronously
flushed JSONL event journal and a compact `summary.json`. The matrix refuses to
reuse existing execution output. Started, completed and failed worker events
retain captured output, while the summary is written only after epoch counts,
one source-byte identity, the global split identity, within-device storage-path
identities and CPU replay equality pass. Preserve the journal even when a worker
fails; a failed or interrupted matrix is evidence about that attempt, not a
successful comparison.

## Interpretation and remaining work

Fixed-seed repetitions describe execution variation, including runtime
nondeterminism, not uncertainty in an estimated scientific quantity. CPU and
CUDA replay are assessed within their own devices; their losses, weights and
posterior draws are not required to agree bit-for-bit. Bank generation and
numerical-solver work are outside these timed training workers.

This workstation has one NUMA node and local storage. Consequently, its pilot
cannot close [issue #43](https://github.com/ACCIDDA/op_infer/issues/43): a second
controlled compute placement and relevant shared-storage placement are still
needed. Early-stop variation must also be studied separately from this fixed-work
matrix. Any calibration claim needs a prespecified reading policy and the
project's powered replicate floor rather than the timing smoke used here.

The first retained workstation result is reported separately in the
[23 October 2026 pilot record](npe-performance-results-2026-10-23.md).
