# Fixed-work NPE pilot result: 23 October 2026

This result is a descriptive engineering pilot of the implemented `fit_npe`
boundary on one workstation. All 75 fresh-process workers completed. The final
gate verified one source-byte identity and one train/validation split across all
workers, the declared epoch count, exact within-device storage-path RNG/state
identity, exact full-budget CPU train/checkpoint/resume state and exact CPU
checkpoint continuation state. No worker failed or skipped.

The immutable bank contained 8,192 parameter rows and float32 observations with
256 units and eight features (64.06 MiB including the two NumPy headers). Each
full-budget training condition had five fixed-seed repetitions and four native
epochs. CPU 7 was the compute affinity and CPU 8 ran the 10 ms telemetry sampler.

## Retained identities

| Item | Retained value |
| --- | --- |
| Raw journal SHA-256 | `28a4ec0cc12ed9804b9661dc016950fcbea5ea080eee6f598c1daf121edf5106` |
| Source-manifest SHA-256 | `9d75ff148f2bf5c2598cab649f6cbb3c273158158ae40c4b17bb54186728a8f0` |
| Harness SHA-256 | `4b39137576faef56658ed12a21247ab6705ca6297bb7f64a48df4663752af504` |
| Observation-bank SHA-256 | `35333c7f6e65b98a304f384afd6ad74fd5e1afdcea91d0e4b1bb1bcdf7cc4b82` |
| Parameter-bank SHA-256 | `dad29f7a63d111bf090c8ed5463cf7e147cfb03ee4a535d0258b6d5a7ee1298c` |

The public [matrix summary](assets/npe-fixed-work-2026-10-23/summary.json)
retains the complete validation statements and unrounded condition-level fit
summary. The [sanitized condition summary](assets/npe-fixed-work-2026-10-23/condition-summary.json)
retains the environment and phase/memory aggregates without local paths,
process identifiers, mount sources or cgroup paths. The raw journal is not
published because its detailed local fields are useful locally but unsuitable
for a public repository. The retained filesystem type and shared-scope
classification are deliberately coarse; the sanitized file excludes mount
sources, mountpoints, cgroup paths, cgroup counters, local paths and process
identifiers.

## Environment

| Component | Recorded value |
| --- | --- |
| Host | Linux 7.1.1, one visible NUMA node, local ext4 storage |
| Python / NumPy | 3.12.15 / 2.5.3 |
| sbi / Torch | 0.27.0 / 2.14.1 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0 |
| CUDA runtime / NVIDIA driver | 13.0 / 580.173.02 |
| Threads and placement | one Torch/OMP/BLAS thread; compute CPU 7; monitor CPU 8 |
| Cgroup interpretation | shared scope; counters may include unrelated processes |

## Instrumented elapsed time

Values are median [minimum, maximum] seconds across five workers. “Cold” is the
first training epoch. “Warm” pools the remaining three epochs per worker. The
10 ms process/disk sampler ran concurrently on CPU 8, so these are instrumented
observations rather than unobserved operation times.

| Device | Bank path | Complete fit | Cold epoch | Warm epoch |
| --- | --- | ---: | ---: | ---: |
| CPU | ordinary | 2.375 [2.311, 2.400] | 0.469 [0.405, 0.528] | 0.177 [0.174, 0.182] |
| CPU | borrowed | 1.787 [1.686, 1.828] | 0.498 [0.463, 0.575] | 0.178 [0.172, 0.181] |
| CPU | mapped | 1.756 [1.666, 1.786] | 0.497 [0.430, 0.525] | 0.177 [0.173, 0.182] |
| CUDA | ordinary | 2.625 [2.599, 2.722] | 0.656 [0.648, 0.667] | 0.143 [0.128, 0.149] |
| CUDA | borrowed | 2.035 [1.984, 2.136] | 0.650 [0.646, 0.680] | 0.137 [0.130, 0.147] |
| CUDA | mapped | 2.050 [2.008, 2.109] | 0.653 [0.633, 0.677] | 0.137 [0.132, 0.143] |

The CUDA warm-epoch observations were lower than the corresponding CPU values,
but its cold-epoch and complete-fit observations were larger. Consequently,
this pilot does not support a general CPU/GPU speedup claim. On this host,
borrowed and mapped banks had lower observed complete-fit elapsed times than
ordinary storage within each device. That descriptive observation is local to
this run and is not a causal storage result or portable performance guarantee.

## Memory and checkpoint observations

Values below are median peak MiB across the five complete-fit workers. Sampled
process memory and Torch allocator memory have different scopes and must not be
added or substituted for one another.

| Device | Bank path | RSS | PSS | USS | CUDA allocated | CUDA reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | ordinary | 1110.4 | 1097.0 | 1087.6 | — | — |
| CPU | borrowed | 981.4 | 968.0 | 958.6 | — | — |
| CPU | mapped | 981.4 | 968.0 | 958.5 | — | — |
| CUDA | ordinary | 2167.9 | 2154.5 | 2145.1 | 68.3 | 70.0 |
| CUDA | borrowed | 2030.5 | 2017.1 | 2007.6 | 68.3 | 70.0 |
| CUDA | mapped | 2030.6 | 2017.2 | 2007.8 | 68.3 | 70.0 |

All 15 CPU resume workers matched their uninterrupted controls. The median
checkpoint restore observation was 0.632 seconds [0.601, 0.662] for a 116,557-byte
checkpoint. Across 120 atomic checkpoint writes, the median write observation
was 0.00155 seconds [0.00144, 0.00208]. CUDA training checkpoints remained
outside the qualified adapter boundary; CUDA workers used guarded inference
snapshot save/load and exact addressed sampling replay instead.

## Scope

This result does not establish posterior calibration, application adequacy,
production throughput, energy efficiency or a causal speedup. The fixed-seed
repetitions measure execution variation, including runtime nondeterminism, not
uncertainty in an estimated scientific quantity. The host has one NUMA node,
local storage and a shared desktop cgroup. A second controlled compute placement,
relevant shared-storage placement and an early-stopping study remain necessary
before issue #43 can be closed.
