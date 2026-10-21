# Native masked-cohort NPE

Install `op_infer[sbi]` and use `op_infer.adapters.sbi`. This bounded adapter wraps
sbi's single-round NPE/MAF training on CPU float32. It does not implement flows,
calibration gates, automatic gate-driven retraining, multi-round proposals or GPU
training.

`NpeSettings` records the complete scalar parameter names, feature dimension,
maximum design unit count, architecture, training controls and `RandomKey`.
`fit_npe(theta, observations, prior=..., settings=..., provenance=...)` consumes
aligned `(simulation, parameter)` and `(simulation, design_unit, feature)` native
Torch tensors. Record the same prior, observation support and acceptance rule
used to create the simulation bank; the adapter applies no plausibility filter.
Exact installed versions come from `sbi_backend_versions()` and the canonical
settings reference from `npe_settings_reference(settings)` must match the supplied
complete `Provenance`.

An absent unit has an all-NaN feature row. Partially missing rows and infinities
are errors; encode within-unit missing features explicitly in the observation
design rather than confusing them with absent units. Each cohort computes its own
count, pools the present unit embeddings by their mean and appends count divided
by the declared design size. With `mask_probability > 0`, training mode removes
one present unit per selected cohort, retaining at least one when possible.
Evaluation and sampling do not apply removal. The embedding handles an empty
cohort numerically, but that does not establish that its posterior equals the
prior; calibration must cover any count/design regime used for reporting.

Training explicitly sets `exclude_invalid_x=False` and `z_score_x="none"`.
Missing units pass to the custom embedding, and a runtime count check refuses any
unexpected exclusion of simulation rows. Native sbi may log its generic NaN
warning; the all-NaN absence convention is intentional here. `NpeFit` retains the
native density estimator, settings, provenance, simulation count and native epoch
counter. A small fit reaching its epoch limit is not calibration evidence.

`NpeEnsemble((fit_a, fit_b, ...)).sample(observation, per_seed=..., key=...)`
draws the same count from every seed and returns native named `Posterior` values
with separate seed/draw coordinates. Seeds are not MCMC chains. Include latent
state components in `parameter_names` and training targets when a subsequent
forecast must start from an inferred end-of-window state. Members must agree on
the full model/prior/design/code/target and controls apart from their training
keys; effective Torch CPU seed collisions are refused. Reporting records include
member weights/settings, conditioning content and the sampling address.

At the qualified sbi 0.27.0 boundary, sampling calls the native density estimator
directly because `DirectPosterior`
rejects the NaN-conditioned cohort. This also bypasses its bounded-prior leakage
correction. The adapter therefore requires a supplied prior with full real-vector
support. Transform bounded parameters first, for example with declared logit
coordinates, and record the transformation in prior/parameter provenance. No
bounded-support correction is silently approximated.

Native Torch training/sampling use a scoped CPU RNG and float32 default dtype,
restoring the caller's RNG and dtype afterwards. Those native global contexts
require serialized calls within a process; use separate worker processes for
concurrent fitting. The data model does not eagerly import Torch/sbi. Seed
ensembles can be retrained and independently re-gated after an initial gate
failure; a changed ensemble needs new provenance and a new calibration record.

`save_npe` atomically writes native state tensors and primitive metadata;
`load_npe(path, expected=independently_supplied_provenance)` uses weights-only
loading and reconstructs the declared architecture. Schema, exact runtime,
settings, scientific provenance and tensor-state digests must agree. This is an
integrity/compatibility check, not authenticated publication or cross-version
migration. Native estimator Python objects are never serialized.

Required optional CI runs real minimum/current CPU fits, analytically specified
mean/count embeddings and gradients, training/evaluation masking, fixed seed
contributions, RNG restoration, provenance refusal and exact snapshot replay.
Scientific use additionally needs powered calibration on the intended design,
including missing-unit strata, before using its design/recovery estimates.

## Large banks and training checkpoints

With sbi 0.27.0, `fit_npe(..., borrow_bank=True)` keeps the original observation
storage instead of materializing sbi's filtering, concatenation and initialization
copies. Inputs must be contiguous CPU float32 Torch tensors without gradients.
For a NumPy `.npy` bank, use `torch.from_numpy(np.load(path, mmap_mode="c"))`.
The copy-on-write mapping gives Torch a writable view without permitting writes
to change the file. Keep both tensors and their backing files immutable while
fitting. Validation and checkpoint bank hashing read the entire bank in bounded
chunks; a memory map does not make those reads free.

Native sbi still owns the loss, split, batch sampling, optimizer, standardization
and early stopping. Parameter standardization materializes the training **theta**
subset, and index/sampler storage grows with the number of simulations. Batch
activations, model/optimizer state and file-backed resident pages also use memory.
This option removes full **observation** copies; it does not promise constant
resident memory, GPU support, arbitrary iterators, multi-round inference or a
one-copy bound for large parameter matrices. Inputs requiring dtype/device/layout
conversion are refused rather than silently copied in this mode.

One separate-process synthetic run on Linux, Python 3.13, sbi 0.27.0 and
Torch 2.14.1 CPU used an 8,192-row, 256-unit, eight-feature bank (64.06 MiB),
one thread and four completed epochs:

| Observation storage | Peak process RSS (MiB) | Fit elapsed time (seconds) |
| --- | ---: | ---: |
| Native sbi copies | 710.03 | 12.41 |
| Borrowed memory map | 538.38 | 11.59 |

The final tensor-state digests agreed exactly. These are individual descriptive
measurements, including imports and mapped resident pages, rather than a scaling
law or a production-bank qualification. The full records retain source/runtime
identities: [native](assets/npe-training/result-native.json) and
[borrowed](assets/npe-training/result-borrowed.json). To repeat in fresh processes:

```bash
python examples/npe_training.py /tmp/npe-memory --rows 8192 --units 256 --features 8 --prepare
python examples/npe_training.py /tmp/npe-memory --rows 8192 --units 256 --features 8 --mode native
python examples/npe_training.py /tmp/npe-memory --rows 8192 --units 256 --features 8 --mode borrowed
```

To train in chunks, hold `settings` and its provenance fixed for the entire run:

```python
fit = fit_npe(
    theta,
    observations,
    prior=prior,
    settings=settings,
    provenance=provenance,
    borrow_bank=True,
    checkpoint_path="training.pt",
    until_epoch=60,
)
# A later process uses the same ordered bank and independently supplied contract.
fit = fit_npe(
    theta,
    observations,
    prior=prior,
    settings=settings,
    provenance=provenance,
    borrow_bank=True,
    resume_from="training.pt",
    checkpoint_path="training.pt",
    until_epoch=120,
)
```

Here the full budget must allow at least 120 completed epochs. `until_epoch` is
cumulative and must be an integer from 2 through `settings.max_epochs + 1`.
The upper limit preserves the existing adapter's native sbi convention:
`max_epochs` is passed as an inclusive, zero-based native epoch limit. Omitting
`until_epoch` uses that full budget. Early stopping can finish sooner; a chunk
does not reset its patience or authorize additional training past the full budget.

Checkpoints contain primitive metadata and state tensors for the last completed
optimizer step, Adam moments, best validation weights/loss, patience counter,
train/validation indices, loss history and Torch CPU RNG. They are captured before
native sbi may restore best validation weights at an epoch limit. Saving only the
returned estimator and optimizer after that restoration would pair different
training states. Checkpoints are separate from `save_npe` inference snapshots.

The default saves after every completed epoch. `checkpoint_interval=k` saves
the first epoch, every kth epoch and the requested chunk limit. Early stopping
or preemption can leave an older complete boundary; an incomplete epoch never
replaces it. Saving uses an atomic file replacement. A failed write preserves
the preceding checkpoint, though durable storage across machine/power failures
still depends on the filesystem.

Resume refuses changed scientific provenance, settings, exact runtime versions,
ordered effective float32 bank content or checkpoint integrity. Paths and file
names alone are not bank identities. No simulation bank, prior object, trainer
or estimator Python object is pickled. This is a compatibility/integrity boundary,
not authentication of an untrusted publisher. The private native training hooks
are qualified only for sbi 0.27.0; other versions are refused for these options.

Run the complete small example in separate processes:

```bash
python examples/npe_training.py /tmp/npe-demo --until-epoch 2
python examples/npe_training.py /tmp/npe-demo --resume
```

It writes inference snapshots and JSON records with source/runtime provenance,
completed epochs, elapsed time and peak process RSS. Required native CI compares
the result with an uninterrupted process at current and minimum Torch versions.
Tests also compare native and borrowed storage, optimizer moments, splits,
losses and RNG, including interruption and native early stopping. These checks
establish execution replay on the stated boundary, not scientific calibration.
