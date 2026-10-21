# Native NumPyro NUTS

The optional `op_infer.adapters.numpyro` module fits native NumPyro models
with NUTS/MCMC and returns a shared `Posterior`.
It wraps NumPyro; it contains no sampler implementation.

```bash
uv sync --dev --extra numpyro
uv run python examples/numpyro_normal.py
```

The [complete example](https://github.com/ACCIDDA/op_infer/blob/main/examples/numpyro_normal.py)
fits a standard Normal location prior to observations `[0.5, 1.5, 1, 2]` with
independent unit-variance Normal measurement noise. Its analytic posterior has mean
`1` and variance `1/5`. The example emits sample moments, raw diagnostics, counts and
provenance identity. It is a deterministic smoke qualification, not powered SBC or
proof of sampler adequacy. It uses no biological model, provider or solver dependency.
The example includes source hashes for its model and the op_infer modules it uses;
dependency versions remain separate.

## Declare the run before fitting

```python
from op_infer.adapters.numpyro import NutsSettings, SiteLayout
from op_infer.rng import RandomKey

settings = NutsSettings(
    key=RandomKey(seed=2026, stream="reference_fit", index=0),
    sites=(SiteLayout(name="theta"),),
    num_warmup=100,
    num_samples=512,
    num_chains=2,
)
settings_payload = settings.manifest_json()  # Exact bytes: UTF-8 encode before storing.
settings_reference = settings.reference
```

`NutsSettings` is immutable and versioned. It validates integer budgets, finite
acceptance controls, RNG identity and complete named event layouts. The resolved
settings reference includes output labels/order, sequential chains, thinning one,
disabled progress output, initialization `init_to_uniform(radius=2)` and requested
sample diagnostics. Remaining native defaults are tied to the exact NumPyro version.
The default numerical backend must be CPU, with JAX's `enable_x64` flag matching the
recorded setting. The adapter checks globals and does not alter them. This flag
records precision availability; it does not assert that every model array is float64.
Replay is qualified within a fixed CPU/software configuration; cross-device/compiler
bitwise equivalence is not claimed.

Build the existing `Provenance` with `estimator_settings=settings.reference` and
the exact versions from `numpyro_backend_versions()`. Other numerical dependencies
can be retained alongside them. Store the settings payload with its reference; a
hash does not store or authenticate its payload. The helper reads installed versions
without importing the backends. `model_dump_json()` saves the input record;
`manifest_json()` additionally includes the resolved fixed adapter policy and is
the exact content hashed by `reference`. The caller owns prior, model, data/unit, target,
source and numerical identities; the adapter does not infer their scientific truth
from a Python callable or hash arbitrary dynamic model inputs. Use independently
verified source/data manifests when scientific claims require them.

```python
from op_infer.adapters.numpyro import fit_nuts

# model is a trusted native NumPyro callable; provenance is constructed as above.
result = fit_nuts(
    model,
    settings=settings,
    provenance=provenance,
    model_args=(observations,),
    model_kwargs={},
)
posterior = result.posterior
```

Inputs are passed to the native model at runtime. They remain caller-owned; this
does not promise cross-backend differentiation or a generic compile-cache policy.
`rng_key`, `init_params` and `extra_fields` are reserved run controls in model kwargs.
Native model source and behavior-changing arguments must match their declared
provenance. The model must supply a deterministic differentiable potential conditional
on its latent state and recorded inputs; uncontrolled randomness is unsupported.
An augmented pseudo-marginal target is refused: the qualified augmented
adapter is not implemented by this interface. Declaring an analytic, deterministic or
frozen-random target never makes ordinary NUTS an exact pseudo-marginal algorithm.

## Complete layouts and retained native outputs

Every returned native sample/deterministic site must have a `SiteLayout`. There is
no implicit dropping, flattening, renaming or coordinate inference. Scalar sites
have no event dimensions. For a vector site, declare its event dimension and supply
an ordered shared `Coordinate`; all event coordinates must be used exactly. Names
must satisfy the shared posterior format's portable ASCII identifier contract.
Native dotted names require a separately scoped explicit mapping; they are refused
by this initial interface. Chain/draw and diagnostic names are reserved dimensions.

The shared posterior contains all post-warmup draws as native JAX arrays, with shape
`(chain, draw, *event_dimensions)` and no implicit thinning. The same objects remain
available in `result.native.get_samples(group_by_chain=True)`. Sample statistics
retain `accept_prob`, `diverging`, `num_steps` and `potential_energy`, including every
transition. The original native MCMC object remains available for advanced inspection.
Native arrays retain their mutability; immutable metadata is not an integrity digest.

## Diagnostics and failures

`result.diagnostics` retains native NumPyro effective sample size and **classic split
R-hat** per site/event element. These are not rank-normalized R-hat or bulk/tail ESS.
The adapter explicitly computes these diagnostics on host views; it leaves posterior
arrays native. A single original chain has `split_r_hat=None`, even though splitting
one chain could give a within-chain comparison. Constant or otherwise undefined
diagnostics retain their nonfinite entries. No site disappears from the denominator.

For rank-normalized R-hat and absolute bulk/tail ESS on selected reported quantities,
install the `diagnostics` extra and use
[`adapters.arviz.diagnose_mcmc`](mcmc-diagnostics.md). Pass `to_numpy=np.asarray`
for an explicit JAX host transfer. The separate report retains unavailable elements,
original coordinates, dependency versions and source/input identities. It leaves
the native posterior and these classic diagnostics unchanged.

`divergences_per_chain` counts every divergent retained transition; `nonfinite_values`
counts scalar entries across all returned sites, including deterministic quantities.
This is a value count, not a count of unique failed replicates. Raw arrays and native
transition statistics remain available to independently reconstruct the counts.

`adequacy_status` is always `not_assessed`. The adapter does not convert finite
diagnostics or zero divergences into a calibrated gate pass. Prespecified
thresholds, coverage and readable/passed policies remain
separate from these raw diagnostics. A native initialization/execution failure
propagates as a failed attempt; the adapter invents neither samples nor a successful
posterior. The orchestration layer must retain that failure in its planned denominator.

## Qualification and remaining scope

After installing the extra, `just qualify-numpyro` requires real JAX/NumPyro imports,
strict typing and the complete metadata/native-fit tests. CPU CI runs current Python
endpoints and an independently pinned minimum NumPyro/JAX environment. Tests check
analytic moments, actual native replay, identity/layout guards, named vector outputs,
single-chain/constant-site diagnostic limits and surfaced invalid initialization.
`just example-numpyro` executes the complete documented example.

SVI and further native bindings remain planned. Predictive artifacts,
coordinate/stage/context semantics and optional structural/study reporting remain
planned. Per-unit likelihood export is absent from this adapter; use the
[separate SBC runner](sbc-runner.md) to schedule calibration fits. Backend-neutral
parameter/prior/observation catalogs, GPU/parallel chains,
custom initialization/mass blocks and general `LogDensity` adaptation remain planned.

The wrapper follows [NumPyro's MCMC API](https://num.pyro.ai/en/stable/mcmc.html) and
[native diagnostic definitions](https://num.pyro.ai/en/stable/diagnostics.html).
