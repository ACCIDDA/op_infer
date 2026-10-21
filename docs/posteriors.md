# Posterior containers and snapshots

`op_infer.posterior.Posterior` stores supplied
draws, sample statistics and optional per-unit log likelihoods with the existing
`Provenance` record. It does not run inference, qualify sampler adequacy, merge
shards or produce predictive reports.

## Native arrays and explicit coordinates

Every variable starts with two sample axes: `chain, draw` or `seed, draw`.
Training seeds remain seeds, not interchangeable MCMC chains. All variables use
the same member/draw coordinates and one native array backend. Event dimensions
follow those axes. Coordinates are required explicitly, are ordered and unique,
and cannot be silently sorted, aligned, flattened or broadcast.

```python
import numpy as np
from op_infer.posterior import Coordinate, Posterior, SampledVariable

# Construct the existing complete Provenance record independently; see
# design/numerical-provenance.md. Its estimator_stage is the container's stage.
result = Posterior(
    draws=(SampledVariable("rate", np.ones((2, 3))),),
    coordinates=(
        Coordinate(name="chain", labels=(11, 23)),
        Coordinate(name="draw", labels=(0, 1, 2)),
        Coordinate(name="unit", labels=("u3", "u1")),
    ),
    provenance=provenance,
    sample_stats=(SampledVariable("diverging", np.zeros((2, 3), dtype=bool)),),
    log_likelihood=(SampledVariable("outcome", np.full((2, 3, 2), -1.0), ("unit",)),),
)
```

Replace the supplied arrays with JAX or Torch arrays to retain native computation
and gradients. Construction validates metadata, shape and dtype without reading
array values, moving devices, detaching graphs or promoting precision. Generic
typing preserves the supplied array type. A container is a Python reporting/data
object, not a registered JAX pytree or an estimator-native posterior distribution.
It may be constructed inside a traced function; returning it directly from JIT
is not qualified. Backend transfer and differentiation through transfer remain
separate operations.

The frame is frozen but native arrays retain their backend's mutability. Call
`validate()` before reuse after mutation; snapshots do this automatically.
Metadata immutability is not a content digest for live arrays.
Container/variable equality uses object identity; compare arrays and metadata
explicitly when numerical equality is intended. Values can include
NaN/infinities: no failed sample or unit is deleted, and no count is silently
renormalized. This container does not certify the values as valid posterior draws.
Readers must apply the independently declared adequacy/failure policies.

Log likelihoods have exactly one event dimension, `unit`, with explicit unit IDs.
They use real floating arrays and retain the full unit population. Whether a unit
likelihood was integrated over latent variables is part of the
[likelihood-target contract](design/likelihood-targets.md) and its provenance;
the container cannot establish it from labels or array shape. MCMC R-hat/ESS
values are available through the [optional named diagnostic adapter](mcmc-diagnostics.md);
adequacy decisions and retained diagnostic history remain project/gate obligations.
One stored likelihood variable is not automatically a qualified comparison score.

## Optional ArviZ 1.x and netCDF

Install `op_infer[arviz]`. Core imports and native data-model use need none of
ArviZ, xarray, netCDF4, JAX, Torch or flepimop2. Only the optional adapter imports
I/O libraries when used.

```python
from op_infer.adapters.arviz import from_arviz, load_netcdf, to_arviz

# Every value must already be an unmasked NumPy array. Explicitly transfer and
# detach native values before constructing a host container for reporting.
tree = to_arviz(result)
restored = from_arviz(tree, expected=provenance)
tree.to_netcdf("posterior.nc", engine="netcdf4")
restored_from_disk = load_netcdf("posterior.nc", expected=provenance)
```

ArviZ 1.x uses `xarray.DataTree`, not the legacy `InferenceData` interface. The
adapter uses [ArviZ's nested-group conversion](https://python.arviz.org/projects/base/en/stable/api/generated/arviz_base.from_dict.html)
and [xarray's netCDF support](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.to_netcdf.html).
It writes `posterior`, optional `sample_stats` and optional `log_likelihood`
groups with explicit `sample_dims`. Variables can have different event dimensions
in different groups, even when they share a name. Generic ArviZ consumers may
assume chain semantics; select the actual seed/draw dimensions explicitly, and
never relabel training seeds as chains to calculate MCMC diagnostics.

Snapshots copy arrays and preserve coordinate order, variable order, dimensions,
bool/integer/float32/float64 values and source dtype interpretation, including
noncontiguous and non-native-endian inputs. Float16, wider floats, masks, complex
and object arrays require explicit caller-side conversion; no silent promotion
or truncation is performed. Restored values are detached NumPy arrays, not native
graphs. Round trips preserve non-finite likelihood outcomes instead of dropping
them. File loading eagerly copies the values and closes its file handles.

The versioned header embeds the shared complete provenance, coordinate/layout
metadata and existing `ArtifactReference` digests of canonical little-endian
C-order numeric bytes. Restoration first requires independently supplied expected
provenance, then refuses missing/unknown schema, undeclared groups/variables,
changed coordinate/sample meanings, dtypes and array corruption. Arbitrary or
legacy sampler trees require an explicit adapter; provenance is never invented.
Header/digests are self-recorded integrity checks, not authentication against a
malicious replacement, availability checks on referenced artifacts or proof of
their scientific meaning. Full artifact availability and shard validation
remain separate, unimplemented contracts. Xarray file-writing behavior, overwrite rules and concurrent publication
are not changed by this adapter; manage output paths accordingly.

Run `just ci` for core submission checks. After installing `arviz`, run
`just qualify-posterior` for strict typing with the real optional dependencies and
actual DataTree/netCDF cases. With JAX and Torch installed,
`just qualify-native-posterior` requires both backends and tests JIT/autograd and
mixed-backend refusal. CI runs snapshot qualification at Python 3.12 and 3.14 and
native graph checks in the existing required CPU qualification job. These
explicit recipes fail on missing dependencies; a skipped optional suite is not
reported as successful qualification.
