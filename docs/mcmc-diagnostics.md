# Named MCMC diagnostics

`op_infer.adapters.arviz.diagnose_mcmc` calculates rank-normalized R-hat and
absolute bulk/tail effective sample sizes (ESS) for explicitly selected posterior
quantities. It wraps the documented NumPy interface in ArviZ Stats 1.x. The
`diagnostics` extra needs NumPy and SciPy; xarray and netCDF are not required.

```bash
uv sync --dev --extra diagnostics
uv run python examples/mcmc_diagnostics.py
```

The [complete example](https://github.com/ACCIDDA/op_infer/blob/main/examples/mcmc_diagnostics.py)
prints a table from independent analytic Normal draws, a deliberately shifted
chain, a rescaled quantity, a constant and a nonfinite input. All five scalar
elements appear, including the two incomplete elements. It demonstrates reporting;
it does not fit a sampler or assess sampler adequacy.

## Select the quantities to report

Supply a validated `Posterior` with its original post-warmup chain and draw axes.
Select names explicitly, in reporting order, and provide a content reference to the
quantity definition. For reported transformations or data-dependent alignment,
retain the transformation, alignment rule and draw selection in that referenced
definition. The function reads supplied quantities; it does not align, transform,
thin or choose them. Retain any definition payload with its reference.

```python
from op_infer.adapters.arviz import diagnose_mcmc

# posterior and definition are independently constructed project artifacts.
report = diagnose_mcmc(
    posterior,
    quantities=("theta", "reported"),
    definition=definition,
)
```

Every selected site and event element remains in its original order. A scalar
has index `()`; an event tensor has one index per declared event dimension.
`report.coordinates` retains the original coordinate labels and ordering, and
each quantity's `dims` maps its component indices to those labels. Chains and
draws are passed to ArviZ separately, without pooling them. Odd draw counts remain
in the input record; ArviZ's split-chain calculation uses its documented splitting
rule internally.

Native JAX or Torch arrays require an explicit host-transfer callback. For JAX,
pass `to_numpy=np.asarray`. For Torch, use a callback that explicitly detaches,
transfers to CPU and converts to NumPy. The callback must preserve shape, axis
order and values; equal shapes alone cannot prove that it did. Computation uses
a copied host array, leaving the original posterior and native fit diagnostics
unchanged. This is a reporting boundary, not a differentiable operation.

## Methods and unavailable values

| Field | ArviZ method and settings |
| --- | --- |
| `rank_rhat` | `rhat(method="rank")`: maximum of rank-normalized split R-hat and rank-normalized folded split R-hat. At least two original chains are required. |
| `bulk_ess` | `ess(method="bulk", relative=False)`: absolute ESS after rank normalization and splitting. |
| `tail_ess` | `ess(method="tail", relative=False, prob=(0.05, 0.95))`: minimum quantile ESS at the two stated probabilities. |

Each `DiagnosticValue` has a method, status, optional finite positive value and
an explicit reason when no value is available. ESS is an absolute count and may
exceed the number of draws for antithetic samples; the wrapper does not cap it.

| Input case | Reported outcome |
| --- | --- |
| One original chain | R-hat unavailable (`single_original_chain`); ESS is calculated when its other preconditions hold. Splitting does not create an independent original chain. |
| Fewer than four draws per chain | All three diagnostics unavailable. |
| Constant quantity, or each chain stuck at its own constant | All three undefined. Constants remain selected elements. |
| Any NaN or infinity in an element | All three unavailable; the number of nonfinite values is retained. No draws are filtered. |
| Training-seed axis | All three unavailable. Seeds retain their meaning and are never relabeled as MCMC chains. |
| Unsupported host conversion, mask or dtype | All affected elements unavailable with conversion reasons; no selected row disappears. |
| Reference calculation fails or returns a nonpositive/nonfinite result | That metric is undefined with a reason; other metrics remain independently visible. |

Supported host inputs are unmasked NumPy bool, 8–64-bit integer and float32/float64
arrays. Diagnostics use float64, with source and host dtypes recorded separately.
Integer elements containing values outside the exactly representable range
`[-2**53, 2**53]` are unavailable to avoid changing ranks during conversion.
Unsupported precision requires a deliberate project-side conversion before this
boundary; the wrapper does not silently convert half precision or discard masks.

`n_components` is the full selected scalar-element denominator. `n_incomplete`
counts elements with any unavailable or undefined metric. The counts do not treat
a poor finite R-hat or ESS as missing. `adequacy_status` is always `not_assessed`.
No finite result, complete denominator or zero divergence count establishes
adequacy. Prespecified project thresholds, divergence counts, calibration evidence
and combined gate qualification remain separate. The existing NumPyro adapter
continues to report its native classic split R-hat and native ESS under their
original names; those are different diagnostics.

## Save the calculation context

The immutable `MCMCDiagnostics` record includes the shared fitting provenance,
quantity-definition reference, coordinate layout, metric settings, implementation
name and diagnostic dtype. It separately records the installed op_infer,
ArviZ Stats, NumPy, SciPy and Python versions and the adapter source digest, since
diagnostics may be calculated after fitting in a different environment.
Each successfully copied quantity includes a digest of its original host dtype's
canonical little-endian C-order bytes, before float64 calculation. A failed host
conversion has no input digest and an unknown nonfinite count.

```python
from op_infer.adapters.arviz import MCMCDiagnostics

payload = report.model_dump_json()
restored = MCMCDiagnostics.load_json(
    payload,
    expected=posterior.provenance,
    definition=definition,
)
```

Loading requires independently supplied fitting provenance and the quantity
definition. It checks the schema, layout, complete event denominator and basic
method preconditions. Self-recorded source/data digests do not authenticate a
replaced report, retain the referenced payloads, or prove the quantities' scientific
meaning. Projects must retain and verify the artifacts they require for reuse.

## Qualification

After installing `diagnostics` and `numpyro`, run `just qualify-diagnostics`.
The recipe requires the actual libraries, runs strict typing, compares scalar,
vector and tensor outputs with ArviZ Stats, and checks location/scale disagreements,
sticky tails, constants, nonfinite data, original-chain limits and explicit native
host transfer. It also executes the standalone table example and an actual
NumPyro Gaussian fit checked against analytic moments. CI runs current dependencies
on Python 3.12 and 3.14 and ArviZ Stats 1.0 with the NumPy/JAX/NumPyro floors on
Python 3.12. Missing optional libraries
fail this qualification rather than produce a successful skipped suite.

The method definitions follow the official
[R-hat array API](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.base.array_stats.rhat.html)
and [ESS array API](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.base.array_stats.ess.html).
