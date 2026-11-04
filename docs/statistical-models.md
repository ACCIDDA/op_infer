# Shared priors and statistical observations

`JointPrior` declares ordered, named independent coordinates. Each coordinate is
scalar or a fixed-shaped iid event. The initial families are `Normal(loc, scale)`,
`LogNormal(loc, scale)` on log coordinates, `Uniform(low, high)`,
`Gamma(concentration, rate)` and `Beta(concentration1, concentration0)`. Positive
scales/shapes/rates, finite parameters, unique names and event layouts are checked
before sampling. Frozen declarations serialize to versioned content references;
changing order, shape or family changes the prior identity.

```python
import numpy as np
from op_infer.priors import JointPrior, LogNormal, PriorParameter
from op_infer.rng import RandomKey

prior = JointPrior(
    parameters=(
        PriorParameter(name="rate", distribution=LogNormal(loc=-0.5, scale=0.3)),
    )
)
parameters = prior.sample(RandomKey(seed=2026, stream="parameters"), sample_shape=(16,))
assert prior.log_prob(parameters).shape == (16,)
```

The NumPy sampler delegates to SciPy with fresh local generators addressed by
parameter name. Adding/reordering another parameter preserves a name's draws;
changing its shape, parameters or dependency versions need not. There is no
implicit rejection, retry or truncation. Proper bounded priors remain explicit
probability laws, rather than unrecorded filters on a training bank.

`numpyro_sites()` and `torch_sites(dtype=..., device=...)` return native named
laws, with declared coordinate shapes marked as event axes. Native libraries own
fitting and sampling. `log_prob` supplies shared support-aware scoring on native
NumPy, JAX or Torch arrays, summing event axes while retaining matching batch
axes and gradients. Uniform scoring uses `low <= x < high`. Raw native laws can
have different validation/endpoint conventions; use the shared scorer when those
conventions matter. In particular, native NumPyro validation is not a substitute
for support checks on arbitrary evaluation points.

The optional JAX scorer requires NumPyro. All native factories are lazy imports.
`torch_normal_vector()` lowers scalar Normal coordinates in their declared order
to the full-real vector law expected by the existing sbi adapter. Other supports
are refused by that convenience: there is no automatic Jacobian, coordinate
change or altered training prior. Correlated/hierarchical priors, fixed-coordinate
and general transform catalogs remain separate planned interfaces.

## Observation parameterizations

`IIDObservation` uses the same model for `sample` and per-cell `log_prob`:

| Family | Meaning of affine latent value `offset + slope * latent` | Other parameters |
| --- | --- | --- |
| `normal` | Physical mean | `scale` is standard deviation |
| `lognormal` | Log-location | `scale` is log-standard deviation |
| `poisson` | Nonnegative mean | Variance equals mean |
| `nb2` | Nonnegative mean `mu` | Positive `concentration=k`; variance `mu + mu**2/k` |
| `binomial` | Probability | Integer `trials` |
| `beta_binomial` | Probability strictly between zero and one | Integer `trials`; positive concentration; shapes `p*k`, `(1-p)*k` |
| `bernoulli` | True-positive probability | Positive-test probability `(1-specificity) + (sensitivity+specificity-1)*p` |

Bernoulli/binomial endpoints and NB2 zero mean receive their exact degenerate
probabilities, avoiding finite clamped-logit values for impossible events.
Count support excludes fractional and negative observations. Invalid latent
parameters score as impossible; native sampling may fail on invalid inputs.
Keep failures visible instead of dropping attempts. Returned counts use the
latent floating dtype; choose precision appropriate to the count range.

`mask=True` means observed. Masks must be native boolean arrays with the broadcast
observation shape. Missing values are replaced before density evaluation and
contribute zero; observed zeros remain observations. No reduction across units
or batches is implicit. Statistical missingness assumptions remain caller-owned.

Normal observations optionally declare `lower_limit` and/or `upper_limit`.
Sampling clips to those limits. Scoring a limit uses the corresponding tail
probability, while interior values use density and values beyond a limit are
impossible. This is a censored probability law, not a truncated/rejection-sampled
Normal. Stable native log-CDF kernels retain tail derivatives. AR(1) dependence
and non-Normal censoring remain planned; unsupported censoring declarations fail
rather than silently switching likelihood.

```python
from op_infer.observe import IIDObservation

sensor = IIDObservation(family="normal", scale=0.2, lower_limit=0.0)
latent = np.array([0.4, 0.8])
measurements = sensor.sample(latent, RandomKey(seed=2026, stream="sensor"))
terms = sensor.log_prob(latent, measurements, mask=np.array([True, True]))
assert terms.shape == latent.shape
```

### Quantized Normal measurements

`BinnedNormalObservation` records which interval contains an underlying Normal
measurement. Declare at least one finite, strictly increasing `cutpoints` tuple.
For `K` cutpoints, the observation is a zero-based code in `0, ..., K`:
`0` means `(-inf, c[0])`, an interior code `j` means `[c[j-1], c[j])`,
and `K` means `[c[K-1], inf)`. An exact boundary draw belongs to the higher bin.
Codes label intervals; they are neither physical measurements nor midpoints.
The affine mean and standard deviation have the same meanings as in the
continuous Normal law. Cutpoints use those measurement coordinates, so callers
must explicitly transform an instrument's reporting scale when necessary.

Sampling draws through the existing Normal implementation and records the bin.
Scoring returns the Normal probability of that same interval, with open-ended
limits receiving tail probability. Finite intervals use native log-CDF kernels
and stable log differences, reflecting right-tail intervals to avoid subtracting
probabilities rounded to one. Tests compare extreme and narrow intervals with
independent scaled quadrature and latent gradients with truncated-Normal moments.
For intervals crossing the mean, lower tails below -40 standard deviations have
unrepresentable float64 mass and are excluded before native differentiation;
this prevents an inactive tail from contaminating a saturated bin's gradient.
NumPy, JAX and Torch CPU float32/64 values and sampling are qualified; float64
gradient and JAX JIT/vmap checks cover the declared scope. This new law has not
been separately qualified on CUDA. Precision must resolve cutpoints, their
standardized separation and the log-CDF difference; very narrow intervals can
lose relative accuracy, and differences that round to zero have zero computed
probability. Use sufficient precision for the declared instrument partition.

```python
from op_infer.observe import BinnedNormalObservation

quantized = BinnedNormalObservation(cutpoints=(-0.5, 0.25, 1.0), scale=0.2)
codes = quantized.sample(latent, RandomKey(seed=2026, stream="quantized-sensor"))
log_mass = quantized.log_prob(latent, codes)
assert np.isfinite(log_mass).all()
```

Integer or real floating arrays may hold codes; samples retain the latent
floating dtype. Fractional, negative, nonfinite or out-of-range codes score as
impossible. Missing masks follow the existing native boolean convention and
allow missing NaNs to contribute zero. The content reference records the entire
partition, boundary convention, affine parameters and noise scale. A partition
is fixed for the whole observation channel; distinct instruments can use distinct
named channels. Arbitrary per-cell interval bounds, mixed exact and interval
reports, correlated errors and a dedicated curvature API remain outside this
interface. The implementation partially addresses
[observation issue #6](https://github.com/ACCIDDA/op_infer/issues/6).

Run the independent quantized-sensor frequency example with
`uv run python examples/quantized_sensor.py`. This is the fixed-cutpoint Normal
interval law also used by
[PyMC OrderedProbit](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.OrderedProbit.html)
and [Stan ordered probit](https://mc-stan.org/docs/functions-reference/bounded_discrete_distributions.html#ordered-probit-distribution),
after adjusting for their respective scale and code conventions.
Tail evaluation uses the native equivalent of
[SciPy log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html).

NumPy uses SciPy laws, JAX uses NumPyro and native JAX keys, and Torch uses local
CPU or device-matched CUDA generators. Gamma-Poisson/beta-binomial Torch draws use Torch's native Gamma
primitive with an explicit generator; that private primitive is covered at the
required Torch dependency floor and current version. No global RNG state is
seeded or consumed. JAX keys can be dynamic inputs to jitted/vmapped sampling;
`RandomKey` lowering is host setup. Cross-backend bitwise draw equality is not a
promise. Beta-binomial scoring uses native log-gamma identities on JAX/Torch:
the qualified JAX beta-log implementation showed about 1e-6 error on a small
float64 reference, so scoring does not use that approximation.

## Compose with an identified forward

`StatisticalModel` binds the existing `ForwardModel`, `JointPrior` and named
`ObservationChannel`s. Construction checks prior identity/order against forward
provenance and checks channel-to-latent bindings. `simulate` takes explicit native
parameters; `sample_numpy` separately draws NumPy parameters on the parameter
stream. Each channel receives independent, name-addressed observation noise.
`SimulationResult` keeps parameters, latent outputs, measurements, stream
addresses and the complete pipeline reference separate.

`log_prob` evaluates those same observation objects and returns named per-cell
terms, with explicit optional masks. The pipeline identity includes observation
laws, channel bindings, prior and complete forward provenance. It does not
silently replace a saved bank's prior/target or upgrade old artifacts. Keep the
pipeline reference in the caller's immutable artifact manifest and supply its
expected identity independently on reuse. Borrowed arrays remain mutable native
objects until an explicit snapshot boundary.

This composition works with callable or qualified compiled forwards. No second
ODE, sampler or trade-study runner is introduced. Optional CPU qualification
covers independent distribution references, support, moments, event/batch shapes,
JAX JIT/vmap, native gradients, channel streams and the analytic decay pipeline:

```shell
uv sync --dev --extra numpyro
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
just qualify-statistics
```

The [separate A100 qualification](cuda-qualification.md) covers scoped JAX/Torch
CUDA densities, sampling and gradients at float32/64. CuPy statistical lowering
remains explicitly unsupported. [CPU pipeline timing and memory](pipeline-performance.md)
and application performance have separate evidence requirements.

References: [SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html),
[NumPyro distributions](https://num.pyro.ai/en/latest/distributions.html),
[PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html).
