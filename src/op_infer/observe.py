"""Native iid observation laws shared by simulation and likelihood evaluation.

Latent means/log-locations/probabilities remain separate from random measurements.
Array operations preserve NumPy, JAX or Torch execution; native distributions own
sampling and density kernels. Correlated observations are not silently iid.
"""

from __future__ import annotations

import importlib
import itertools
import json
from typing import Annotated, Any, Literal, Protocol, Self

from array_api_compat import array_namespace
from pydantic import Field, model_validator

from op_infer.posterior import ArtifactReference
from op_infer.priors import _backend, _Record, _validate_sample_shape
from op_infer.rng import RandomKey

__all__ = ["BinnedNormalObservation", "IIDObservation", "ObservationModel"]

type _Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type _Finite = Annotated[float, Field(allow_inf_nan=False)]
type _Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
type _Family = Literal[
    "normal", "lognormal", "poisson", "nb2", "binomial", "beta_binomial", "bernoulli"
]


class ObservationModel(Protocol):
    """Expose one statistical target for sampling, scoring and artifact identity."""

    @property
    def reference(self) -> ArtifactReference:
        """Identify family, parameterization, support and censoring semantics."""
        ...

    def log_prob(self, latent: Any, observed: Any, *, mask: Any = None) -> Any:  # noqa: ANN401
        """Return per-cell density/mass with True meaning observed."""
        ...

    def sample(
        self,
        latent: Any,  # noqa: ANN401
        key: Any,  # noqa: ANN401
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:  # noqa: ANN401
        """Draw addressed native measurements without changing global RNG state."""
        ...


class BinnedNormalObservation(_Record):
    """Record a Normal measurement as an interval code on a fixed partition.

    With ``K`` finite, strictly increasing cutpoints, codes ``0, ..., K``
    denote ``(-inf, c[0])``, ``[c[0], c[1])``, ..., ``[c[K-1], inf)``.
    Cutpoints use measurement coordinates, whose Normal mean is
    ``offset + slope * latent`` and standard deviation is ``scale``.
    Codes are labels, not interval midpoints or numerical measurements.

    Sampling quantizes the existing native Normal sampler. Scoring returns
    interval probability mass, including both open tails. Invalid codes or
    nonfinite means score as impossible; masked cells contribute zero even
    when missing values are NaN. A common partition applies to every cell.
    Per-cell partitions, correlated errors and explicit curvature interfaces
    are outside this law.

    Examples:
        >>> import numpy as np
        >>> sensor = BinnedNormalObservation(cutpoints=(-1.0, 1.0))
        >>> np.exp(sensor.log_prob(np.array(0.0), np.arange(3))).round(6)
        array([0.158655, 0.682689, 0.158655])
    """

    cutpoints: Annotated[tuple[_Finite, ...], Field(min_length=1)]
    scale: _Positive = 1.0
    offset: _Finite = 0.0
    slope: _Finite = 1.0

    @model_validator(mode="after")
    def _ordered_cutpoints(self) -> Self:
        if any(a >= b for a, b in itertools.pairwise(self.cutpoints)):
            msg = "Normal bin cutpoints must be strictly increasing"
            raise ValueError(msg)
        return self

    @property
    def reference(self) -> ArtifactReference:
        """Identify the partition, boundary convention, affine map and noise."""
        data = {
            "schema_version": 1,
            "family": "normal",
            "dependence": "iid",
            "mask": "true-is-observed",
            "observation": "zero-based-bin-code",
            "intervals": "left-closed-right-open-with-infinite-end-tails",
            "parameters": self.model_dump(mode="json"),
        }
        payload = json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return ArtifactReference.from_bytes(
            payload.encode(), kind="binned-normal-observation-v1"
        )

    def log_prob(self, latent: Any, observed: Any, *, mask: Any = None) -> Any:  # noqa: ANN401
        """Score integer-valued bin codes with native tail-stable probabilities.

        Integer and real floating code arrays are accepted; fractional,
        negative, nonfinite and out-of-range codes receive ``-inf``. Masks
        must be native boolean arrays with exactly the broadcast shape.
        Native JAX/Torch derivatives are with respect to the latent input.
        Choose precision that resolves cutpoints, standardized separation and
        log-CDF differences. Very narrow intervals can lose relative accuracy;
        differences that collapse in that precision have zero computed mass.
        """
        xp = array_namespace(latent, observed)
        backend = _backend(xp)
        if not xp.isdtype(latent.dtype, "real floating"):
            msg = "Observation latent values must have real floating dtype"
            raise TypeError(msg)
        if not xp.isdtype(observed.dtype, ("integral", "real floating")):
            msg = "Normal bin codes must have integer or real floating dtype"
            raise TypeError(msg)
        mean, code = xp.broadcast_arrays(self.offset + self.slope * latent, observed)
        selected = xp.ones_like(mean, dtype=xp.bool) if mask is None else mask
        if (
            array_namespace(selected).__name__ != xp.__name__
            or selected.dtype != xp.bool
            or tuple(selected.shape) != tuple(mean.shape)
        ):
            msg = "Observation mask must be native boolean with the broadcast shape"
            raise ValueError(msg)
        valid = xp.isfinite(mean)
        mean = xp.where(selected & valid, mean, xp.zeros_like(mean))
        code = xp.where(selected, code, xp.zeros_like(code))
        valid = (
            valid
            & xp.isfinite(code)
            & (code >= 0)
            & (code <= len(self.cutpoints))
            & (code == xp.floor(code))
        )
        interior = valid & (code > 0) & (code < len(self.cutpoints))
        lower, upper = -xp.ones_like(mean), xp.ones_like(mean)
        for index, (left, right) in enumerate(
            itertools.pairwise(self.cutpoints), start=1
        ):
            lower = xp.where(code == index, xp.full_like(mean, left), lower)
            upper = xp.where(code == index, xp.full_like(mean, right), upper)
        center = xp.where(interior, mean, xp.zeros_like(mean))
        lower = (lower - center) / self.scale
        upper = (upper - center) / self.scale
        # Reflect right-tail intervals before subtraction; never subtract CDFs
        # rounded to one. Inactive open tails retain finite dummy endpoints.
        reflected = lower > 0
        low = xp.where(reflected, -upper, lower)
        high = xp.where(reflected, -lower, upper)
        special = importlib.import_module(
            {
                "numpy": "scipy.special",
                "jax": "jax.scipy.special",
                "torch": "torch.special",
            }[backend]
        )
        log_high = special.log_ndtr(high)
        # For intervals crossing the mean, a lower tail below -40 standard
        # deviations has less mass than float64 can represent. Do not evaluate
        # its native backward kernel: its undefined derivative can otherwise
        # contaminate an exactly saturated interval through zero times infinity.
        negligible_lower_tail = (low < -40) & (high >= 0)
        safe_low = xp.where(negligible_lower_tail, xp.zeros_like(low), low)
        difference = special.log_ndtr(safe_low) - log_high
        resolved = difference < 0
        safe_difference = xp.where(resolved, difference, -xp.ones_like(mean))
        density = xp.where(
            resolved, log_high + xp.log(-xp.expm1(safe_difference)), -xp.inf
        )
        density = xp.where(negligible_lower_tail, log_high, density)
        left_mean = xp.where(code == 0, mean, xp.full_like(mean, self.cutpoints[0]))
        right_mean = xp.where(
            code == len(self.cutpoints),
            mean,
            xp.full_like(mean, self.cutpoints[-1]),
        )
        density = xp.where(
            code == 0,
            special.log_ndtr((self.cutpoints[0] - left_mean) / self.scale),
            density,
        )
        density = xp.where(
            code == len(self.cutpoints),
            special.log_ndtr((right_mean - self.cutpoints[-1]) / self.scale),
            density,
        )
        return xp.where(selected, xp.where(valid, density, -xp.inf), 0.0)

    def sample(
        self,
        latent: Any,  # noqa: ANN401
        key: Any,  # noqa: ANN401
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:  # noqa: ANN401
        """Quantize addressed native Normal draws, returning latent float dtype.

        A draw exactly on a cutpoint enters the higher bin. Replay, leading
        sample axes and local-generator semantics are those of IIDObservation.
        Nonfinite underlying draws remain NaN, never valid-looking bin codes.
        """
        draws = IIDObservation(
            family="normal", scale=self.scale, offset=self.offset, slope=self.slope
        ).sample(latent, key, sample_shape=sample_shape)
        xp = array_namespace(draws)
        codes = xp.zeros_like(draws)
        for cutpoint in self.cutpoints:
            codes = codes + xp.astype(draws >= cutpoint, draws.dtype)
        return xp.where(xp.isfinite(draws), codes, xp.full_like(draws, xp.nan))


class IIDObservation(_Record):
    """Independent measurements on an affine map of one named latent output.

    Normal uses a physical mean and standard deviation; LogNormal uses a
    log-location and log-standard deviation. NB2 has variance mu + mu**2 / k,
    where ``concentration=k``. Beta-binomial uses probability p and concentration
    k, hence alpha=p*k and beta=(1-p)*k. Bernoulli sensitivity/specificity map a
    true probability to the probability of a positive measurement.

    Optional Normal clipping records left/right censoring as probability mass
    at the limit. Missing cells contribute zero and can contain NaNs. Values
    outside support contribute -inf. Families with other censoring or dependence
    require another explicitly qualified ObservationModel.

    Examples:
        >>> import numpy as np
        >>> model = IIDObservation(family="normal", scale=0.5)
        >>> model.log_prob(np.array([0.0]), np.array([0.0])).round(6)
        array([-0.225791])
    """

    family: _Family
    scale: _Positive = 1.0
    concentration: _Positive = 1.0
    trials: Annotated[int, Field(ge=0)] = 1
    sensitivity: _Probability = 1.0
    specificity: _Probability = 1.0
    offset: _Finite = 0.0
    slope: _Finite = 1.0
    lower_limit: _Finite | None = None
    upper_limit: _Finite | None = None

    @model_validator(mode="after")
    def _limits(self) -> Self:
        if (
            self.lower_limit is not None or self.upper_limit is not None
        ) and self.family != "normal":
            msg = "Censoring is qualified only for Normal measurements"
            raise ValueError(msg)
        if (
            self.lower_limit is not None
            and self.upper_limit is not None
            and self.lower_limit >= self.upper_limit
        ):
            msg = "Censoring limits require lower_limit < upper_limit"
            raise ValueError(msg)
        return self

    @property
    def reference(self) -> ArtifactReference:
        """Return stable versioned observation-law identity."""
        data = {
            "schema_version": 1,
            "dependence": "iid",
            "mask": "true-is-observed",
            "parameters": self.model_dump(mode="json"),
        }
        payload = json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return ArtifactReference.from_bytes(payload.encode(), kind="iid-observation-v1")

    def log_prob(self, latent: Any, observed: Any, *, mask: Any = None) -> Any:  # noqa: ANN401
        """Score measurements per cell, retaining graphs and leading batch axes.

        The latent input must be a native floating array. Observed arrays share
        its namespace. Masks must be boolean and match the broadcast result;
        excluded values are replaced before density evaluation, preventing
        missing NaNs from contaminating gradients.
        """
        xp = array_namespace(latent, observed)
        backend = _backend(xp)
        if not xp.isdtype(latent.dtype, "real floating"):
            msg = "Observation latent values must have real floating dtype"
            raise TypeError(msg)
        mean, value = xp.broadcast_arrays(self.offset + self.slope * latent, observed)
        selected = xp.ones_like(mean, dtype=xp.bool) if mask is None else mask
        if (
            array_namespace(selected).__name__ != xp.__name__
            or selected.dtype != xp.bool
            or tuple(selected.shape) != tuple(mean.shape)
        ):
            msg = "Observation mask must be native boolean with the broadcast shape"
            raise ValueError(msg)
        safe_mean, valid = self._safe_mean(mean, xp)
        placeholder = 1.0 if self.family == "lognormal" else 0.0
        value = xp.where(selected, value, xp.full_like(mean, placeholder))
        law = self._native(safe_mean, backend)
        if backend == "numpy":
            density = (
                law.logpmf(value)
                if self.family
                in {"poisson", "nb2", "binomial", "beta_binomial", "bernoulli"}
                else law.logpdf(value)
            )
        elif self.family == "beta_binomial":
            density = self._beta_binomial_density(safe_mean, value, xp)
        else:
            support = (
                law.support(value) if backend == "jax" else law.support.check(value)
            )
            safe_value = xp.where(support, value, xp.full_like(mean, placeholder))
            density = xp.where(support, law.log_prob(safe_value), -xp.inf)
        if self.family in {"binomial", "bernoulli"}:
            probability = (
                (1 - self.specificity)
                + (self.sensitivity + self.specificity - 1) * mean
                if self.family == "bernoulli"
                else mean
            )
            trials = 1 if self.family == "bernoulli" else self.trials
            density = xp.where(
                probability == 0, xp.where(value == 0, 0.0, -xp.inf), density
            )
            density = xp.where(
                probability == 1, xp.where(value == trials, 0.0, -xp.inf), density
            )
        if self.family in {"poisson", "nb2"}:
            density = xp.where(mean == 0, xp.where(value == 0, 0.0, -xp.inf), density)
        density = xp.where(valid, density, -xp.inf)
        if self.lower_limit is not None or self.upper_limit is not None:
            special = importlib.import_module(
                {
                    "numpy": "scipy.special",
                    "jax": "jax.scipy.special",
                    "torch": "torch.special",
                }[backend]
            )
            if self.lower_limit is not None:
                z = (self.lower_limit - safe_mean) / self.scale
                density = xp.where(
                    value == self.lower_limit,
                    special.log_ndtr(z),
                    xp.where(value < self.lower_limit, -xp.inf, density),
                )
            if self.upper_limit is not None:
                z = (safe_mean - self.upper_limit) / self.scale
                density = xp.where(
                    value == self.upper_limit,
                    special.log_ndtr(z),
                    xp.where(value > self.upper_limit, -xp.inf, density),
                )
        density = xp.where(xp.isnan(value), xp.full_like(mean, xp.nan), density)
        return xp.where(selected, density, 0.0)

    def _safe_mean(self, mean: Any, xp: Any) -> tuple[Any, Any]:  # noqa: ANN401
        valid = xp.isfinite(mean)
        if self.family in {"poisson", "nb2"}:
            valid = valid & (mean >= 0)
        elif self.family in {"binomial", "bernoulli", "beta_binomial"}:
            valid = valid & (mean >= 0) & (mean <= 1)
            if self.family == "beta_binomial":
                valid = valid & (mean > 0) & (mean < 1)
        safe = xp.where(valid, mean, xp.full_like(mean, 0.5))
        if self.family in {"poisson", "nb2"}:
            safe = xp.where(mean == 0, xp.ones_like(mean), safe)
        return safe, valid

    def _native(self, mean: Any, backend: str) -> Any:  # noqa: ANN401, PLR0911, C901
        if backend == "numpy":
            stats = importlib.import_module("scipy.stats")
            if self.family == "normal":
                return stats.norm(loc=mean, scale=self.scale)
            if self.family == "lognormal":
                return stats.lognorm(
                    s=self.scale, scale=importlib.import_module("numpy").exp(mean)
                )
            if self.family == "poisson":
                return stats.poisson(mu=mean)
            if self.family == "nb2":
                return stats.nbinom(
                    n=self.concentration,
                    p=self.concentration / (self.concentration + mean),
                )
            if self.family == "beta_binomial":
                return stats.betabinom(
                    n=self.trials,
                    a=mean * self.concentration,
                    b=(1 - mean) * self.concentration,
                )
            probability = (
                (1 - self.specificity)
                + (self.sensitivity + self.specificity - 1) * mean
                if self.family == "bernoulli"
                else mean
            )
            return stats.binom(
                n=1 if self.family == "bernoulli" else self.trials, p=probability
            )
        distributions = importlib.import_module(
            "numpyro.distributions" if backend == "jax" else "torch.distributions"
        )
        if self.family in {"normal", "lognormal"}:
            return getattr(
                distributions, "Normal" if self.family == "normal" else "LogNormal"
            )(mean, self.scale, validate_args=False)
        if self.family == "poisson":
            return distributions.Poisson(mean, validate_args=False)
        if self.family == "nb2":
            if backend == "jax":
                return distributions.NegativeBinomial2(
                    mean, self.concentration, validate_args=False
                )
            return distributions.NegativeBinomial(
                self.concentration,
                probs=mean / (mean + self.concentration),
                validate_args=False,
            )
        if self.family == "beta_binomial":
            return (
                distributions.BetaBinomial(
                    mean * self.concentration,
                    (1 - mean) * self.concentration,
                    total_count=self.trials,
                    validate_args=False,
                )
                if backend == "jax"
                else None
            )
        probability = (
            (1 - self.specificity) + (self.sensitivity + self.specificity - 1) * mean
            if self.family == "bernoulli"
            else mean
        )
        return distributions.Binomial(
            total_count=1 if self.family == "bernoulli" else self.trials,
            probs=probability,
            validate_args=False,
        )

    def _beta_binomial_density(self, mean: Any, value: Any, xp: Any) -> Any:  # noqa: ANN401
        backend = _backend(xp)
        special = importlib.import_module(
            "jax.scipy.special" if backend == "jax" else "torch"
        )
        a = mean * self.concentration
        b = (1 - mean) * self.concentration
        valid = (value >= 0) & (value <= self.trials) & (value == xp.floor(value))
        safe = xp.where(valid, value, xp.zeros_like(mean))
        log_gamma = special.gammaln if backend == "jax" else special.lgamma
        n = xp.full_like(mean, float(self.trials))
        density = (
            log_gamma(n + 1)
            - log_gamma(safe + 1)
            - log_gamma(n - safe + 1)
            + log_gamma(safe + a)
            + log_gamma(n - safe + b)
            - log_gamma(n + a + b)
            - log_gamma(a)
            - log_gamma(b)
            + log_gamma(a + b)
        )
        return xp.where(valid, density, -xp.inf)

    def sample(
        self,
        latent: Any,  # noqa: ANN401
        key: Any,  # noqa: ANN401
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:  # noqa: ANN401
        """Draw native measurements; RandomKey lowering happens outside tracing.

        JAX accepts a native key for jitted/vmapped sampling. Torch accepts a
        local device-matched Generator; RandomKey creates one without seeding
        global state. Scoped Torch CUDA sampling uses float32/64. Returned counts
        use latent float dtype, matching native Torch count distributions.
        """
        _validate_sample_shape(sample_shape)
        xp = array_namespace(latent)
        backend = _backend(xp)
        if not xp.isdtype(latent.dtype, "real floating"):
            msg = "Observation latent values must have real floating dtype"
            raise TypeError(msg)
        mean = self.offset + self.slope * latent
        shape = sample_shape + tuple(mean.shape)
        if backend == "numpy":
            numpy_generator = RandomKey.model_validate(key).numpy_generator()
            result = xp.asarray(
                self._native(mean, backend).rvs(
                    size=shape, random_state=numpy_generator
                ),
                dtype=latent.dtype,
            )
        elif backend == "jax":
            from op_infer.adapters.rng import jax_key  # noqa: PLC0415

            native_key = jax_key(key) if isinstance(key, RandomKey) else key
            result = (
                self._native(mean, backend)
                .sample(native_key, sample_shape=sample_shape)
                .astype(latent.dtype)
            )
        else:
            from op_infer.adapters.rng import torch_generators  # noqa: PLC0415

            if latent.device.type not in {"cpu", "cuda"}:
                msg = "Observation Torch sampling is qualified only on CPU and CUDA"
                raise ValueError(msg)
            if latent.device.type == "cuda" and latent.dtype not in {
                xp.float32,
                xp.float64,
            }:
                msg = "CUDA observation sampling requires float32 or float64"
                raise TypeError(msg)
            generator = (
                torch_generators((key,), device=latent.device)[0]
                if isinstance(key, RandomKey)
                else key
            )
            if getattr(generator, "device", None) != latent.device:
                msg = "Torch observation generator device must match the latent device"
                raise ValueError(msg)
            result = self._torch_sample(mean, generator, shape)
        if self.lower_limit is not None:
            result = xp.maximum(result, xp.full_like(result, self.lower_limit))
        if self.upper_limit is not None:
            result = xp.minimum(result, xp.full_like(result, self.upper_limit))
        return result

    def _torch_sample(self, mean: Any, generator: Any, shape: tuple[int, ...]) -> Any:  # noqa: ANN401
        torch = importlib.import_module("torch")
        mean = mean.expand(shape)
        options = {"dtype": mean.dtype, "device": mean.device, "generator": generator}
        if self.family in {"normal", "lognormal"}:
            result = mean + self.scale * torch.randn(shape, **options)
            return torch.exp(result) if self.family == "lognormal" else result
        if self.family == "poisson":
            return torch.poisson(mean, generator=generator)
        if self.family == "nb2":
            concentration = torch.full_like(mean, self.concentration)
            rate = (
                torch._standard_gamma(concentration, generator=generator)  # noqa: SLF001
                * mean
                / self.concentration
            )
            return torch.poisson(rate, generator=generator)
        if self.family == "beta_binomial":
            a = torch._standard_gamma(  # noqa: SLF001
                mean * self.concentration, generator=generator
            )
            b = torch._standard_gamma(  # noqa: SLF001 -- native primitive takes a local generator
                (1 - mean) * self.concentration, generator=generator
            )
            probability = a / (a + b)
        else:
            probability = (
                (1 - self.specificity)
                + (self.sensitivity + self.specificity - 1) * mean
                if self.family == "bernoulli"
                else mean
            )
        return torch.binomial(
            torch.full_like(mean, 1 if self.family == "bernoulli" else self.trials),
            probability,
            generator=generator,
        )
