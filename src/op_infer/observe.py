"""Native iid observation laws shared by simulation and likelihood evaluation.

Latent means/log-locations/probabilities remain separate from random measurements.
Array operations preserve NumPy, JAX or Torch execution; native distributions own
sampling and density kernels. Correlated observations are not silently iid.
"""

from __future__ import annotations

import importlib
import json
from typing import Annotated, Any, Literal, Protocol, Self

from array_api_compat import array_namespace
from pydantic import Field, model_validator

from op_infer.posterior import ArtifactReference
from op_infer.priors import _backend, _Record, _validate_sample_shape
from op_infer.rng import RandomKey

__all__ = ["IIDObservation", "ObservationModel"]

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
        local CPU Generator; RandomKey creates one without seeding global state.
        This first qualification is CPU only. Returned counts use latent float
        dtype, matching native Torch count distributions.
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

            generator = (
                torch_generators((key,))[0] if isinstance(key, RandomKey) else key
            )
            if latent.device.type != "cpu":
                msg = "Observation Torch sampling is qualified only on CPU"
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
