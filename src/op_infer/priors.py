"""Immutable scalar distribution declarations and independently addressed priors.

Native lowerings wrap SciPy, NumPyro and Torch distributions. The first catalog
is independent Normal, LogNormal, Uniform, Gamma(shape/rate) and Beta; hierarchy,
correlated blocks and automatic coordinate changes are explicitly outside it.
"""

from __future__ import annotations

import importlib
import json
import math
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

import numpy as np
from array_api_compat import array_namespace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from op_infer.posterior import ArtifactReference
from op_infer.rng import RandomKey

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "Beta",
    "Gamma",
    "JointPrior",
    "LogNormal",
    "Normal",
    "PriorParameter",
    "Uniform",
]

type _Finite = Annotated[float, Field(allow_inf_nan=False)]
type _Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type _Name = Annotated[str, StringConstraints(pattern=r"\A[A-Za-z_][A-Za-z0-9_]*\z")]


class _Record(BaseModel):
    """Reject mutable metadata and implicit integer/boolean coercion."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )


class _Distribution(_Record):
    """Delegate scalar-law lowerings to established distribution libraries."""

    def scipy(self) -> Any:  # noqa: ANN401
        """Return a frozen native SciPy scalar distribution."""
        stats = importlib.import_module("scipy.stats")
        data = self.model_dump()
        family = data.pop("family")
        if family == "gamma":
            return stats.gamma(a=data["concentration"], scale=1 / data["rate"])
        if family == "beta":
            return stats.beta(a=data["concentration1"], b=data["concentration0"])
        if family == "uniform":
            return stats.uniform(loc=data["low"], scale=data["high"] - data["low"])
        if family == "lognormal":
            return stats.lognorm(s=data["scale"], scale=math.exp(data["loc"]))
        return stats.norm(**data)

    def numpyro(self) -> Any:  # noqa: ANN401
        """Return a native NumPyro law; the optional dependency is required."""
        distributions = importlib.import_module("numpyro.distributions")
        data = self.model_dump()
        family = data.pop("family")
        names = {
            "normal": "Normal",
            "lognormal": "LogNormal",
            "uniform": "Uniform",
            "gamma": "Gamma",
            "beta": "Beta",
        }
        return getattr(distributions, names[family])(**data)

    def torch(
        self,
        *,
        dtype: Any = None,  # noqa: ANN401
        device: Any = "cpu",  # noqa: ANN401
        validate_args: bool = True,
    ) -> Any:  # noqa: ANN401
        """Return a native Torch law with explicit tensor precision and device."""
        torch = importlib.import_module("torch")
        data = self.model_dump()
        family = data.pop("family")
        names = {
            "normal": "Normal",
            "lognormal": "LogNormal",
            "uniform": "Uniform",
            "gamma": "Gamma",
            "beta": "Beta",
        }
        parameters = {
            name: torch.as_tensor(value, dtype=dtype, device=device)
            for name, value in data.items()
        }
        return getattr(torch.distributions, names[family])(
            **parameters, validate_args=validate_args
        )

    @property
    def reference(self) -> ArtifactReference:
        """Identify canonical declaration bytes, separate from a sampling stream."""
        return ArtifactReference.from_bytes(
            self.model_dump_json().encode(), kind="scalar-prior-v1"
        )


class Normal(_Distribution):
    """Normal law with location and positive standard deviation."""

    family: Literal["normal"] = "normal"
    loc: _Finite = 0.0
    scale: _Positive = 1.0


class LogNormal(_Distribution):
    """Exp(Normal(loc, scale)); parameters are on the logarithmic scale."""

    family: Literal["lognormal"] = "lognormal"
    loc: _Finite = 0.0
    scale: _Positive = 1.0


class Uniform(_Distribution):
    """Uniform law on low <= x < high, consistently across density adapters."""

    family: Literal["uniform"] = "uniform"
    low: _Finite = 0.0
    high: _Finite = 1.0

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low >= self.high:
            msg = "Uniform requires low < high"
            raise ValueError(msg)
        return self


class Gamma(_Distribution):
    """Gamma shape/rate law, with mean concentration/rate."""

    family: Literal["gamma"] = "gamma"
    concentration: _Positive
    rate: _Positive


class Beta(_Distribution):
    """Beta law with mean concentration1/(concentration1 + concentration0)."""

    family: Literal["beta"] = "beta"
    concentration1: _Positive
    concentration0: _Positive


type Distribution = Annotated[
    Normal | LogNormal | Uniform | Gamma | Beta, Field(discriminator="family")
]


class PriorParameter(_Record):
    """Name an independent scalar or fixed-shaped array of iid coordinates."""

    name: _Name
    distribution: Distribution
    shape: tuple[Annotated[int, Field(ge=1)], ...] = ()


class JointPrior(_Record):
    """Ordered independent declarations, their stable identity and native sites.

    Shapes are event axes; ``sample_shape`` adds leading independent draws.
    Sampling never retries or drops a draw. Native distributions own their
    sampling APIs; no cross-backend bitwise replay is claimed.

    Examples:
        >>> prior = JointPrior(
        ...     parameters=(
        ...         PriorParameter(
        ...             name="rate", distribution=Gamma(concentration=2.0, rate=3.0)
        ...         ),
        ...     )
        ... )
        >>> key = RandomKey(seed=2026, stream="parameters")
        >>> prior.sample(key)["rate"] == prior.sample(key)["rate"]
        np.True_
    """

    schema_version: Literal[1] = 1
    parameters: tuple[PriorParameter, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "schema_version must be an integer"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names = self.parameter_names
        if not names or len(set(names)) != len(names):
            msg = "Declare nonempty unique prior parameter names"
            raise ValueError(msg)
        return self

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the semantic parameter order."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def reference(self) -> ArtifactReference:
        """Identify the ordered declaration with canonical JSON."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return ArtifactReference.from_bytes(
            payload.encode(), kind="independent-joint-prior-v1"
        )

    def sample(
        self, key: RandomKey, *, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Draw NumPy arrays using per-name streams and native SciPy sampling.

        Adding/reordering another parameter does not change an existing name's
        draws. Changes to a distribution or shape still change prior identity.
        """
        _validate_sample_shape(sample_shape)
        RandomKey.model_validate(key)
        result = {}
        for parameter in self.parameters:
            address = key.model_copy(
                update={"stream": f"{key.stream}/prior/{parameter.name}"}
            )
            result[parameter.name] = np.asarray(
                parameter.distribution.scipy().rvs(
                    size=sample_shape + parameter.shape,
                    random_state=address.numpy_generator(),
                )
            )
        return result

    def log_prob(self, values: Mapping[str, Any]) -> Any:  # noqa: ANN401
        """Sum event axes while retaining leading native batch axes and graphs.

        NumPy/SciPy, JAX/NumPyro and Torch are the qualified namespaces. A mixed
        namespace, wrong names or mismatched event/batch shapes fails explicitly.
        """
        if set(values) != set(self.parameter_names):
            msg = "Prior values must contain exactly the declared names"
            raise ValueError(msg)
        xp = array_namespace(*(values[name] for name in self.parameter_names))
        backend = _backend(xp)
        result = None
        batch_shape = None
        for parameter in self.parameters:
            value = values[parameter.name]
            if not xp.isdtype(value.dtype, "real floating"):
                msg = "Prior coordinates must have real floating dtype"
                raise TypeError(msg)
            event_ndim = len(parameter.shape)
            if event_ndim and tuple(value.shape[-event_ndim:]) != parameter.shape:
                msg = "Prior event shape differs from its declaration"
                raise ValueError(msg)
            batch = (
                tuple(value.shape[:-event_ndim]) if event_ndim else tuple(value.shape)
            )
            if batch_shape is not None and batch != batch_shape:
                msg = "Prior batch shapes must match exactly"
                raise ValueError(msg)
            batch_shape = batch
            density = _log_density(parameter.distribution, value, xp, backend)
            if event_ndim:
                density = xp.sum(
                    density, axis=tuple(range(density.ndim - event_ndim, density.ndim))
                )
            result = density if result is None else result + density
        return result

    def numpyro_sites(self) -> dict[str, Any]:
        """Return named native laws, with shaped coordinates declared as events."""
        return {
            p.name: p.distribution.numpyro().expand(p.shape).to_event(len(p.shape))
            for p in self.parameters
        }

    def torch_sites(self, *, dtype: Any = None, device: Any = "cpu") -> dict[str, Any]:  # noqa: ANN401
        """Return native named laws, with shaped coordinates declared as events."""
        distributions = importlib.import_module("torch.distributions")
        return {
            p.name: distributions.Independent(
                p.distribution.torch(dtype=dtype, device=device).expand(p.shape),
                len(p.shape),
            )
            for p in self.parameters
        }

    def torch_normal_vector(self, *, dtype: Any = None, device: Any = "cpu") -> Any:  # noqa: ANN401
        """Lower scalar Normal sites to one real-vector law for existing sbi NPE.

        Other supports and automatic coordinate transforms are refused. Parameter
        order is the declared order and is not inferred from a training bank.
        """
        torch = importlib.import_module("torch")
        if any(
            not isinstance(p.distribution, Normal) or p.shape for p in self.parameters
        ):
            msg = "The NPE vector lowering requires scalar Normal coordinates"
            raise ValueError(msg)
        normals = [
            p.distribution
            for p in self.parameters
            if isinstance(p.distribution, Normal)
        ]
        loc = torch.tensor([p.loc for p in normals], dtype=dtype, device=device)
        scale = torch.tensor([p.scale for p in normals], dtype=dtype, device=device)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), 1
        )


def _backend(xp: Any) -> Literal["numpy", "jax", "torch"]:  # noqa: ANN401
    name = xp.__name__
    if "numpy" in name and "jax" not in name:
        return "numpy"
    if "jax" in name:
        return "jax"
    if "torch" in name:
        return "torch"
    msg = "Only NumPy, JAX and Torch statistical lowerings are qualified"
    raise TypeError(msg)


def _validate_sample_shape(shape: tuple[int, ...]) -> None:
    if not isinstance(shape, tuple) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        msg = "sample_shape must be a tuple of nonnegative integers"
        raise ValueError(msg)


def _log_density(law: Distribution, value: Any, xp: Any, backend: str) -> Any:  # noqa: ANN401
    """Retain native kernels with explicit support and constant-density graphs."""
    if backend == "numpy":
        density = law.scipy().logpdf(value)
        if isinstance(law, Uniform):
            density = xp.where(value < law.high, density, -xp.inf)
    elif backend == "jax":
        native = law.numpyro()
        support = native.support(value)
        safe = xp.where(support, value, native.mean)
        density = xp.where(support, native.log_prob(safe), -xp.inf)
    else:
        native = law.torch(dtype=value.dtype, device=value.device, validate_args=False)
        support = native.support.check(value)
        safe = xp.where(support, value, native.mean)
        density = xp.where(support, native.log_prob(safe), -xp.inf)
    if isinstance(law, Uniform):
        valid = (value >= law.low) & (value < law.high)
        safe = xp.where(valid, value, xp.zeros_like(value))
        density = xp.where(valid, density + 0.0 * safe, -xp.inf)
    return xp.where(xp.isfinite(value), density, -xp.inf)
