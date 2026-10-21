"""Thin optional NumPyro NUTS adapter for native posterior fitting.

Run native NUTS and preserve
grouped samples and per-fit diagnostics. No sampler or adequacy gate is
implemented here. SVI and predictive/report adapters remain planned.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from op_infer.adapters.rng import jax_key
from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    Coordinate,
    Posterior,
    Provenance,
    SampledVariable,
)
from op_infer.rng import RandomKey  # noqa: TC001 -- Pydantic resolves the field type

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax import Array
    from numpy.typing import NDArray
    from numpyro.infer import MCMC

__all__ = [
    "NutsResult",
    "NutsSettings",
    "ParameterDiagnostics",
    "SiteLayout",
    "fit_nuts",
    "numpyro_backend_versions",
]

type _Name = Annotated[str, StringConstraints(pattern=r"\A[A-Za-z_][A-Za-z0-9_]*\z")]

_EXTRA_FIELDS = ("accept_prob", "diverging", "num_steps", "potential_energy")


class _Record(BaseModel):
    """Validate immutable settings before expensive native execution."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )


class SiteLayout(_Record):
    """Declare one complete native output layout.

    Attributes:
        name: Portable native NumPyro site name; no implicit renaming or filtering.
        dims: Event dimensions only, in native array order.
    """

    name: _Name
    dims: tuple[_Name, ...] = ()


class NutsSettings(_Record):
    """Declare a CPU run and its estimator-settings identity.

    Native defaults are tied to recorded versions.

    Attributes:
        schema_version: Adapter format; unknown versions are refused.
        key: Explicit fitting stream address.
        sites: Every native returned sample/deterministic site, with event dimensions.
        coordinates: Ordered event labels; chain/draw labels are generated separately.
        num_warmup: Adaptation iterations per chain, excluded from returned samples.
        num_samples: Retained transitions per chain, with thinning fixed at one.
        num_chains: Sequential CPU chains; a single chain has unavailable R-hat.
        target_accept_prob: Native target acceptance probability.
        max_tree_depth: Native NUTS tree-depth limit.
        dense_mass: Native full versus diagonal mass adaptation.
        enable_x64: Required JAX precision setting, checked without changing globals.

    Examples:
        >>> config = NutsSettings(
        ...     key=RandomKey(seed=2026, stream="fit"),
        ...     sites=(SiteLayout(name="theta"),),
        ... )
        >>> config.reference.kind
        'numpyro-nuts-settings-v1'
    """

    schema_version: Literal[1] = 1
    key: RandomKey
    sites: tuple[SiteLayout, ...]
    coordinates: tuple[Coordinate, ...] = ()
    num_warmup: Annotated[int, Field(ge=1)] = 500
    num_samples: Annotated[int, Field(ge=4)] = 500
    num_chains: Annotated[int, Field(ge=1)] = 2
    target_accept_prob: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.8
    max_tree_depth: Annotated[int, Field(ge=1)] = 10
    dense_mass: bool = False
    enable_x64: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "schema_version must be an integer"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _complete_layout(self) -> Self:
        names = [site.name for site in self.sites]
        coords = [coordinate.name for coordinate in self.coordinates]
        used = {dim for site in self.sites for dim in site.dims}
        if not names or len(set(names)) != len(names):
            msg = "Declare nonempty unique native sample sites"
            raise ValueError(msg)
        if (
            len(set(coords)) != len(coords)
            or set(coords) & {"chain", "draw", "seed", *_EXTRA_FIELDS}
            or set(names) & {*coords, "chain", "draw"}
            or used != set(coords)
            or any(len(set(site.dims)) != len(site.dims) for site in self.sites)
        ):
            msg = "Event coordinates must be unique, fully used and distinct from sites"
            raise ValueError(msg)
        return self

    def manifest_json(self) -> str:
        """Serialize the exact resolved settings payload for fit artifacts.

        Returns:
            Canonical JSON containing inputs and the fixed native adapter policy.
            Store these bytes with the shared estimator-settings reference.
        """
        settings = NutsSettings.model_validate(self)
        payload = {
            "settings": settings.model_dump(mode="json"),
            "adapter": "numpyro-nuts-cpu-v1",
            "chain_method": "sequential",
            "thinning": 1,
            "progress_bar": False,
            "initialization": "init_to_uniform(radius=2.0)",
            "extra_fields": _EXTRA_FIELDS,
            "remaining_defaults": "recorded numpyro version",
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def reference(self) -> ArtifactReference:
        """Hash the exact fit manifest using the shared artifact reference.

        The reference alone does not store/authenticate payload or model identities.
        """
        return ArtifactReference.from_bytes(
            self.manifest_json().encode(), kind="numpyro-nuts-settings-v1"
        )


def numpyro_backend_versions() -> tuple[BackendVersion, ...]:
    """Collect exact fit dependencies for shared fit provenance before running.

    Returns:
        Sorted dependency/Python versions; no backend is imported by this helper.

    Raises:
        ImportError: A required optional distribution is not installed.
    """
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "array-api-compat",
                "jax",
                "jaxlib",
                "numpy",
                "numpyro",
                "op_infer",
                "pydantic",
                "scipy",
            )
        }
    except importlib.metadata.PackageNotFoundError as exc:
        msg = "Install op_infer[numpyro] to use the NumPyro NUTS adapter"
        raise ImportError(msg) from exc
    versions["python"] = platform.python_version()
    return tuple(
        BackendVersion(name=name, version=version)
        for name, version in sorted(versions.items())
    )


@dataclass(frozen=True, slots=True, eq=False)
class ParameterDiagnostics:
    """Retain NumPyro's raw host diagnostics without asserting sampler adequacy.

    Attributes:
        name: Same native site as the posterior variable.
        dims: Same ordered event dimensions, without chain/draw axes.
        ess: Native NumPyro ESS, with nonfinite entries retained (not bulk/tail ESS).
        split_r_hat: Classic NumPyro split R-hat; absent for a single original chain.
            This is not rank-normalized R-hat. Undefined values remain nonfinite.
    """

    name: str
    dims: tuple[str, ...]
    ess: NDArray[np.float64]
    split_r_hat: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True, eq=False)
class NutsResult:
    """Preserve the common posterior and native fit.

    Attributes:
        posterior: Native JAX samples/statistics, with complete shared provenance.
        diagnostics: Host diagnostic arrays; no nonfinite site is discarded.
        divergences_per_chain: Counts over every retained transition in each chain.
        nonfinite_values: Count of nonfinite scalar entries over all returned sites.
        native: Original MCMC instance; advanced native inspection remains available.
    """

    posterior: Posterior[Array]
    diagnostics: tuple[ParameterDiagnostics, ...]
    divergences_per_chain: tuple[int, ...]
    nonfinite_values: int
    native: MCMC

    @property
    def adequacy_status(self) -> Literal["not_assessed"]:
        """Keep raw diagnostics distinct from prespecified adequacy gates."""
        return "not_assessed"


def _summaries(posterior: Posterior[Array]) -> tuple[ParameterDiagnostics, ...]:
    """Use native diagnostics on explicit host views, retaining undefined outcomes."""
    diagnostics = importlib.import_module("numpyro.diagnostics")
    result = []
    for variable in posterior.draws:
        values = np.asarray(variable.values)
        # Constant sites legitimately yield undefined diagnostics, retained below.
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = np.asarray(
                diagnostics.effective_sample_size(values), dtype=np.float64
            )
            r_hat = (
                np.asarray(diagnostics.split_gelman_rubin(values), dtype=np.float64)
                if values.shape[0] >= 2
                else None
            )
        result.append(ParameterDiagnostics(variable.name, variable.dims, ess, r_hat))
    return tuple(result)


def fit_nuts(
    model: Callable[..., object],
    *,
    settings: NutsSettings,
    provenance: Provenance,
    model_args: tuple[object, ...] = (),
    model_kwargs: Mapping[str, object] | None = None,
) -> NutsResult:
    """Run native NUTS and retain the fitted posterior.

    The caller supplies matching prior/model/data/target identities; their
    scientific truth is not inferred from hashes. All native returned sites
    need declared layouts. Models receive dynamic arguments without closure
    conversion. Failures propagate; no posterior or successful gate is fabricated.

    Args:
        model: Trusted native NumPyro model callable.
        settings: Immutable CPU run, RNG and complete output layout.
        provenance: Shared identity with matching settings and installed versions.
        model_args: Dynamic positional model inputs, owned by the caller.
        model_kwargs: Dynamic named inputs; native run-control names are reserved.

    Returns:
        Common native posterior, raw diagnostics and original NumPyro fit.

    Raises:
        ImportError: The optional NumPyro/JAX dependencies are unavailable.
        TypeError: The model or positional input collection is invalid.
        ValueError: Settings/provenance/layout/precision/backend are incompatible,
            an augmented target is requested, or native fitting fails.
        RuntimeError: Native initialization or execution fails.
    """
    settings = NutsSettings.model_validate(settings)
    provenance = Provenance.model_validate(provenance)
    if not callable(model) or not isinstance(model_args, tuple):
        msg = "Supply a native model callable and a tuple of positional model inputs"
        raise TypeError(msg)
    kwargs = dict(model_kwargs or {})
    if set(kwargs) & {"extra_fields", "init_params", "rng_key"}:
        msg = "Model kwargs cannot override native run controls"
        raise ValueError(msg)
    if provenance.estimator_settings != settings.reference:
        msg = "Estimator settings provenance mismatch"
        raise ValueError(msg)
    if provenance.target.kind == "augmented_pseudo_marginal":
        msg = (
            "Augmented pseudo-marginal targets require the separately qualified "
            "#16 adapter"
        )
        raise ValueError(msg)
    declared = {item.name: item.version for item in provenance.backend_versions}
    if any(
        declared.get(item.name) != item.version for item in numpyro_backend_versions()
    ):
        msg = "Backend versions provenance mismatch"
        raise ValueError(msg)
    try:
        jax = importlib.import_module("jax")
        infer = importlib.import_module("numpyro.infer")
    except ImportError as exc:
        msg = "Install op_infer[numpyro] to use the NumPyro NUTS adapter"
        raise ImportError(msg) from exc
    if jax.default_backend() != "cpu":
        msg = "This NUTS adapter is qualified for the default CPU backend only"
        raise ValueError(msg)
    if bool(jax.config.jax_enable_x64) != settings.enable_x64:
        msg = "JAX precision differs from the recorded enable_x64 setting"
        raise ValueError(msg)
    kernel = infer.NUTS(
        model,
        target_accept_prob=settings.target_accept_prob,
        max_tree_depth=settings.max_tree_depth,
        dense_mass=settings.dense_mass,
        init_strategy=infer.init_to_uniform(radius=2.0),
    )
    native = cast(
        "MCMC",
        infer.MCMC(
            kernel,
            num_warmup=settings.num_warmup,
            num_samples=settings.num_samples,
            num_chains=settings.num_chains,
            chain_method="sequential",
            thinning=1,
            progress_bar=False,
        ),
    )
    native.run(jax_key(settings.key), *model_args, extra_fields=_EXTRA_FIELDS, **kwargs)
    samples = cast("dict[str, Array]", native.get_samples(group_by_chain=True))
    stats = cast("dict[str, Array]", native.get_extra_fields(group_by_chain=True))
    if set(samples) != {site.name for site in settings.sites}:
        msg = "Declare every returned native site; no site is silently omitted"
        raise ValueError(msg)
    posterior = Posterior(
        draws=tuple(
            SampledVariable(site.name, samples[site.name], site.dims)
            for site in settings.sites
        ),
        coordinates=(
            Coordinate(name="chain", labels=tuple(range(settings.num_chains))),
            Coordinate(name="draw", labels=tuple(range(settings.num_samples))),
            *settings.coordinates,
        ),
        provenance=provenance,
        sample_stats=tuple(
            SampledVariable(name, stats[name]) for name in _EXTRA_FIELDS
        ),
    )
    summaries = _summaries(posterior)
    return NutsResult(
        posterior=posterior,
        diagnostics=summaries,
        divergences_per_chain=tuple(
            int(np.count_nonzero(chain)) for chain in np.asarray(stats["diverging"])
        ),
        nonfinite_values=sum(
            int(np.count_nonzero(~np.isfinite(np.asarray(variable.values))))
            for variable in posterior.draws
        ),
        native=native,
    )
