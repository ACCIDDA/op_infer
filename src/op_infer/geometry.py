"""Scoped structural records and finite-design Gaussian mean information.

Local numerical rank never establishes a global symmetry or an identifiable
parameter dimension. No coordinate or prior transformation occurs here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

import numpy as np
from array_api_compat import array_namespace
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from op_infer.posterior import ArtifactReference

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "GaussianInformation",
    "InformationContext",
    "InformationLimits",
    "LocalSpectrum",
    "ScalingAssessment",
    "ScalingGenerator",
    "ScalingVerification",
    "gaussian_mean_information",
    "local_spectrum",
]

type _Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
type _Finite = Annotated[float, Field(allow_inf_nan=False)]
type _Count = Annotated[int, Field(ge=0)]


class _Record(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )
    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "Geometry/report schema version must be an integer"
            raise ValueError(msg)
        return value

    @property
    def reference(self) -> ArtifactReference:
        """Identify the complete record, including its semantic type."""
        return ArtifactReference.from_bytes(
            self.model_dump_json().encode(), kind=f"op-infer-{type(self).__name__}-v1"
        )


class ScalingGenerator(_Record):
    """Copy a shared-template scaling action without changing coordinates."""

    name: _Text
    states: tuple[tuple[_Text, _Finite], ...]
    parameters: tuple[tuple[_Text, _Finite], ...]


class ScalingVerification(_Record):
    """Retain producer probes; finite witnesses are not global proofs."""

    seed: _Count
    n_probes: Annotated[int, Field(ge=1)]
    scales: tuple[Annotated[float, Field(gt=0, allow_inf_nan=False)], ...]
    rtol: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    atol: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    max_absolute_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    observables: tuple[str, ...]
    structural_homogeneity_checked: bool


class ScalingAssessment(_Record):
    """Version an existing compiler report, with distinct unavailable statuses.

    Ranks describe the reported shared state/parameter-base action. Initial
    conditions, priors and statistical identifiability are not assessed here.
    """

    source_model: ArtifactReference
    status: Literal["unassessed", "unsupported", "reported"]
    reason: _Text
    parameter_bases: tuple[str, ...] = ()
    generators: tuple[ScalingGenerator, ...] = ()
    action_rank: _Count | None = None
    parameter_action_rank: _Count | None = None
    invariant_monomials: tuple[tuple[tuple[str, int], ...], ...] = ()
    detection_requested: bool = False
    verification: ScalingVerification | None = None

    @model_validator(mode="after")
    def _status(self) -> Self:
        if self.status != "reported":
            if (
                self.action_rank is not None
                or self.parameter_action_rank is not None
                or self.generators
                or self.invariant_monomials
                or self.verification is not None
            ):
                msg = "Unavailable scaling reports cannot claim ranks or actions"
                raise ValueError(msg)
        elif (
            self.action_rank is None
            or self.parameter_action_rank is None
            or self.action_rank != len(self.generators)
            or self.parameter_action_rank
            > min(self.action_rank, len(self.parameter_bases))
        ):
            msg = "Scaling ranks must match the producer's independent basis"
            raise ValueError(msg)
        return self


class InformationContext(_Record):
    """Declare a physical-coordinate, data-only, finite-design target.

    References retain design, parameter point, observation target, numerical
    method and independently supplied qualification evidence. They do not
    authenticate those claims. Units expose the spectrum's coordinate dependence.
    """

    source_model: ArtifactReference
    design: ArtifactReference
    parameter_point: ArtifactReference
    initialization: ArtifactReference
    observation: ArtifactReference
    numerical: ArtifactReference
    qualification: ArtifactReference
    parameter_names: tuple[_Text, ...]
    parameter_units: tuple[_Text, ...]
    observable_names: tuple[_Text, ...]
    coordinates: Literal["physical"] = "physical"
    nuisance_treatment: Literal["none"] = "none"

    @model_validator(mode="after")
    def _layout(self) -> Self:
        if (
            not self.parameter_names
            or not self.observable_names
            or len(self.parameter_names) != len(self.parameter_units)
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or len(set(self.observable_names)) != len(self.observable_names)
        ):
            msg = "Information context requires ordered unique names and complete units"
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class GaussianInformation:
    """Borrow native information arrays without detaching their graphs.

    Matrix entries sum over selected independent measurement cells. Validity and
    observed count are native scalars. Invalid selected inputs or an empty set
    return valid=False and a NaN matrix, never a fake rank.
    """

    context: InformationContext
    matrix: object
    valid: object
    n_observed: object


@dataclass(frozen=True, slots=True)
class InformationLimits:
    """Bound dense numerical work; borrowed input storage is owned by the caller."""

    max_parameters: int = 128
    max_elements: int = 16_777_216

    def __post_init__(self) -> None:
        """Reject invalid bounds, keeping native selected counts within int32.

        Raises:
            ValueError: Limits are not positive integers or exceed count storage.
        """
        if (
            any(
                type(value) is not int or value < 1
                for value in (self.max_parameters, self.max_elements)
            )
            or self.max_elements >= 2**31
        ):
            msg = "Information limits require positive integers within count capacity"
            raise ValueError(msg)


def gaussian_mean_information(
    sensitivities: object,
    standard_deviation: object,
    *,
    context: InformationContext,
    mask: object | None = None,
    limits: InformationLimits | None = None,
) -> GaussianInformation:
    """Compute native ``J.T @ diag(1/sigma**2) @ J`` for measured-mean derivatives.

    Shape is (..., observables, parameters). All leading cells are independent
    measurements. Sigma is scalar or matches the measurement shape; masks are
    native booleans of that exact shape. Missing NaNs are replaced before
    arithmetic. Fixed known variances give conditional Gaussian information;
    parameter-dependent variance needs terms beyond this mean contribution.

    Returns:
        Bounded native dense information and validity, without adding a prior.

    Raises:
        ValueError: Static layout, bounds or precision differs.
        TypeError: Inputs are not native real floating arrays.
    """
    context = InformationContext.model_validate(context)
    if limits is not None and not isinstance(limits, InformationLimits):
        msg = "Information bounds require InformationLimits"
        raise TypeError(msg)
    bounds = limits or InformationLimits()
    values: Any = sensitivities
    sigma: Any = standard_deviation
    if any(
        getattr(value, "dtype", None) is None or getattr(value, "shape", None) is None
        for value in (values, sigma)
    ):
        msg = "Gaussian information inputs must be native arrays"
        raise TypeError(msg)
    xp = array_namespace(values, sigma)
    shape = tuple(values.shape)
    if not xp.isdtype(values.dtype, "real floating") or not xp.isdtype(
        sigma.dtype, "real floating"
    ):
        msg = "Gaussian information requires real floating native inputs"
        raise TypeError(msg)
    if (
        len(shape) < 2
        or shape[-2:] != (len(context.observable_names), len(context.parameter_names))
        or tuple(sigma.shape) not in {(), shape[:-1]}
        or sigma.dtype != values.dtype
    ):
        msg = "Sensitivity/noise layout or precision differs from information context"
        raise ValueError(msg)
    if shape[-1] > bounds.max_parameters or math.prod(shape) > bounds.max_elements:
        msg = "Information exceeds its dense parameter/element bound"
        raise ValueError(msg)
    sigma = xp.broadcast_to(sigma, shape[:-1])
    selected: Any = xp.ones_like(sigma, dtype=xp.bool) if mask is None else mask
    if (
        array_namespace(selected) is not xp
        or selected.dtype != xp.bool
        or tuple(selected.shape) != shape[:-1]
    ):
        msg = "Information mask must be native boolean with the measurement shape"
        raise ValueError(msg)
    sigma_ok = xp.isfinite(sigma) & (sigma > 0)
    valid = xp.all(
        ~selected | (sigma_ok & xp.all(xp.isfinite(values), axis=-1))
    ) & xp.any(selected)
    safe_sigma = xp.where(selected & sigma_ok, sigma, xp.ones_like(sigma))
    safe_values = xp.where(
        xp.expand_dims(selected, axis=-1), values, xp.zeros_like(values)
    )
    weighted = xp.reshape(
        safe_values / xp.expand_dims(safe_sigma, axis=-1), (-1, shape[-1])
    )
    matrix = xp.matrix_transpose(weighted) @ weighted
    valid = valid & xp.all(xp.isfinite(matrix))
    matrix = xp.where(valid, matrix, xp.full_like(matrix, xp.nan))
    return GaussianInformation(
        context, matrix, valid, xp.sum(xp.astype(selected, xp.int32))
    )


class LocalSpectrum(_Record):
    """Host report of local eigenvalues with an explicit rank tolerance.

    This rank at one point/design is not an identifiable dimension, structural
    result, quotient dimension or model penalty.
    """

    context: InformationContext
    eigenvalues: tuple[_Finite, ...]
    eigenvectors: tuple[tuple[_Finite, ...], ...]
    rtol: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    atol: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    local_rank: _Count
    n_observed: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _spectrum(self) -> Self:
        count = len(self.context.parameter_names)
        threshold = self.atol + self.rtol * max(map(abs, self.eigenvalues), default=0.0)
        if (
            len(self.eigenvalues) != count
            or len(self.eigenvectors) != count
            or any(len(row) != count for row in self.eigenvectors)
            or self.rtol + self.atol <= 0
            or tuple(sorted(self.eigenvalues)) != self.eigenvalues
            or not math.isclose(self.threshold, threshold, rel_tol=1e-12, abs_tol=0.0)
            or self.local_rank
            != sum(value > self.threshold for value in self.eigenvalues)
        ):
            msg = "Saved local spectrum layout, threshold or rank is inconsistent"
            raise ValueError(msg)
        return self


def _host(value: object) -> NDArray[np.float64]:
    native: Any = value
    if hasattr(native, "detach"):
        native = native.detach().cpu().numpy()
    elif hasattr(native, "get"):
        native = native.get()
    return np.asarray(native, dtype=np.float64)


def local_spectrum(
    information: GaussianInformation, *, rtol: float, atol: float
) -> LocalSpectrum:
    """Snapshot a valid small matrix and report its local eigenspectrum.

    This explicit host boundary synchronizes/copies native data. Keep the
    original matrix for AD/JIT; no prior or posterior coordinates change.

    Returns:
        Ascending eigenvalues, column eigenvectors, context and threshold.

    Raises:
        ValueError: Tolerances, validity or matrix are inconsistent/nonfinite.
    """
    if (
        any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, Real)
            or not math.isfinite(v)
            or v < 0
            for v in (rtol, atol)
        )
        or rtol + atol <= 0
    ):
        msg = "Local spectrum requires finite nonnegative, nonzero tolerances"
        raise ValueError(msg)
    valid = _host(information.valid)
    if valid.shape != () or valid.item() != 1:
        msg = "Invalid/empty selected data cannot have an information spectrum"
        raise ValueError(msg)
    matrix = _host(information.matrix)
    count = len(information.context.parameter_names)
    if (
        matrix.shape != (count, count)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)
    ):
        msg = "Local information must be finite, ordered and symmetric"
        raise ValueError(msg)
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    threshold = float(atol + rtol * np.max(np.abs(eigenvalues), initial=0.0))
    if float(eigenvalues.min(initial=0.0)) < -threshold:
        msg = "Gaussian information has a materially negative eigenvalue"
        raise ValueError(msg)
    return LocalSpectrum(
        context=information.context,
        eigenvalues=tuple(float(value) for value in eigenvalues),
        eigenvectors=tuple(tuple(float(v) for v in row) for row in eigenvectors),
        rtol=float(rtol),
        atol=float(atol),
        threshold=threshold,
        local_rank=int(np.count_nonzero(eigenvalues > threshold)),
        n_observed=int(_host(information.n_observed).item()),
    )
