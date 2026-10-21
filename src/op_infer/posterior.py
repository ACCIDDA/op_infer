"""Posterior containers and provenance.

Status: native posterior containers and immutable provenance are implemented.
Optional ArviZ snapshots live in ``op_infer.adapters.arviz``; shard merging remains
planned.

Scope: Posterior (draws per chain or seed, sample statistics, per-unit log-likelihoods,
estimator stage) converting to and from ArviZ 1.x and netCDF; Provenance (prior hash,
design hash, simulator version, estimator settings, backend versions, commit); guarded
loading and shard merging that refuse mismatches, gaps and overlaps.


Required target identity and approximation/auxiliary-state provenance are specified in
``docs/design/likelihood-targets.md``. Future
loaders must check expected identities and preserve unknown legacy claims as unknown.
Posterior snapshots and fitting adapters use the shared references below. Numerical
qualification
is evidence for its recorded scope, not automatic proof or a universal error bound.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from array_api_compat import array_namespace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "ArtifactReference",
    "BackendVersion",
    "Coordinate",
    "DerivativeContract",
    "ErrorComponent",
    "LikelihoodTargetReference",
    "NumericalExecution",
    "Posterior",
    "Provenance",
    "Qualification",
    "SampledVariable",
]

type _Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
type _Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
type _ErrorKind = Literal[
    "time",
    "space_quadrature",
    "nonlinear_solve",
    "linear_solve",
    "unit_integration",
    "monte_carlo",
    "sampler",
]

_ERROR_KINDS = frozenset(
    {
        "time",
        "space_quadrature",
        "nonlinear_solve",
        "linear_solve",
        "unit_integration",
        "monte_carlo",
        "sampler",
    }
)


class _Record(BaseModel):
    """Validate immutable scalar metadata, with no mutable nested containers."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )


class ArtifactReference(_Record):
    """Identify immutable artifact bytes.

    Attributes:
        kind: Semantic artifact type, including its format/version where applicable.
        digest: SHA-256 of the exact referenced bytes. Payload availability, semantics
            and authenticity require separate verification by the consumer.
    """

    kind: _Text
    digest: _Digest

    @classmethod
    def from_bytes(cls, payload: bytes, *, kind: str) -> Self:
        """Hash exact artifact bytes without reading external data.

        Args:
            payload: Immutable serialized content; canonicalization is the writer's
                responsibility, including dtype, shape and ordering for arrays.
            kind: Semantic artifact type.

        Returns:
            A content reference, without storing or authenticating the payload.

        Examples:
            >>> ref = ArtifactReference.from_bytes(
            ...     b'{"method":"rk4"}', kind="solver-v1"
            ... )
            >>> ref.digest.startswith("sha256:")
            True
        """
        return cls(kind=kind, digest=f"sha256:{hashlib.sha256(payload).hexdigest()}")


class Qualification(_Record):
    """Record scoped evidence status without certifying a claim.

    Attributes:
        status: Explicit qualified, failed or unqualified outcome.
        scope: Referenced domain, physical quantities, transforms and tested setup.
        reason: Human-readable outcome or limitation.
        evidence: Immutable qualification artifacts; required for a qualified claim.
    """

    status: Literal["qualified", "failed", "unqualified"]
    scope: ArtifactReference
    reason: _Text
    evidence: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def _require_evidence(self) -> Self:
        if self.status == "qualified" and not self.evidence:
            msg = "Qualified claims require scoped evidence references"
            raise ValueError(msg)
        return self


class ErrorComponent(_Record):
    """Keep error controls distinct with settings distinct from achieved error bounds.

    Attributes:
        kind: One of seven separately recorded numerical/statistical components.
        status: Settings specified, not applicable, or explicitly unqualified.
        reason: Applicability or qualification limitation.
        controls: Settings artifact; required when specified, absent when inapplicable.
        evidence: Optional achieved-error/uncertainty evidence with its own scope.
    """

    kind: _ErrorKind
    status: Literal["specified", "not_applicable", "unqualified"]
    reason: _Text
    controls: ArtifactReference | None
    evidence: ArtifactReference | None

    @model_validator(mode="after")
    def _check_controls(self) -> Self:
        if self.status == "specified" and self.controls is None:
            msg = "Specified error components require control settings"
            raise ValueError(msg)
        if self.status == "not_applicable" and self.controls is not None:
            msg = "Inapplicable error components cannot declare control settings"
            raise ValueError(msg)
        return self


class DerivativeContract(_Record):
    """Identify the differentiated map with explicit graph-portability limits.

    Attributes:
        mode: Derivative method, or none for a value-only evaluation.
        parameters: Ordered differentiation coordinates, including their transforms.
        adaptive_path: Whether replay is fixed or controller choices differentiated.
        graph_scope: Backend, transforms, geometry treatment and transfer boundaries.
        qualification: Independent derivative checks and their supported domain.
    """

    mode: Literal[
        "none",
        "finite_difference",
        "discrete_autodiff",
        "continuous_sensitivity",
        "implicit",
        "custom",
    ]
    parameters: tuple[_Text, ...]
    adaptive_path: Literal[
        "not_applicable", "frozen_replay", "differentiate_controller", "unqualified"
    ]
    graph_scope: ArtifactReference
    qualification: Qualification

    @model_validator(mode="after")
    def _check_parameters(self) -> Self:
        if (self.mode == "none") != (not self.parameters):
            msg = "Value-only modes need no coordinates; other modes need coordinates"
            raise ValueError(msg)
        if len(set(self.parameters)) != len(self.parameters):
            msg = "Derivative coordinates must be unique"
            raise ValueError(msg)
        return self


class NumericalExecution(_Record):
    """Reference the evaluated numerical behavior.

    Attributes:
        solver: Method, tolerance, precision, solver/library version and settings.
        time_mesh: Time discretization and output coordinates.
        spatial_mesh: Spatial/quadrature coordinates and measures, or explicit absence.
        topology: Fixed connectivity identity; required with a spatial mesh.
        geometry: Frozen geometry or learnable coordinates with fixed topology.
        replay: Accepted replay artifact, or explicit absence.
        mesh_policy: Refresh/reuse rules defining one consistent evaluated likelihood.
        observation: Physical observation definition, evaluation and aggregation.
        field_prior: State representation, basis, measures and prior interpretation.
        derivative: Differentiated map and qualification scope.
        errors: Exactly one record for each of the seven error/uncertainty components.
        qualification: Refinement/reference results for this execution's scope.
    """

    solver: ArtifactReference
    time_mesh: ArtifactReference
    spatial_mesh: ArtifactReference | None
    topology: ArtifactReference | None
    geometry: Literal["frozen", "learnable_fixed_topology"]
    replay: ArtifactReference | None
    mesh_policy: ArtifactReference
    observation: ArtifactReference
    field_prior: ArtifactReference
    derivative: DerivativeContract
    errors: tuple[ErrorComponent, ...]
    qualification: Qualification

    @model_validator(mode="after")
    def _check_mesh_and_errors(self) -> Self:
        if (self.spatial_mesh is None) != (self.topology is None):
            msg = "Spatial meshes and their fixed topology must be recorded together"
            raise ValueError(msg)
        if self.geometry == "learnable_fixed_topology" and self.topology is None:
            msg = "Learnable geometry requires a spatial mesh and fixed topology"
            raise ValueError(msg)
        kinds = [component.kind for component in self.errors]
        if len(kinds) != len(_ERROR_KINDS) or set(kinds) != _ERROR_KINDS:
            msg = "Record each of the seven error components exactly once"
            raise ValueError(msg)
        if self.replay is None and self.derivative.adaptive_path == "frozen_replay":
            msg = "Frozen-replay derivatives require a replay artifact"
            raise ValueError(msg)
        return self


class LikelihoodTargetReference(_Record):
    """Preserve the intended and evaluated likelihood-target contract.

    Attributes:
        kind: Evaluated target category, not an exactness certificate.
        intended: Prior/model/data/unit/measure identities of the intended target.
        evaluated: Full approximation/settings/support/randomness identity of the
            evaluated target.
        qualification: Exactness/approximation evidence and limitations for its scope.
    """

    kind: Literal[
        "analytic",
        "deterministic_approximation",
        "frozen_random_approximation",
        "augmented_pseudo_marginal",
    ]
    intended: ArtifactReference
    evaluated: ArtifactReference
    qualification: Qualification


class BackendVersion(_Record):
    """Record one numerical/backend dependency."""

    name: _Text
    version: _Text


class Provenance(_Record):
    """Versioned shared artifact identity.

    Attributes:
        schema_version: Required format version; unknown/missing versions are refused.
        prior: Prior definition and parameter interpretation.
        design: Physical design and data/unit manifest.
        simulator: Simulator implementation/version and observation definition.
        target: Intended and evaluated likelihood identity and qualification.
        numerical: Numerical execution, or explicit absence for an analytic calculation.
        estimator_stage: Distinct training, fitting, calibration or reporting stage.
        estimator_settings: Full estimator settings and randomness policy.
        backend_versions: Unique dependency names and exact versions in canonical order.
        commit: Code content identity, normally an immutable commit/source manifest.
        structural_model: Schema-2 compiler or explicitly declared callable content
            reference, separate from runtime/prior/design/estimator identity.
    """

    schema_version: Literal[1, 2]
    prior: ArtifactReference
    design: ArtifactReference
    simulator: ArtifactReference
    target: LikelihoodTargetReference
    numerical: NumericalExecution | None
    estimator_stage: _Text
    estimator_settings: ArtifactReference
    backend_versions: tuple[BackendVersion, ...]
    commit: ArtifactReference
    structural_model: ArtifactReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "Schema version must be an integer, not a boolean or float"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check_versions(self) -> Self:
        names = tuple(backend.name for backend in self.backend_versions)
        if not names or names != tuple(sorted(set(names))):
            msg = "Record nonempty, unique backend versions sorted by name"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_structural_model(self) -> Self:
        if self.schema_version == 1 and self.structural_model is not None:
            msg = "Structural model references require provenance schema 2"
            raise ValueError(msg)
        if self.schema_version == 2 and self.structural_model is None:
            msg = "Provenance schema 2 requires an explicit structural model reference"
            raise ValueError(msg)
        if self.structural_model is not None:
            kind = self.structural_model.kind
            if (
                kind.startswith("op-system-structural-model-")
                and kind != "op-system-structural-model-v1"
            ):
                msg = "Unsupported op_system structural model identity schema"
                raise ValueError(msg)
        return self

    def with_structural_model(self, model: ArtifactReference) -> Self:
        """Explicitly create schema-2 provenance from an independently supplied model.

        This does not upgrade old artifacts or verify that their producing code
        used the supplied model. Consumers must supply expectations independently.
        All native fitting/bank/posterior records reuse this
        provenance rather than another artifact identity hierarchy.

        Args:
            model: Compiler manifest reference or a declared callable model manifest.

        Returns:
            Validated schema-2 provenance, leaving this record unchanged.
        """
        return type(self).model_validate(
            {**self.model_dump(), "schema_version": 2, "structural_model": model}
        )

    @property
    def identity(self) -> str:
        """Return deterministic complete-record identity."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def require_match(self, expected: Self) -> None:
        """Refuse behavior/provenance changes.

        Args:
            expected: Complete expected provenance, including qualification evidence.

        Raises:
            ValueError: Any required field differs. Payload authenticity and semantic
                compatibility require separate verification; equality is conservative.
        """
        if self != expected:
            msg = (
                "Provenance mismatch: artifact differs from expected behavior identity"
            )
            raise ValueError(msg)

    @classmethod
    def load_json(cls, payload: str, *, expected: Self) -> Self:
        """Validate and guard provenance JSON before artifact reuse.

        Args:
            payload: Serialized provenance; this loader does not load referenced data.
            expected: Required identity, supplied independently by the caller.

        Returns:
            Validated immutable provenance matching the caller's expectation.

        Raises:
            ValueError: Invalid, missing, unknown-schema or mismatched provenance.
        """
        result = cls.model_validate_json(payload)
        result.require_match(expected)
        return result


def _portable_name(name: str) -> None:
    """Check portable variable/dimension identifiers.

    Raises:
        ValueError: A name is not an ASCII identifier.
    """
    if not isinstance(name, str) or not name.isascii() or not name.isidentifier():
        msg = "Variable and dimension names must be nonempty ASCII identifiers"
        raise ValueError(msg)


class Coordinate(_Record):
    """Immutable ordered coordinate labels , not a physical-unit claim.

    Attributes:
        name: Portable dimension name.
        labels: Nonempty unique, homogeneous strings, signed 64-bit integers or
            finite floats. Order is preserved; no automatic sorting/alignment.

    Examples:
        >>> Coordinate(name="unit", labels=("u3", "u1")).labels
        ('u3', 'u1')
    """

    name: _Text
    labels: tuple[int | float | str, ...]

    @model_validator(mode="after")
    def _check_labels(self) -> Self:
        _portable_name(self.name)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            msg = "Coordinate labels must be nonempty and unique"
            raise ValueError(msg)
        if len({type(label) for label in self.labels}) != 1:
            msg = "Coordinate labels must have one scalar type"
            raise ValueError(msg)
        for label in self.labels:
            if (
                (isinstance(label, float) and not math.isfinite(label))
                or (type(label) is int and not -(2**63) <= label < 2**63)
                or (isinstance(label, str) and (not label or "\0" in label))
                or isinstance(label, bool)
            ):
                msg = "Coordinate labels must be finite, portable scalar identities"
                raise ValueError(msg)
        return self


def _array_shape(values: object) -> tuple[int, ...]:
    """Validate a real native array without copying or inspecting its values.

    Returns:
        Concrete array shape, leaving device and differentiation graph untouched.

    Raises:
        TypeError: The input is not a supported real/bool Array-API array.
    """
    xp = array_namespace(values)
    shape = getattr(values, "shape", None)
    if (
        not isinstance(shape, tuple)
        or any(type(size) is not int for size in shape)
        or not xp.isdtype(
            getattr(values, "dtype", None), ("real floating", "integral", "bool")
        )
    ):
        msg = "Samples require a real numeric/bool native array with a concrete shape"
        raise TypeError(msg)
    return shape


@dataclass(frozen=True, slots=True, eq=False)
class SampledVariable[ArrayT]:
    """Named native values without host transfer or graph detachment.

    Attributes:
        name: Portable name, unique within its group.
        values: Native array with member and draw axes first, then event dimensions.
            Arrays retain backend mutability; frozen metadata is not a payload hash.
        dims: Ordered event dimensions only; sample dimensions belong to the posterior.

    Examples:
        >>> import numpy as np
        >>> variable = SampledVariable("theta", np.ones((2, 3)))
        >>> variable.values.shape, variable.dims
        ((2, 3), ())
    """

    name: str
    values: ArrayT
    dims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check names and rank without evaluating native data.

        Raises:
            TypeError: Dimensions are mutable or values are not native real arrays.
            ValueError: Names, event dimensions or array rank are invalid.
        """
        _portable_name(self.name)
        if not isinstance(self.dims, tuple):
            msg = "Event dimensions must be an immutable tuple"
            raise TypeError(msg)
        for name in self.dims:
            _portable_name(name)
        if len(set(self.dims)) != len(self.dims) or len(
            _array_shape(self.values)
        ) != 2 + len(self.dims):
            msg = "Values need two sample axes and unique declared event dimensions"
            raise ValueError(msg)


def _group_dimensions[ArrayT](
    group: tuple[SampledVariable[ArrayT], ...],
    coords: dict[str, tuple[int | float | str, ...]],
    sample_axis: str,
) -> set[str]:
    """Check a group's layout and return its used dimensions.

    Returns:
        All explicitly labeled sample/event dimensions used by the group.

    Raises:
        TypeError: A variable is not a sampled-variable record.
        ValueError: Names, coordinate membership or shapes disagree.
    """
    if any(not isinstance(variable, SampledVariable) for variable in group):
        msg = "Groups require SampledVariable records"
        raise TypeError(msg)
    names = [variable.name for variable in group]
    if len(set(names)) != len(names) or set(names) & coords.keys():
        msg = "Group variable names must be unique and separate from coordinates"
        raise ValueError(msg)
    used: set[str] = set()
    for variable in group:
        variable.__post_init__()
        dims = (sample_axis, "draw", *variable.dims)
        if len(set(dims)) != len(dims) or not set(dims) <= coords.keys():
            msg = "Every sample/event axis needs a distinct explicit coordinate"
            raise ValueError(msg)
        if _array_shape(variable.values) != tuple(len(coords[dim]) for dim in dims):
            msg = "Sample/event shapes must match the same ordered coordinates"
            raise ValueError(msg)
        used.update(dims)
    return used


@dataclass(frozen=True, slots=True, eq=False)
class Posterior[ArrayT]:
    """Coordinate-aware native samples , with shared provenance.

    This container neither fits nor certifies a posterior. Non-finite values remain
    present; downstream adequacy/comparison policies must count their outcomes.

    Attributes:
        draws: Nonempty parameter variables; no warmup is implicitly discarded.
        coordinates: Every used dimension, including member/draw coordinates.
        provenance: Existing complete target, numerical and estimator-stage identity.
        sample_axis: Independent chains or training seeds; seeds are not MCMC chains.
        sample_stats: Optional sampled diagnostics, with the same member/draw ordering.
        log_likelihood: Optional real floating per-unit log likelihoods, with exactly
            one event dimension named ``unit``. Integration semantics are declared in
            target provenance, not inferred or certified from the shape.
    """

    draws: tuple[SampledVariable[ArrayT], ...]
    coordinates: tuple[Coordinate, ...]
    provenance: Provenance
    sample_axis: Literal["chain", "seed"] = "chain"
    sample_stats: tuple[SampledVariable[ArrayT], ...] = ()
    log_likelihood: tuple[SampledVariable[ArrayT], ...] = ()

    def __post_init__(self) -> None:
        """Validate the complete layout; see :meth:`validate`."""
        self.validate()

    @property
    def estimator_stage(self) -> str:
        """Return the shared provenance stage without duplicating its identity."""
        return self.provenance.estimator_stage

    def validate(self) -> None:
        """Check current array shapes/backend and immutable coordinate metadata.

        Call this before reuse or snapshotting, because native
        arrays may have been mutated after construction. Data values and gradients are
        not evaluated, copied, filtered or promoted. Arrays require one native backend.

        Raises:
            TypeError: A collection is mutable, a record is invalid, or backends differ.
            ValueError: Names, dimensions, coordinate lengths or sample layouts differ.
        """
        groups = (self.draws, self.sample_stats, self.log_likelihood)
        if any(not isinstance(group, tuple) for group in (*groups, self.coordinates)):
            msg = "Posterior collections must be immutable tuples"
            raise TypeError(msg)
        if not self.draws or self.sample_axis not in {"chain", "seed"}:
            msg = "Nonempty draws and an explicit chain/seed sample axis are required"
            raise ValueError(msg)
        Provenance.model_validate(self.provenance)
        for coordinate in self.coordinates:
            Coordinate.model_validate(coordinate)
        coords = {coordinate.name: coordinate.labels for coordinate in self.coordinates}
        if len(coords) != len(self.coordinates):
            msg = "Coordinate dimension names must be unique"
            raise ValueError(msg)
        used: set[str] = set()
        for group in groups:
            used.update(_group_dimensions(group, coords, self.sample_axis))
        if used != coords.keys():
            msg = "Unused coordinates are not silently retained or discarded"
            raise ValueError(msg)
        array_namespace(*(variable.values for group in groups for variable in group))
        for variable in self.log_likelihood:
            xp = array_namespace(variable.values)
            if variable.dims != ("unit",) or not xp.isdtype(
                getattr(variable.values, "dtype", None), "real floating"
            ):
                msg = "Log likelihoods need floating values and one 'unit' dimension"
                raise ValueError(msg)
