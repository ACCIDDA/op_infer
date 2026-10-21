"""Optional ArviZ snapshots and named MCMC diagnostics, using shared provenance.

No consumer model/data or inference library is imported here.
Only op_infer-authored snapshots are accepted; arbitrary sampler trees need an
explicit adapter with independently supplied provenance. Full authenticated
artifact loading, publication and shard merging remain planned.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from op_infer.posterior import (
    ArtifactReference,
    BackendVersion,
    Coordinate,
    Posterior,
    Provenance,
    SampledVariable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from xarray import DataTree

__all__ = [
    "DiagnosticComponent",
    "DiagnosticQuantity",
    "DiagnosticValue",
    "MCMCDiagnostics",
    "diagnose_mcmc",
    "from_arviz",
    "load_netcdf",
    "to_arviz",
]

_HEADER = "op_infer_posterior"
_GROUPS = ("posterior", "sample_stats", "log_likelihood")


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class _Variable(_Record):
    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    content: ArtifactReference


class _Group(_Record):
    name: Literal["posterior", "sample_stats", "log_likelihood"]
    variables: tuple[_Variable, ...]


class _Header(_Record):
    schema_version: Literal[1]
    provenance: Provenance
    sample_axis: Literal["chain", "seed"]
    coordinates: tuple[Coordinate, ...]
    groups: tuple[_Group, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "Snapshot schema version must be an integer"
            raise ValueError(msg)
        return value


def _host(values: object) -> NDArray[np.generic]:
    """Require an explicit portable NumPy snapshot boundary.

    Returns:
        Original host array; the caller copies it when taking the snapshot.

    Raises:
        TypeError: Native graphs/devices, masks or unsupported storage dtypes need
            explicit caller-side conversion rather than silent transfer/promotion.
    """
    if not isinstance(values, np.ndarray) or isinstance(values, np.ma.MaskedArray):
        msg = "Explicitly transfer/detach to unmasked NumPy before ArviZ snapshotting"
        raise TypeError(msg)
    if not (
        (values.dtype.kind in "biu" and values.dtype.itemsize <= 8)
        or (values.dtype.kind == "f" and values.dtype.itemsize in {4, 8})
    ):
        msg = "Snapshots support bool, 8-64 bit integers, and float32/float64"
        raise TypeError(msg)
    return values


def _content(values: NDArray[np.generic]) -> ArtifactReference:
    """Hash canonical little-endian C-order numeric bytes for snapshot integrity.

    Returns:
        Existing content-reference type; this is not authentication of its header.
    """
    canonical = values.astype(values.dtype.newbyteorder("<"), copy=False)
    return ArtifactReference.from_bytes(
        canonical.tobytes(order="C"), kind="numpy-c-little-endian-v1"
    )


def to_arviz[ArrayT](posterior: Posterior[ArrayT]) -> DataTree:
    """Take an independent host snapshot using ArviZ 1.x's DataTree conventions.

    Args:
        posterior: Valid native container whose values were explicitly transferred
            to NumPy. Live arrays are copied; later mutations do not change the tree.

    Returns:
        An xarray DataTree with versioned layout/shared provenance and array digests.
        Use its standard ``to_netcdf(path, engine="netcdf4")`` for file writing.

    Raises:
        ImportError: Install ``op_infer[arviz]`` to use this optional integration.
        TypeError: Values have not crossed the explicit host/storage boundary.
        ValueError: The current posterior layout is invalid.
    """
    posterior.validate()
    for group in (posterior.draws, posterior.sample_stats, posterior.log_likelihood):
        for variable in group:
            _host(variable.values)

    import arviz_base as az  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    datasets: dict[str, xr.Dataset] = {}
    descriptors: list[_Group] = []
    variables = (posterior.draws, posterior.sample_stats, posterior.log_likelihood)
    for name, group in zip(_GROUPS, variables, strict=True):
        if not group:
            continue
        arrays = {
            variable.name: np.array(_host(variable.values), copy=True)
            for variable in group
        }
        records = tuple(
            _Variable(
                name=variable.name,
                dims=variable.dims,
                shape=arrays[variable.name].shape,
                dtype=arrays[variable.name].dtype.str,
                content=_content(arrays[variable.name]),
            )
            for variable in group
        )
        descriptors.append(
            _Group(
                name=cast(
                    "Literal['posterior', 'sample_stats', 'log_likelihood']", name
                ),
                variables=records,
            )
        )
        dimensions = {
            posterior.sample_axis,
            "draw",
            *(dim for variable in group for dim in variable.dims),
        }
        converted = az.from_dict(
            {name: arrays},
            sample_dims=[posterior.sample_axis, "draw"],
            coords={
                coordinate.name: list(coordinate.labels)
                for coordinate in posterior.coordinates
                if coordinate.name in dimensions
            },
            dims={variable.name: list(variable.dims) for variable in group},
        )
        dataset = converted[name].to_dataset()
        # ArviZ 1.0 does not automatically retain this attribute. Record the
        # explicit contract on every version rather than relying on rcParams.
        dataset.attrs["sample_dims"] = [posterior.sample_axis, "draw"]
        for stored in dataset.variables.values():
            stored.encoding["_FillValue"] = None
        datasets[name] = dataset
    header = _Header(
        schema_version=1,
        provenance=posterior.provenance,
        sample_axis=posterior.sample_axis,
        coordinates=posterior.coordinates,
        groups=tuple(descriptors),
    )
    datasets["/"] = xr.Dataset(attrs={_HEADER: header.model_dump_json()})
    return xr.DataTree.from_dict(datasets)


def _restore_group(
    tree: DataTree, descriptor: _Group, header: _Header
) -> tuple[SampledVariable[NDArray[np.generic]], ...]:
    """Restore one exact declared group, refusing alignment/casting/data corruption.

    Returns:
        Detached host variables in the original order.

    Raises:
        ValueError: The tree disagrees with declared dimensions, coordinates or bytes.
    """
    node = tree[descriptor.name].to_dataset()
    names = tuple(variable.name for variable in descriptor.variables)
    if not names or len(set(names)) != len(names) or set(node.data_vars) != set(names):
        msg = "Snapshot variables differ from the declared group"
        raise ValueError(msg)
    if tuple(node.attrs.get("sample_dims", ())) != (header.sample_axis, "draw"):
        msg = "Snapshot sample-axis meaning changed"
        raise ValueError(msg)
    dimensions = {
        header.sample_axis,
        "draw",
        *(dim for variable in descriptor.variables for dim in variable.dims),
    }
    coords = {
        coordinate.name: coordinate.labels
        for coordinate in header.coordinates
        if coordinate.name in dimensions
    }
    if set(node.coords) != dimensions:
        msg = "Snapshot coordinates differ from declared dimensions"
        raise ValueError(msg)
    for name, labels in coords.items():
        if node[name].dims != (name,) or tuple(node[name].values.tolist()) != labels:
            msg = "Snapshot coordinate identity/order changed"
            raise ValueError(msg)
        if any(
            type(actual) is not type(expected)
            for actual, expected in zip(node[name].values.tolist(), labels, strict=True)
        ):
            msg = "Snapshot coordinate scalar types changed"
            raise ValueError(msg)
    restored: list[SampledVariable[NDArray[np.generic]]] = []
    for variable in descriptor.variables:
        data = node[variable.name]
        dtype = np.dtype(variable.dtype)
        values = _host(np.asarray(data.values))
        if (
            data.dims != (header.sample_axis, "draw", *variable.dims)
            or values.shape != variable.shape
            or (values.dtype.kind, values.dtype.itemsize)
            != (dtype.kind, dtype.itemsize)
        ):
            msg = "Snapshot dimension order, shape or dtype changed"
            raise ValueError(msg)
        copied = np.array(values, dtype=dtype, copy=True)
        if _content(copied) != variable.content:
            msg = "Snapshot array content does not match its recorded digest"
            raise ValueError(msg)
        restored.append(SampledVariable(variable.name, copied, variable.dims))
    return tuple(restored)


def from_arviz(
    tree: DataTree, *, expected: Provenance
) -> Posterior[NDArray[np.generic]]:
    """Restore an op_infer-authored snapshot, checking expected provenance first.

    Args:
        tree: DataTree produced by :func:`to_arviz`, possibly read from netCDF.
        expected: Independently supplied complete provenance required for reuse.

    Returns:
        Detached NumPy samples, with all original unit entries and seed/chain meaning.

    Raises:
        ValueError: Missing/unknown schema, provenance mismatch, undeclared data or
            inconsistent layout/content. Self-recorded digests detect corruption;
            they do not authenticate a header or verify referenced scientific claims.
    """
    payload = tree.attrs.get(_HEADER)
    if payload is None:
        msg = "Missing op_infer snapshot header; legacy trees need an adapter"
        raise ValueError(msg)
    if not isinstance(payload, str):
        msg = "Snapshot header must be JSON text"
        raise TypeError(msg)
    header = _Header.model_validate_json(payload)
    header.provenance.require_match(expected)
    names = tuple(group.name for group in header.groups)
    if (
        "posterior" not in names
        or len(set(names)) != len(names)
        or set(tree.groups) != {"/", *(f"/{name}" for name in names)}
        or tree.data_vars
        or tree.coords
    ):
        msg = "Snapshot contains missing, duplicate or undeclared groups/root data"
        raise ValueError(msg)
    groups = {
        group.name: _restore_group(tree, group, header) for group in header.groups
    }
    return Posterior(
        draws=groups["posterior"],
        coordinates=header.coordinates,
        provenance=header.provenance,
        sample_axis=header.sample_axis,
        sample_stats=groups.get("sample_stats", ()),
        log_likelihood=groups.get("log_likelihood", ()),
    )


def load_netcdf(
    path: str | Path, *, expected: Provenance
) -> Posterior[NDArray[np.generic]]:
    """Guard a netCDF snapshot before eagerly copying its sample arrays.

    Args:
        path: Local file written from :func:`to_arviz` using netCDF4.
        expected: Independently supplied complete provenance required for reuse.

    Returns:
        Detached NumPy container; no lazy arrays or open file handles escape.

    Raises:
        ImportError: The optional ``arviz`` extra is unavailable.
        OSError: The file cannot be read.
        ValueError: The guarded snapshot contract is violated.
    """
    import xarray as xr  # noqa: PLC0415

    with xr.open_datatree(
        path, engine="netcdf4", mask_and_scale=False, decode_times=False
    ) as tree:
        return from_arviz(tree, expected=expected)


class DiagnosticValue(_Record):
    """A named ArviZ method result or an explicit unavailable/undefined outcome.

    Available values are finite and positive; ESS is absolute and can exceed the
    number of draws for antithetic samples. No adequacy threshold is applied.

    Examples:
        >>> result = DiagnosticValue(
        ...     method="rank",
        ...     value=None,
        ...     status="unavailable",
        ...     reason="single_original_chain",
        ... )
        >>> result.status, result.value
        ('unavailable', None)
    """

    method: Literal["rank", "bulk", "tail"]
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    status: Literal["available", "unavailable", "undefined"]
    reason: Annotated[str, Field(min_length=1, pattern=r"\S")] | None = None

    @model_validator(mode="after")
    def _outcome(self) -> Self:
        if (self.status == "available") != (self.value is not None) or (
            (self.reason is None) != (self.status == "available")
        ):
            msg = "Available diagnostics need a value; other outcomes need a reason"
            raise ValueError(msg)
        return self


class DiagnosticComponent(_Record):
    """One event element, indexed in its original named event dimensions."""

    index: tuple[int, ...]
    nonfinite_values: Annotated[int, Field(ge=0)] | None
    rank_rhat: DiagnosticValue
    bulk_ess: DiagnosticValue
    tail_ess: DiagnosticValue


class DiagnosticQuantity(_Record):
    """A selected variable with every event element retained, including failures.

    ``content`` hashes the copied host input in its original host dtype before
    float64 diagnostic calculation. It is unavailable if host conversion failed.
    The caller's definition reference identifies reported/aligned transformations.
    """

    name: str
    dims: tuple[str, ...]
    source_dtype: str
    host_dtype: str | None
    content: ArtifactReference | None
    components: tuple[DiagnosticComponent, ...]


class MCMCDiagnostics(_Record):
    """Versioned named diagnostic values, with original axes and provenance.

    Rank R-hat is ArviZ's maximum of rank-normalized split and folded R-hat.
    Tail ESS uses quantiles 0.05 and 0.95. Values do not imply sampler adequacy;
    thresholds, divergences and combined-gate qualification remain separate.
    """

    schema_version: Literal[1] = 1
    provenance: Provenance
    definition: ArtifactReference
    implementation_reference: ArtifactReference
    coordinates: tuple[Coordinate, ...]
    sample_axis: Literal["chain", "seed"]
    backend_versions: tuple[BackendVersion, ...]
    quantities: tuple[DiagnosticQuantity, ...]
    implementation: Literal["arviz_stats.base.array_stats"] = (
        "arviz_stats.base.array_stats"
    )
    diagnostic_dtype: Literal["float64"] = "float64"
    tail_probabilities: tuple[float, float] = (0.05, 0.95)
    relative_ess: Literal[False] = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "Diagnostic schema version must be an integer"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _layout(self) -> Self:
        if self.tail_probabilities != (0.05, 0.95):
            msg = "Tail ESS is defined at quantiles 0.05 and 0.95"
            raise ValueError(msg)
        coords = {c.name: c.labels for c in self.coordinates}
        names = tuple(q.name for q in self.quantities)
        if (
            not names
            or len(set(names)) != len(names)
            or len(coords) != len(self.coordinates)
            or not {self.sample_axis, "draw"} <= coords.keys()
        ):
            msg = "Diagnostics need distinct quantities and sample coordinates"
            raise ValueError(msg)
        for quantity in self.quantities:
            if (
                len(set(quantity.dims)) != len(quantity.dims)
                or not set(quantity.dims) <= coords.keys()
                or {self.sample_axis, "draw"}.intersection(quantity.dims)
            ):
                msg = "Diagnostic dimensions need their original coordinates"
                raise ValueError(msg)
            expected = tuple(np.ndindex(*(len(coords[d]) for d in quantity.dims)))
            if tuple(c.index for c in quantity.components) != expected:
                msg = "Every event component must remain in original coordinate order"
                raise ValueError(msg)
            for component in quantity.components:
                if (
                    component.rank_rhat.method,
                    component.bulk_ess.method,
                    component.tail_ess.method,
                ) != ("rank", "bulk", "tail"):
                    msg = "Diagnostic method names do not match their fields"
                    raise ValueError(msg)
                available = any(
                    value.status == "available"
                    for value in (
                        component.rank_rhat,
                        component.bulk_ess,
                        component.tail_ess,
                    )
                )
                if available and (
                    self.sample_axis != "chain"
                    or len(coords["draw"]) < 4
                    or component.nonfinite_values != 0
                    or quantity.host_dtype is None
                    or quantity.content is None
                ):
                    msg = "Available diagnostics require finite original MCMC inputs"
                    raise ValueError(msg)
                if (
                    component.rank_rhat.status == "available"
                    and len(coords[self.sample_axis]) < 2
                ):
                    msg = "Rank R-hat requires two original chains"
                    raise ValueError(msg)
        return self

    @property
    def n_components(self) -> int:
        """Return the complete selected scalar-quantity denominator."""
        return sum(len(q.components) for q in self.quantities)

    @property
    def n_incomplete(self) -> int:
        """Count elements with any unavailable or undefined required diagnostic."""
        return sum(
            any(v.status != "available" for v in (c.rank_rhat, c.bulk_ess, c.tail_ess))
            for q in self.quantities
            for c in q.components
        )

    @property
    def adequacy_status(self) -> Literal["not_assessed"]:
        """Keep raw diagnostics distinct from a prespecified adequacy decision."""
        return "not_assessed"

    @classmethod
    def load_json(
        cls, payload: str, *, expected: Provenance, definition: ArtifactReference
    ) -> Self:
        """Load a report only with independently supplied context and definition.

        Returns:
            The validated immutable report; self-recorded hashes are not authentication.

        Raises:
            ValueError: Schema, layout, provenance or quantity definition differs.
        """
        result = cls.model_validate_json(payload)
        result.provenance.require_match(expected)
        if result.definition != definition:
            msg = "Reported-quantity definition differs"
            raise ValueError(msg)
        return result


def _unavailable_component(
    index: tuple[int, ...],
    reason: str,
    *,
    nonfinite: int | None = None,
    status: Literal["unavailable", "undefined"] = "unavailable",
) -> DiagnosticComponent:
    values = [
        DiagnosticValue(method=m, value=None, status=status, reason=reason)
        for m in ("rank", "bulk", "tail")
    ]
    return DiagnosticComponent(
        index=index,
        nonfinite_values=nonfinite,
        rank_rhat=values[0],
        bulk_ess=values[1],
        tail_ess=values[2],
    )


def _metric(
    method: Literal["rank", "bulk", "tail"], calculate: Callable[[], object]
) -> DiagnosticValue:
    try:
        value = float(np.asarray(calculate()).item())
    except Exception as error:  # noqa: BLE001 - retain a failed diagnostic in its denominator
        return DiagnosticValue(
            method=method,
            value=None,
            status="undefined",
            reason=f"reference_error:{type(error).__name__}:{error}",
        )
    if not math.isfinite(value) or value <= 0:
        return DiagnosticValue(
            method=method,
            value=None,
            status="undefined",
            reason="reference_returned_nonfinite_or_nonpositive",
        )
    return DiagnosticValue(method=method, value=value, status="available")


class _ArrayDiagnostics(Protocol):
    def rhat(self, data: NDArray[np.float64], **kwargs: object) -> object:
        """Call the documented array R-hat interface."""
        ...

    def ess(self, data: NDArray[np.float64], **kwargs: object) -> object:
        """Call the documented array ESS interface."""
        ...


def _diagnose_component(
    values: NDArray[np.generic], index: tuple[int, ...], stats: object
) -> DiagnosticComponent:
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        return _unavailable_component(index, "nonfinite_input", nonfinite=nonfinite)
    if values.shape[1] < 4:
        return _unavailable_component(index, "fewer_than_four_draws", nonfinite=0)
    if values.dtype.kind in "iu" and (
        int(values.min()) < -(2**53) or int(values.max()) > 2**53
    ):
        return _unavailable_component(
            index, "integer_not_exact_in_float64", nonfinite=0
        )
    if np.all(values == values.flat[0]):
        return _unavailable_component(
            index, "constant_quantity", nonfinite=0, status="undefined"
        )
    if np.all(values == values[:, :1]):
        return _unavailable_component(
            index, "constant_within_each_chain", nonfinite=0, status="undefined"
        )
    data = values.astype(np.float64, copy=False)
    # Pin the documented NumPy implementation independently of global rcParams.
    reference = cast("_ArrayDiagnostics", stats)
    rhat = reference.rhat
    ess = reference.ess
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rank = (
            _metric(
                "rank",
                lambda: rhat(data, chain_axis=0, draw_axis=1, method="rank"),
            )
            if data.shape[0] >= 2
            else DiagnosticValue(
                method="rank",
                value=None,
                status="unavailable",
                reason="single_original_chain",
            )
        )
        bulk = _metric(
            "bulk",
            lambda: ess(data, chain_axis=0, draw_axis=1, method="bulk", relative=False),
        )
        tail = _metric(
            "tail",
            lambda: ess(
                data,
                chain_axis=0,
                draw_axis=1,
                method="tail",
                relative=False,
                prob=(0.05, 0.95),
            ),
        )
    return DiagnosticComponent(
        index=index, nonfinite_values=0, rank_rhat=rank, bulk_ess=bulk, tail_ess=tail
    )


def _diagnostic_host(values: object, shape: tuple[int, ...]) -> NDArray[np.generic]:
    host = np.array(_host(values), copy=True, order="C")
    if host.shape != shape:
        msg = "Host transfer must preserve original shape and axis order"
        raise ValueError(msg)
    return host


def diagnose_mcmc[ArrayT](
    posterior: Posterior[ArrayT],
    *,
    quantities: Sequence[str],
    definition: ArtifactReference,
    to_numpy: Callable[[ArrayT], NDArray[np.generic]] | None = None,
) -> MCMCDiagnostics:
    """Calculate named rank R-hat and absolute bulk/tail ESS using ArviZ Stats.

    Active consumer: Sheep_Immune's named reported/aligned quantity diagnostics.
    No estimator is reimplemented. Every selected scalar/event element remains
    present, including constants, nonfinite data and unsupported inputs.

    Args:
        posterior: Original post-warmup chain/draw container. Seed members are
            retained with unavailable MCMC diagnostics, never relabeled as chains.
        quantities: Explicit nonempty variable names in ``posterior.draws``, in
            reporting order. Derived/aligned quantities must be supplied there
            with their original sample axes and declared event coordinates.
        definition: Content reference to the reporting/transformation definition,
            including data-dependent alignment and sample selection when used.
        to_numpy: Explicit host-transfer/detachment callback for native arrays.
            Without it only unmasked supported NumPy arrays are read. Conversion
            must preserve shape/order; it does not authorize a new transformation.

    Returns:
        Immutable values/reasons, complete event denominator, dependency versions,
        original coordinates, host input digests and shared provenance. Calculation
        uses float64; source/host dtypes are recorded. No thresholds or pass label.

    Raises:
        ImportError: Install the diagnostics extra for ArviZ Stats.
        ValueError: Selection is empty, duplicated, unknown, or layout is invalid.
        TypeError: The original layout/backend is invalid or selection is a string.
    """
    posterior.validate()
    if isinstance(quantities, str):
        msg = "Select distinct nonempty names as a sequence, not one string"
        raise TypeError(msg)
    names = tuple(quantities)
    variables = {variable.name: variable for variable in posterior.draws}
    if not names or len(set(names)) != len(names) or not set(names) <= variables.keys():
        msg = "Select distinct nonempty names from posterior.draws"
        raise ValueError(msg)
    try:
        stats = importlib.import_module("arviz_stats.base").array_stats
    except ImportError as error:
        msg = "Install op_infer[diagnostics] for named MCMC diagnostics"
        raise ImportError(msg) from error
    results = []
    coords = {
        coordinate.name: coordinate.labels for coordinate in posterior.coordinates
    }
    # Avoid contextual tuple inference through NumPy 2.0's indexing overloads.
    sample_slices = (slice(None), slice(None))
    for name in names:
        variable = variables[name]
        shape = tuple(
            len(coords[d]) for d in (posterior.sample_axis, "draw", *variable.dims)
        )
        indices = tuple(np.ndindex(*shape[2:]))
        host = None
        reason = None
        if posterior.sample_axis != "chain":
            reason = "seed_axis_is_not_mcmc_chains"
        else:
            try:
                supplied = (
                    variable.values if to_numpy is None else to_numpy(variable.values)
                )
                host = _diagnostic_host(supplied, shape)
            except Exception as error:  # noqa: BLE001 - retain unsupported selected quantities
                reason = f"unsupported_input:{type(error).__name__}:{error}"
                host = None
        components = tuple(
            _unavailable_component(index, reason or "unavailable_input")
            if host is None
            else _diagnose_component(host[sample_slices + index], index, stats)
            for index in indices
        )
        results.append(
            DiagnosticQuantity(
                name=name,
                dims=variable.dims,
                source_dtype=str(getattr(variable.values, "dtype", None)),
                host_dtype=None if host is None else host.dtype.str,
                content=None if host is None else _content(host),
                components=components,
            )
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("op_infer", "arviz-stats", "numpy", "scipy")
    }
    versions["python"] = platform.python_version()
    return MCMCDiagnostics(
        provenance=posterior.provenance,
        definition=definition,
        implementation_reference=ArtifactReference.from_bytes(
            Path(__file__).read_bytes(), kind="op-infer-arviz-adapter-source-v1"
        ),
        coordinates=posterior.coordinates,
        sample_axis=posterior.sample_axis,
        quantities=tuple(results),
        backend_versions=tuple(
            BackendVersion(name=name, version=value)
            for name, value in sorted(versions.items())
        ),
    )
