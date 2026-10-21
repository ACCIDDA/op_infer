"""Optional real DataTree/netCDF round trips and refused snapshot reuse."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.adapters.arviz import from_arviz, load_netcdf, to_arviz
from op_infer.posterior import Coordinate, SampledVariable
from tests.test_posterior import posterior
from tests.test_provenance import reference

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

xr = pytest.importorskip("xarray")
pytest.importorskip("arviz_base")
pytest.importorskip("netCDF4")


@pytest.mark.parametrize("axis", ["chain", "seed"])
@pytest.mark.parametrize("likelihood_dtype", ["float32", "float64"])
def test_round_trip_preserves_all_groups_and_unit_failures(
    tmp_path: Path, axis: Literal["chain", "seed"], likelihood_dtype: str
) -> None:
    """Distinct values/coordinates and mixed dtypes survive memory and disk exactly."""
    original = posterior(sample_axis=axis)
    original = replace(
        original,
        log_likelihood=(
            SampledVariable(
                "outcome",
                original.log_likelihood[0].values.astype(likelihood_dtype),
                ("unit",),
            ),
        ),
    )
    tree = to_arviz(original)
    assert tree["posterior"]["theta"].dims == (axis, "draw", "component")
    assert tree["log_likelihood"]["outcome"].shape == (2, 3, 3)
    file = tmp_path / "posterior.nc"
    tree.to_netcdf(file, engine="netcdf4")
    for restored in (
        from_arviz(tree, expected=original.provenance),
        load_netcdf(file, expected=original.provenance),
    ):
        assert restored.provenance == original.provenance
        assert restored.sample_axis == axis
        assert restored.coordinates == original.coordinates
        for old, new in zip(
            (*original.draws, *original.sample_stats, *original.log_likelihood),
            (*restored.draws, *restored.sample_stats, *restored.log_likelihood),
            strict=True,
        ):
            assert old.name == new.name
            assert old.dims == new.dims
            assert old.values.dtype == new.values.dtype
            np.testing.assert_array_equal(old.values, new.values)
            assert not np.shares_memory(old.values, new.values)
        assert np.count_nonzero(~np.isfinite(restored.log_likelihood[0].values)) == 3
    file.unlink()  # Restoration must not retain a lazy file reference.


def test_snapshot_is_detached_from_later_mutations() -> None:
    """Changing a live array cannot change a previously recorded snapshot."""
    original = posterior()
    tree = to_arviz(original)
    original.draws[0].values[:] = 1000
    restored = from_arviz(tree, expected=original.provenance)
    np.testing.assert_array_equal(
        restored.draws[0].values, np.arange(12).reshape(2, 3, 2)
    )


def test_masked_units_cannot_be_silently_omitted() -> None:
    """A reporting mask cannot erase failed units or alter the fixed denominator."""
    original = posterior()
    # This public factory has no annotations in NumPy 2.0.
    factory = cast("Callable[..., NDArray[np.generic]]", np.ma.array)
    masked = factory(original.log_likelihood[0].values, mask=True)
    source = replace(
        original, log_likelihood=(SampledVariable("outcome", masked, ("unit",)),)
    )
    with pytest.raises(TypeError, match="unmasked NumPy"):
        to_arviz(source)


def test_provenance_is_checked_before_loading_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incompatible artifact fails on identity, before any sample access."""
    original = posterior()
    tree = to_arviz(original)
    expected = original.provenance.model_copy(
        update={"design": reference("different design")}
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(f"Payload was read before provenance was checked: {args}, {kwargs}")

    monkeypatch.setattr(xr.DataTree, "__getitem__", forbidden)
    with pytest.raises(ValueError, match="Provenance mismatch"):
        from_arviz(tree, expected=expected)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("data", "content"),
        ("coordinate", "identity/order"),
        ("coordinate_type", "scalar types"),
        ("sample_meaning", "sample-axis"),
        ("extra_variable", "variables"),
        ("extra_group", "groups"),
        ("missing_group", "groups"),
        ("dtype", "dtype"),
        ("dimensions", "dimension order"),
    ],
)
def test_changed_tree_is_refused(case: str, message: str) -> None:
    """Reordering, corruption and undeclared data cannot disappear."""
    original = posterior()
    tree = to_arviz(original)
    if case == "data":
        tree["posterior"].to_dataset()["theta"].values[0, 0, 0] += 1
    elif case == "coordinate":
        dataset = (
            tree["log_likelihood"].to_dataset().assign_coords(unit=["u1", "u3", "u2"])
        )
        tree["log_likelihood"] = dataset
    elif case == "coordinate_type":
        tree["posterior"] = (
            tree["posterior"].to_dataset().assign_coords(draw=[10.0, 20.0, 30.0])
        )
    elif case == "sample_meaning":
        tree["posterior"].attrs["sample_dims"] = ["seed", "draw"]
    elif case == "extra_variable":
        tree["posterior"]["extra"] = xr.DataArray(
            np.ones((2, 3)), dims=("chain", "draw")
        )
    elif case == "extra_group":
        tree["predictions"] = xr.Dataset()
    elif case == "missing_group":
        del tree["sample_stats"]
    elif case == "dtype":
        tree["posterior"]["theta"] = tree["posterior"]["theta"].astype(np.float32)
    elif case == "dimensions":
        tree["posterior"]["theta"] = tree["posterior"]["theta"].transpose(
            "draw", "chain", "component"
        )
    with pytest.raises(ValueError, match=message):
        from_arviz(tree, expected=original.provenance)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"unexpected": 1},
    ],
)
def test_unknown_or_ambiguous_schema_is_refused(change: dict[str, object]) -> None:
    """Layout schema is strict and cannot masquerade as another version."""
    original = posterior()
    tree = to_arviz(original)
    header = json.loads(tree.attrs["op_infer_posterior"])
    header.update(change)
    tree.attrs["op_infer_posterior"] = json.dumps(header)
    with pytest.raises(ValidationError):
        from_arviz(tree, expected=original.provenance)


def test_unlabeled_arviz_tree_requires_an_adapter() -> None:
    """Sampler outputs cannot acquire invented provenance during import."""
    with pytest.raises(ValueError, match="Missing"):
        from_arviz(xr.DataTree(), expected=posterior().provenance)


@pytest.mark.parametrize("dtype", ["float16", "longdouble"])
def test_storage_cannot_silently_promote_or_truncate(dtype: str) -> None:
    """Unsupported storage types require caller-side conversion."""
    original = posterior()
    source = replace(
        original,
        draws=(
            SampledVariable(
                "theta", original.draws[0].values.astype(dtype), ("component",)
            ),
        ),
    )
    with pytest.raises(TypeError, match="Snapshots support"):
        to_arviz(source)


def test_same_variable_name_can_have_group_specific_dimensions() -> None:
    """ArviZ's global dims mapping cannot accidentally conflate groups."""
    original = posterior()
    source = replace(
        original,
        sample_stats=(
            *original.sample_stats,
            SampledVariable("theta", np.ones((2, 3))),
        ),
    )
    restored = from_arviz(to_arviz(source), expected=source.provenance)
    assert restored.draws[0].dims == ("component",)
    assert restored.sample_stats[-1].dims == ()


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8", "<i8", ">i8", "<u8"])
def test_noncontiguous_big_endian_and_float_coordinates(
    tmp_path: Path, dtype: str
) -> None:
    """Physical coordinate order and array storage interpretation remain explicit."""
    original = posterior()
    values = np.arange(24, dtype=dtype).reshape(2, 3, 4)[..., ::2]
    coordinates = tuple(
        Coordinate(name="draw", labels=(0.1, 0.4, 0.9))
        if coord.name == "draw"
        else coord
        for coord in original.coordinates
    )
    source = replace(
        original,
        draws=(SampledVariable("theta", values, ("component",)),),
        coordinates=coordinates,
    )
    file = tmp_path / "noncontiguous.nc"
    to_arviz(source).to_netcdf(file, engine="netcdf4")
    restored = load_netcdf(file, expected=source.provenance)
    assert restored.draws[0].values.dtype == values.dtype
    assert restored.coordinates == coordinates
    np.testing.assert_array_equal(restored.draws[0].values, values)
