"""Independent layouts and failure preservation for native posterior containers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.posterior import Coordinate, Posterior, SampledVariable
from tests.test_provenance import provenance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def posterior(
    *, sample_axis: Literal["chain", "seed"] = "chain"
) -> Posterior[NDArray[np.generic]]:
    """Create identifiable synthetic members, diagnostics and failed unit entries."""
    values = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    likelihood = -np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    likelihood[0, 0, :] = (-np.inf, np.nan, np.inf)
    return Posterior(
        draws=(SampledVariable("theta", values, ("component",)),),
        coordinates=(
            Coordinate(name=sample_axis, labels=(17, 29)),
            Coordinate(name="draw", labels=(10, 20, 30)),
            Coordinate(name="component", labels=("left", "right")),
            Coordinate(name="unit", labels=("u3", "u1", "u2")),
        ),
        provenance=provenance(),
        sample_axis=sample_axis,
        sample_stats=(
            SampledVariable(
                "diverging", np.array([[False, True, False], [True, False, False]])
            ),
            SampledVariable(
                "steps", np.array([[2, 3, 5], [7, 11, 13]], dtype=np.int32)
            ),
        ),
        log_likelihood=(SampledVariable("outcome", likelihood, ("unit",)),),
    )


def test_native_arrays_and_provenance_are_retained() -> None:
    """No flattening, sorting, host copy or automatic failure deletion occurs."""
    result = posterior()
    assert result.estimator_stage == result.provenance.estimator_stage
    assert result.coordinates[3].labels == ("u3", "u1", "u2")
    np.testing.assert_array_equal(result.draws[0].values[1, 2], [10, 11])
    assert np.count_nonzero(~np.isfinite(result.log_likelihood[0].values)) == 3
    assert result.log_likelihood[0].values.shape == (2, 3, 3)
    values = result.draws[0].values
    assert replace(result).draws[0].values is values
    with pytest.raises(FrozenInstanceError):
        result.sample_axis = "seed"  # type: ignore[misc]
    values[0, 0, 0] = 123
    assert result.draws[0].values[0, 0, 0] == 123


def test_live_array_containers_use_identity_equality() -> None:
    """Equality cannot accidentally evaluate a native graph or ambiguous array bool."""
    first, second = posterior(), posterior()
    assert first != second
    assert first.draws[0] != second.draws[0]


@pytest.mark.parametrize("axis", ["chain", "seed"])
def test_member_meaning_is_explicit(axis: Literal["chain", "seed"]) -> None:
    """Training seeds retain a separate sample dimension from MCMC chains."""
    result = posterior(sample_axis=axis)
    assert result.sample_axis == axis
    assert result.coordinates[0].name == axis
    assert result.draws[0].values.shape[:2] == (2, 3)


@pytest.mark.parametrize(
    "labels",
    [
        (),
        (1, 1),
        (1, "a"),
        (True,),
        (float("nan"),),
        (float("inf"),),
        (2**63,),
        ("",),
        ("a\0b",),
    ],
)
def test_coordinate_failures(labels: tuple[Any, ...]) -> None:
    """Ambiguous, non-finite and nonportable identities are refused."""
    with pytest.raises(ValidationError):
        Coordinate(name="unit", labels=labels)


@pytest.mark.parametrize("name", ["", "a/b", "has space", "1x", "θ"])
def test_portable_names(name: str) -> None:
    """Names cannot create group paths or conflict with portable storage syntax."""
    with pytest.raises(ValidationError):
        Coordinate(name=name, labels=(1,))
    with pytest.raises(ValueError, match="ASCII"):
        SampledVariable(name, np.ones((1, 1)))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("shape", "shapes"),
        ("missing", "coordinate"),
        ("extra", "Unused"),
        ("duplicate_coordinate", "Coordinate"),
        ("duplicate_variable", "unique"),
        ("reserved_variable", "separate"),
        ("sample_event", "distinct"),
        ("loglikelihood_axis", "Log likelihoods"),
        ("integer_likelihood", "Log likelihoods"),
    ],
)
def test_incompatible_layouts_are_rejected(case: str, message: str) -> None:
    """Equal array sizes alone cannot justify changing coordinate/sample meaning."""
    original = posterior()
    changes: dict[str, Any] = {}
    if case == "shape":
        changes["draws"] = (
            SampledVariable("theta", np.zeros((3, 2, 2)), ("component",)),
        )
    elif case == "missing":
        changes["coordinates"] = original.coordinates[1:]
    elif case == "extra":
        changes["coordinates"] = (
            *original.coordinates,
            Coordinate(name="unused", labels=(1,)),
        )
    elif case == "duplicate_coordinate":
        changes["coordinates"] = (*original.coordinates, original.coordinates[0])
    elif case == "duplicate_variable":
        changes["draws"] = (original.draws[0], original.draws[0])
    elif case == "reserved_variable":
        changes["draws"] = (
            SampledVariable("draw", np.ones((2, 3, 2)), ("component",)),
        )
    elif case == "sample_event":
        changes["draws"] = (SampledVariable("theta", np.ones((2, 3, 3)), ("draw",)),)
    elif case == "loglikelihood_axis":
        changes["coordinates"] = tuple(
            coord for coord in original.coordinates if coord.name != "unit"
        )
        changes["log_likelihood"] = (
            SampledVariable("outcome", np.ones((2, 3, 2)), ("component",)),
        )
    elif case == "integer_likelihood":
        changes["log_likelihood"] = (
            SampledVariable("outcome", np.ones((2, 3, 3), dtype=np.int64), ("unit",)),
        )
    with pytest.raises(ValueError, match=message):
        replace(original, **changes)


def test_empty_draws_are_not_a_posterior() -> None:
    """Missing fits remain explicit errors rather than empty successful output."""
    with pytest.raises(ValueError, match="Nonempty draws"):
        replace(posterior(), draws=())


def test_mutated_shape_is_checked_on_reuse() -> None:
    """Native payload mutability cannot silently change the declared layout."""
    result = posterior()
    result.draws[0].values.resize((3, 2, 2), refcheck=False)
    with pytest.raises(ValueError, match="shapes"):
        result.validate()


@pytest.mark.parametrize(
    "values",
    [
        np.ones((2,)),
        np.ones((2, 3, 4)),
        np.ones((2, 3), dtype=np.complex128),
        [[1, 2]],
        np.array([["x"]]),
    ],
)
def test_invalid_arrays(values: object) -> None:
    """Lists, complex/object-like data and undeclared axes are not converted."""
    with pytest.raises((TypeError, ValueError)):
        SampledVariable("theta", values)


def test_mutable_collections_are_refused() -> None:
    """Layout identities cannot mutate behind frozen records."""
    original = posterior()
    with pytest.raises(TypeError, match="immutable"):
        replace(original, draws=list(original.draws))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Coordinate(name="draw", labels=[1, 2])  # type: ignore[arg-type]


def test_optional_adapter_is_importable_in_core_install() -> None:
    """Core CI has no optional backends or I/O dependencies installed."""
    from op_infer.adapters.arviz import (  # noqa: PLC0415
        from_arviz,
        load_netcdf,
        to_arviz,
    )

    assert all(callable(function) for function in (to_arviz, from_arviz, load_netcdf))
