"""Independent native Gaussian information and scoped structural-record checks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.identify import (
    InformationContext,
    InformationLimits,
    ScalingAssessment,
    gaussian_mean_information,
    local_spectrum,
)
from op_infer.posterior import ArtifactReference


def _ref(name: str) -> ArtifactReference:
    return ArtifactReference.from_bytes(name.encode(), kind="toy-v1")


def _context(parameters: int = 2) -> InformationContext:
    return InformationContext(
        source_model=_ref("model"),
        design=_ref("design"),
        parameter_point=_ref("point"),
        initialization=_ref("initialization"),
        observation=_ref("noise"),
        numerical=_ref("method"),
        qualification=_ref("checked"),
        parameter_names=tuple(f"p{i}" for i in range(parameters)),
        parameter_units=("unitless",) * parameters,
        observable_names=("measurement",),
    )


def _backend(name: str) -> tuple[Any, Any]:
    if name == "numpy":
        return np, nullcontext()
    module = pytest.importorskip(name)
    if name == "torch":
        return module, nullcontext()
    enable = getattr(module, "enable_x64", None)
    if enable is None:
        enable = module.experimental.enable_x64
    return pytest.importorskip("jax.numpy"), enable()


def _host(value: Any) -> Any:  # noqa: ANN401
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_information_matches_independent_weighted_linear_design(
    backend: str, dtype: str
) -> None:
    """Unequal noise and a missing NaN row detect weighting and mask mistakes."""
    xp, precision = _backend(backend)
    with precision:
        values = xp.asarray(
            [[[1.0, 2.0]], [[3.0, 4.0]], [[np.nan, np.nan]]], dtype=getattr(xp, dtype)
        )
        sigma = xp.asarray([[2.0], [1.0], [np.nan]], dtype=values.dtype)
        mask = xp.asarray([[True], [True], [False]], dtype=xp.bool)
        context = _context()
        information = gaussian_mean_information(
            values, sigma, context=context, mask=mask
        )
        expected = np.array([[9.25, 12.5], [12.5, 17.0]])
        np.testing.assert_allclose(
            _host(information.matrix), expected, rtol=2e-6, atol=1e-7
        )
        assert getattr(information.matrix, "dtype", None) == values.dtype
        assert bool(_host(information.valid))
        assert int(_host(information.n_observed)) == 2
        report = local_spectrum(information, rtol=1e-6, atol=1e-8)
        trace = 26.25
        gap = np.sqrt((9.25 - 17.0) ** 2 + 4 * 12.5**2)
        np.testing.assert_allclose(
            report.eigenvalues,
            [(trace - gap) / 2, (trace + gap) / 2],
            rtol=2e-6,
            atol=1e-7,
        )
        vectors = np.asarray(report.eigenvectors)
        np.testing.assert_allclose(
            expected @ vectors, vectors * np.asarray(report.eigenvalues), atol=1e-6
        )
        assert report.local_rank == 2
        assert report.context == context
        assert type(report).model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_information_preserves_native_jit_and_parameter_gradient_graphs(
    backend: str,
) -> None:
    """An independent quadratic derivative catches detached/native snapshot paths."""
    xp, precision = _backend(backend)
    with precision:
        values = xp.asarray(
            [[[1.0, 2.0]], [[3.0, 4.0]], [[np.nan, np.nan]]], dtype=xp.float64
        )
        sigma = xp.asarray([[2.0], [1.0], [np.nan]], dtype=xp.float64)
        mask = xp.asarray([[True], [True], [False]], dtype=xp.bool)
        context = _context()

        def quadratic(value: Any) -> Any:  # noqa: ANN401
            return xp.sum(
                gaussian_mean_information(
                    value, sigma, context=context, mask=mask
                ).matrix
            )

        if backend == "jax":
            jax = pytest.importorskip("jax")
            gradient = jax.jit(jax.grad(quadratic))(values)
        else:
            values.requires_grad_()
            quadratic(values).backward()
            gradient = values.grad
        np.testing.assert_allclose(
            _host(gradient),
            [[[1.5, 1.5]], [[14.0, 14.0]], [[0.0, 0.0]]],
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_invalid_or_empty_measurements_never_receive_a_spectrum(backend: str) -> None:
    """Invalid selected noise and all-missing data remain distinct from rank zero."""
    xp, precision = _backend(backend)
    with precision:
        values = xp.asarray([[1.0, 2.0]], dtype=xp.float64)
        for sigma, mask in ((-1.0, True), (0.0, True), (np.nan, True), (1.0, False)):
            information = gaussian_mean_information(
                values,
                xp.asarray(sigma, dtype=values.dtype),
                context=_context(),
                mask=xp.asarray([mask], dtype=xp.bool),
            )
            assert not bool(_host(information.valid))
            assert np.isnan(_host(information.matrix)).all()
            with pytest.raises(ValueError, match="Invalid/empty"):
                local_spectrum(information, rtol=1e-8, atol=1e-12)


def test_zero_local_derivative_does_not_establish_a_global_symmetry() -> None:
    """The injective mean theta**3 has zero local information at theta=0."""
    information = gaussian_mean_information(
        np.zeros((1, 1)), np.array(1.0), context=_context(1)
    )
    report = local_spectrum(information, rtol=1e-8, atol=1e-12)
    assert report.local_rank == 0
    structural = ScalingAssessment(
        source_model=_context(1).source_model,
        status="unassessed",
        reason="No structural assessment was requested",
    )
    assert structural.action_rank is None
    with pytest.raises(ValidationError):
        ScalingAssessment.model_validate(
            structural.model_copy(update={"action_rank": 0})
        )


def test_information_layout_coordinates_and_work_bounds_are_explicit() -> None:
    """Parameter reorderings and unqualified transformations cannot pass implicitly."""
    with pytest.raises(ValueError, match="layout"):
        gaussian_mean_information(np.ones((2, 3)), np.array(1.0), context=_context())
    with pytest.raises(ValueError, match="bound"):
        gaussian_mean_information(
            np.ones((1, 2)),
            np.array(1.0),
            context=_context(),
            limits=InformationLimits(max_parameters=1),
        )
    with pytest.raises(TypeError, match="native arrays"):
        gaussian_mean_information(np.ones((1, 2)), 1.0, context=_context())
    with pytest.raises(ValidationError):
        InformationContext.model_validate(
            {**_context().model_dump(), "coordinates": "log"}
        )
    with pytest.raises(ValidationError):
        InformationContext.model_validate(
            {**_context().model_dump(), "nuisance_treatment": "profile"}
        )
    with pytest.raises(ValidationError):
        InformationContext.model_validate(
            {**_context().model_dump(), "schema_version": True}
        )


def test_numerical_range_is_not_mistaken_for_valid_information() -> None:
    """Small-scale weighting is stable; overflowing Gram products are invalid."""
    stable = gaussian_mean_information(
        np.array([[1e-200]]), np.array(1e-200), context=_context(1)
    )
    np.testing.assert_allclose(_host(stable.matrix), [[1.0]])
    with np.errstate(over="ignore"):
        overflow = gaussian_mean_information(
            np.array([[1e308]]), np.array(1.0), context=_context(1)
        )
    assert not bool(overflow.valid)
    assert np.isnan(_host(overflow.matrix)).all()
