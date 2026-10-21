"""Mandatory native graph qualification; no sampler or consumer model is used."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from op_infer.adapters.arviz import to_arviz
from op_infer.posterior import Coordinate, Posterior, SampledVariable
from tests.test_provenance import provenance


def test_jax_jit_and_gradient_preserve_native_samples() -> None:
    """A container can be constructed during tracing without host evaluation."""
    jax = pytest.importorskip("jax")
    xp = importlib.import_module("jax.numpy")
    coordinates = (
        Coordinate(name="chain", labels=(0, 1)),
        Coordinate(name="draw", labels=(0, 1, 2)),
    )
    identity = provenance()

    def objective(rate: object) -> object:
        values = xp.reshape(xp.arange(1, 7, dtype=xp.float32), (2, 3)) * rate
        result = Posterior(
            draws=(SampledVariable("rate", values),),
            coordinates=coordinates,
            provenance=identity,
        )
        assert result.draws[0].values is values
        return xp.sum(result.draws[0].values ** 2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(xp.asarray(2.0))
    assert float(value) == pytest.approx(4 * 91)
    assert float(gradient) == pytest.approx(4 * 91)


def test_torch_autograd_and_mixed_backend_rejection() -> None:
    """Autograd stays attached; a mixed NumPy/Torch group cannot be aligned silently."""
    torch = pytest.importorskip("torch")
    rate = torch.tensor(2.0, requires_grad=True)
    values = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3) * rate
    result = Posterior(
        draws=(SampledVariable("rate", values),),
        coordinates=(
            Coordinate(name="seed", labels=(3, 5)),
            Coordinate(name="draw", labels=(0, 1, 2)),
        ),
        provenance=provenance(),
        sample_axis="seed",
    )
    assert result.draws[0].values is values
    torch.sum(result.draws[0].values ** 2).backward()
    assert float(rate.grad) == pytest.approx(4 * 91)
    with pytest.raises(TypeError):
        Posterior(
            draws=(
                SampledVariable("rate", values),
                SampledVariable("other", np.ones((2, 3))),
            ),
            coordinates=result.coordinates,
            provenance=result.provenance,
            sample_axis=result.sample_axis,
        )
    with pytest.raises(TypeError, match="Explicitly transfer/detach"):
        to_arviz(result)
