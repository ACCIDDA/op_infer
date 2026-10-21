"""Bounded optional op_engine refinement/derivative qualification for #43.

Core imports remain numerical-library independent. Run ``just qualify-numerical``
in an environment containing the op, jax and torch extras; that recipe requires all
three backends rather than treating an optional skip as qualification.
"""

import math
from collections.abc import Callable
from contextlib import nullcontext
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from op_infer.posterior import ArtifactReference

TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)
DATA = (0.83, 0.55, 0.28, 0.18)
SIGMA = 0.2


def _analytic_reference(rate: float) -> tuple[float, float]:
    """Evaluate the independent analytic likelihood and continuous-model gradient."""
    predictions = np.exp(-rate * np.asarray(TIMES[1:]))
    value = float(
        -0.5 * np.sum(((predictions - DATA) / SIGMA) ** 2)
        - len(DATA) * math.log(SIGMA * math.sqrt(2 * math.pi))
    )
    gradient = float(
        np.sum(
            (np.asarray(DATA) - predictions)
            / SIGMA**2
            * (-np.asarray(TIMES[1:]) * predictions)
        )
    )
    return value, gradient


def _likelihood_factory(
    solver: Any,  # noqa: ANN401
    initial: Any,  # noqa: ANN401
    xp: Any,  # noqa: ANN401
    partitions: tuple[tuple[float, ...], ...],
) -> Callable[[Any], Any]:
    """Compose the optional engine's public fixed-step kernel on a static partition."""
    observed = xp.asarray(DATA, dtype=initial.dtype)

    def log_likelihood(parameter: Any) -> Any:  # noqa: ANN401
        state = initial
        predictions = []
        for interval, steps in enumerate(partitions):
            time = TIMES[interval]
            for step in steps:
                state, _ = solver.fixed_explicit_step(
                    lambda _t, y: -parameter * y,
                    method="rk4",
                    t=time,
                    dt=step,
                    y=state,
                )
                time += step
            # One scalar physical concentration, evaluated at stored times.
            predictions.append(xp.sum(state))
        predicted = xp.stack(predictions)
        return -0.5 * xp.sum(((predicted - observed) / SIGMA) ** 2) - len(
            DATA
        ) * math.log(SIGMA * math.sqrt(2 * math.pi))

    return log_likelihood


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("rate", [0.4, 1.1])
def test_fixed_mesh_likelihood_and_gradient_refinement(
    backend: str, rate: float
) -> None:
    """Value/derivative errors shrink with the physical observations/prior fixed."""
    engine = pytest.importorskip("op_engine")
    core_solver = pytest.importorskip("op_engine.core_solver")
    model_core = pytest.importorskip("op_engine.model_core")
    module: Any = np
    jax: Any = None
    if backend == "jax":
        jax = pytest.importorskip("jax")
        module = pytest.importorskip("jax.numpy")
    elif backend == "torch":
        module = pytest.importorskip("torch")
    precision = nullcontext()
    if jax is not None:
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        precision = enable()
    with precision:
        initial = module.ones((1, 1), dtype=module.float64)
        core = engine.ModelCore(
            1,
            1,
            np.asarray(TIMES),
            options=model_core.ModelCoreOptions(dtype=np.dtype("float64")),
        )
        solver = engine.CoreSolver(core)
        xp = engine.array_namespace(initial)
        exact_log_likelihood, exact_gradient = _analytic_reference(rate)
        value_errors = []
        gradient_errors = []
        mesh_identities = []
        # The physical design, concentration interpretation and Uniform(0.1, 2)
        # physical-rate prior do not depend on the numerical partition.
        for max_step in (0.25, 0.125, 0.0625):
            mesh = tuple(
                core_solver.fixed_step_sizes(start, end, max_step)
                for start, end in pairwise(TIMES)
            )
            mesh_identities.append(
                ArtifactReference.from_bytes(repr(mesh).encode(), kind="fixed-mesh-v1")
            )

            log_likelihood = _likelihood_factory(solver, initial, xp, mesh)

            if jax is not None:
                value, gradient = jax.jit(jax.value_and_grad(log_likelihood))(
                    module.asarray(rate, dtype=module.float64)
                )
            elif backend == "torch":
                parameter = module.tensor(
                    rate, dtype=module.float64, requires_grad=True
                )
                value = log_likelihood(parameter)
                value.backward()
                gradient = parameter.grad
                value, gradient = value.detach(), gradient.detach()
            else:
                difference_step = 1e-5
                value = log_likelihood(rate)
                gradient = (
                    log_likelihood(rate + difference_step)
                    - log_likelihood(rate - difference_step)
                ) / (2 * difference_step)
            value_errors.append(abs(float(value) - exact_log_likelihood))
            gradient_errors.append(abs(float(gradient) - exact_gradient))
        assert len(set(mesh_identities)) == 3
        assert value_errors[0] / value_errors[1] > 12
        assert value_errors[1] / value_errors[2] > 12
        assert gradient_errors[0] / gradient_errors[1] > 12
        assert gradient_errors[1] / gradient_errors[2] > 12
        assert value_errors[-1] < 3e-6
        assert gradient_errors[-1] < 2e-5
