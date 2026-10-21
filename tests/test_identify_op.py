"""Actual compiler/engine artifacts feed scoped structural and local reports."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from op_infer.adapters.op import structural_model_reference
from op_infer.adapters.op_identify import (
    information_from_sensitivities,
    scaling_assessment,
    sensitivity_execution_reference,
)
from op_infer.identify import InformationContext, ScalingAssessment, local_spectrum
from op_infer.observe import IIDObservation
from op_infer.posterior import ArtifactReference


def _ref(name: str) -> ArtifactReference:
    return ArtifactReference.from_bytes(name.encode(), kind="toy-v1")


def test_existing_compiler_scaling_report_keeps_unassessed_and_zero_distinct() -> None:
    """A missing analysis is not a zero-rank result; reported scalings remain scoped."""
    compiler = pytest.importorskip("op_system")
    basic = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "-rate*x"},
        "observables": {"measurement": {"expr": "x"}},
    }
    unassessed = scaling_assessment(compiler.compile_spec(basic))
    assert unassessed.status == "unassessed"
    assert unassessed.action_rank is None
    checked = scaling_assessment(compiler.compile_spec(basic, detect_gauges=True))
    assert checked.status == "reported"
    assert checked.action_rank == 0
    assert checked.detection_requested
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "rate*x*(1-x/K)"},
        "observables": {"measurement": {"expr": "gain*x"}},
        "gauges": [
            {"name": "scale", "states": {"x": 1}, "parameters": {"K": 1, "gain": -1}}
        ],
    }
    compiled = compiler.compile_spec(spec)
    report = scaling_assessment(compiled, expected=structural_model_reference(compiled))
    assert report.action_rank == report.parameter_action_rank == 1
    assert report.verification is not None
    assert report.verification.structural_homogeneity_checked
    assert report.verification.seed == 19471
    assert report.verification.observables == ("measurement",)
    assert ScalingAssessment.model_validate_json(report.model_dump_json()) == report
    fixed = compiler.compile_spec(spec, gauge_fix={"K": 1.0})
    unsupported = scaling_assessment(fixed)
    assert unsupported.status == "unsupported"
    assert unsupported.action_rank is None
    with pytest.raises(ValueError, match="identity mismatch"):
        scaling_assessment(compiled, expected=_ref("different"))


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


def _solution(module: Any, backend: str) -> tuple[Any, Any]:  # noqa: ANN401
    compiler = pytest.importorskip("op_system")
    pytest.importorskip("op_system.symbolic")
    engine = pytest.importorskip("op_engine.sensitivities")
    solver = pytest.importorskip("op_engine.core_solver")
    compiled = compiler.compile_spec(
        {
            "kind": "expr",
            "state": ["x", "z"],
            "equations": {"x": "-rate*x", "z": "-rate*z"},
            "observables": {"measurement": {"expr": "gain*(x+2*z)"}},
        }
    )
    system = engine.SensitivitySystem(
        compiled.eval_fn,
        compiler.symbolic_derivatives(compiled),
        initial_parameters=("amplitude",),
    )
    initial = engine.SensitivityInitial(
        module.asarray([2.0, 4.0], dtype=module.float64),
        module.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=module.float64),
    )
    parameters = {
        name: module.asarray(value, dtype=module.float64)
        for name, value in zip(system.parameter_names, (0.7, 1.5, 2.0), strict=True)
    }
    solution = engine.solve_sensitivities(
        system,
        parameters,
        initial,
        np.asarray([0.0, 0.5, 1.0]),
        options=engine.SensitivityOptions(
            store_history=True,
            solver=solver.RunConfig(
                method="rk4",
                fixed_max_step=0.0625,
                fixed_loop="scan" if backend == "jax" else "unroll",
            ),
        ),
    )
    return compiled, solution


def _context(
    compiled: object,
    solution: Any,  # noqa: ANN401
    observation: IIDObservation,
) -> InformationContext:
    return InformationContext(
        source_model=structural_model_reference(compiled),
        design=_ref("times-and-masks"),
        parameter_point=_ref("rate,gain,amplitude=.7,1.5,2"),
        initialization=_ref("y0=amplitude*(1,2)"),
        observation=observation.reference,
        numerical=sensitivity_execution_reference(solution),
        qualification=_ref("analytic-decay-check"),
        parameter_names=solution.parameter_names,
        parameter_units=("1/time", "measurement/state", "state"),
        observable_names=solution.observable_names,
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_measured_sensitivities_include_initial_and_affine_noise_map(
    backend: str,
) -> None:
    """An independent mean derivative detects source/measurement omissions."""
    module, precision = _backend(backend)
    with precision:
        compiled, solution = _solution(module, backend)
        observation = IIDObservation(family="normal", scale=0.4, slope=2.0, offset=1.0)
        context = _context(compiled, solution, observation)
        information = information_from_sensitivities(
            solution, observation, context=context
        )
        times = np.array([0.0, 0.5, 1.0])
        decay = np.exp(-0.7 * times)
        jacobian = np.stack((-15 * times * decay, 10 * decay, 7.5 * decay), axis=1)
        expected = 25 * jacobian.T @ jacobian
        actual: Any = information.matrix
        if backend == "torch":
            actual = actual.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-6)
        report = local_spectrum(information, rtol=1e-7, atol=1e-10)
        assert report.local_rank == 2
        assert report.n_observed == 3
        assert report.context.initialization == context.initialization
        with pytest.raises(ValueError, match="context mismatch"):
            information_from_sensitivities(
                solution,
                observation,
                context=context.model_copy(update={"numerical": _ref("wrong mesh")}),
            )
        with pytest.raises(ValueError, match="context mismatch"):
            information_from_sensitivities(
                solution,
                observation,
                context=context.model_copy(
                    update={
                        "parameter_names": tuple(reversed(solution.parameter_names))
                    }
                ),
            )
        with pytest.raises(NotImplementedError, match="uncensored iid Normal"):
            information_from_sensitivities(
                solution,
                IIDObservation(family="normal", lower_limit=0.0),
                context=context,
            )
