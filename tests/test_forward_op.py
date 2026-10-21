"""Independent fixed-decay references through the actual compiler/engine adapter."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

from op_infer.adapters.op import (
    FixedForwardSettings,
    compiled_forward,
    compiled_observation_reference,
    fixed_forward_mesh_reference,
    fixed_forward_settings_reference,
    structural_model_reference,
)
from op_infer.forward import CallableForward, ForwardDesign
from op_infer.posterior import Provenance
from op_infer.rng import RandomKey, SimulationStreams
from tests.test_provenance import provenance, reference

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

compiler = pytest.importorskip("op_system")
pytest.importorskip("op_system.identity")
pytest.importorskip("op_engine")


def fixture(  # noqa: PLR0913 - independent static execution choices
    *,
    method: Literal["euler", "heun", "rk4", "dopri5"] = "rk4",
    loop: Literal["auto", "scan", "unroll"] = "auto",
    checkpoint: bool = False,
    dtype: Literal["float32", "float64"] = "float64",
    max_step: float = 0.05,
    spec: dict[str, Any] | None = None,
) -> tuple[Any, FixedForwardSettings, Provenance]:
    """Reference one linear decay with a separately declared observed output."""
    compiled = compiler.compile_spec(
        spec
        or {
            "kind": "expr",
            "state": ["x"],
            "equations": {"x": "-rate*x"},
            "observables": {"value": {"expr": "scale*x"}},
        }
    )
    settings = FixedForwardSettings(
        times=(0.0, 0.5, 1.0),
        method=method,
        max_step=max_step,
        fixed_loop=loop,
        checkpoint=checkpoint,
        dtype=dtype,
    )
    base = provenance().with_structural_model(structural_model_reference(compiled))
    assert base.numerical is not None
    numerical = base.numerical.model_copy(
        update={
            "solver": fixed_forward_settings_reference(settings),
            "time_mesh": fixed_forward_mesh_reference(settings),
            "observation": compiled_observation_reference(compiled, ("value",)),
        }
    )
    return (
        compiled,
        settings,
        Provenance.model_validate({**base.model_dump(), "numerical": numerical}),
    )


@contextmanager
def backend(name: str) -> Iterator[tuple[Any, Any]]:
    """Require a real selected CPU namespace, with scoped JAX double precision."""
    if name == "jax":
        jax = pytest.importorskip("jax")
        xp = pytest.importorskip("jax.numpy")
        with jax.enable_x64():
            yield xp, jax
    elif name == "torch":
        yield pytest.importorskip("torch"), None
    else:
        yield np, None


def adapter(
    compiled: Any,  # noqa: ANN401 - optional compiler
    settings: FixedForwardSettings,
    expected: Provenance,
    xp: Any,  # noqa: ANN401 - optional compiler/native namespace
) -> CallableForward:
    """Pass the initial value as a traced parameter rather than closing over it."""

    def initial(theta: Mapping[str, object], _design: object) -> object:
        return xp.reshape(theta["initial"], (1,))

    return compiled_forward(
        compiled,
        settings=settings,
        initial_state=initial,
        parameter_names=("rate", "scale", "initial"),
        output_names=("value",),
        provenance=expected,
    )


def host(value: Any) -> Any:  # noqa: ANN401
    """Snapshot results only after native differentiation."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.parametrize("name", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_values_and_parameter_initial_gradients(
    name: str,
    method: Literal["euler", "heun", "rk4", "dopri5"],
    dtype: Literal["float32", "float64"],
) -> None:
    """The analytic value and both gradients bound the independently known error."""
    compiled, settings, expected = fixture(
        method=method,
        max_step=0.005 if method in {"euler", "heun"} else 0.05,
        dtype=dtype,
    )
    with backend(name) as (xp, jax):
        forward = adapter(compiled, settings, expected, xp)
        design = ForwardDesign(expected.design, None)
        key = RandomKey(seed=1, stream="simulation")

        def value(rate: Any, initial: Any) -> Any:  # noqa: ANN401
            return forward(
                {
                    "rate": rate,
                    "scale": xp.asarray(3.0, dtype=getattr(xp, dtype)),
                    "initial": initial,
                },
                design,
                key,
            )["value"]

        rate = xp.asarray(0.7, dtype=getattr(xp, dtype))
        initial = xp.asarray(2.0, dtype=getattr(xp, dtype))
        if jax is not None:
            values = jax.jit(value)(rate, initial)
            grad_rate, grad_initial = jax.jit(
                jax.grad(lambda r, a: xp.sum(value(r, a)), argnums=(0, 1))
            )(rate, initial)
        elif name == "torch":
            rate.requires_grad_()
            initial.requires_grad_()
            values = value(rate, initial)
            xp.sum(values).backward()
            grad_rate, grad_initial = rate.grad, initial.grad
        else:
            values = value(rate, initial)
            step = 0.002 if dtype == "float32" else 1e-5
            grad_rate = (
                np.sum(value(rate + step, initial))
                - np.sum(value(rate - step, initial))
            ) / (2 * step)
            grad_initial = (
                np.sum(value(rate, initial + step))
                - np.sum(value(rate, initial - step))
            ) / (2 * step)
        times = np.asarray(settings.times)
        exact = 6.0 * np.exp(-0.7 * times)
        tolerance = {"euler": 0.008, "heun": 2e-5, "rk4": 2e-7, "dopri5": 1e-9}[method]
        if dtype == "float32":
            tolerance = max(tolerance, 2e-5)
        gradient_tolerance = (
            max(2 * tolerance, 0.002)
            if dtype == "float32" and name == "numpy"
            else 2 * tolerance
        )
        assert values.dtype == initial.dtype
        np.testing.assert_allclose(host(values), exact, atol=tolerance, rtol=0)
        np.testing.assert_allclose(
            host(grad_rate), -np.sum(times * exact), atol=gradient_tolerance, rtol=0
        )
        np.testing.assert_allclose(
            host(grad_initial), np.sum(exact) / 2, atol=gradient_tolerance, rtol=0
        )


def test_bounded_jax_batch_reuses_trace_and_has_no_state_history() -> None:
    """Two parameter/initial values share a compiled scan and return only outputs."""
    compiled, settings, expected = fixture(loop="scan", checkpoint=True)
    with backend("jax") as (xp, jax):
        forward = adapter(compiled, settings, expected, xp)
        design = ForwardDesign(expected.design, None)
        key = RandomKey(seed=1, stream="simulation")
        traces: list[None] = []

        def values(rate: Any, initial: Any) -> Any:  # noqa: ANN401
            traces.append(None)
            return forward(
                {"rate": rate, "scale": xp.asarray(3.0), "initial": initial},
                design,
                key,
            )["value"]

        native = jax.jit(jax.vmap(values))
        rates = xp.asarray([0.4, 0.9])
        initials = xp.asarray([1.0, 2.0])
        result = native(rates, initials)
        traced = len(traces)
        np.testing.assert_allclose(
            result,
            3
            * np.array([1.0, 2.0])[:, None]
            * np.exp(-np.array([0.4, 0.9])[:, None] * np.array(settings.times)),
            atol=3e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            native(rates + 0.1, initials),
            3
            * np.array([1.0, 2.0])[:, None]
            * np.exp(-np.array([0.5, 1.0])[:, None] * np.array(settings.times)),
            atol=3e-7,
            rtol=0,
        )
        assert len(traces) == traced
        graph = jax.make_jaxpr(values)(rates[0], initials[0])
        scans = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == "scan"]
        assert len(scans) == 2
        assert scans[0].outvars[0].aval.shape == (1, 1)
        # Carry is the current state. Stacked outputs are scalar observations,
        # never a trajectory with the original state/sample axes.
        assert all(
            variable.aval.shape in {(1, 1), (1,)}
            for scan in scans
            for variable in scan.outvars
        )
        assert any(
            eq.primitive.name.startswith("remat")
            for eq in scans[0].params["jaxpr"].jaxpr.eqns
        )
        gradient = jax.jit(jax.grad(lambda rate: xp.sum(values(rate, initials[0]))))(
            rates[0]
        )
        np.testing.assert_allclose(
            gradient,
            -np.sum(
                3 * np.array(settings.times) * np.exp(-0.4 * np.array(settings.times))
            ),
            atol=3e-7,
            rtol=0,
        )


def test_forced_jax_unroll_has_no_scan_and_retains_gradients() -> None:
    """Explicit unroll retains the requested graph and dynamic parameters."""
    compiled, settings, expected = fixture(loop="unroll")
    with backend("jax") as (xp, jax):
        forward = adapter(compiled, settings, expected, xp)

        def value(rate: Any) -> Any:  # noqa: ANN401
            return xp.sum(
                forward(
                    {
                        "rate": rate,
                        "scale": xp.asarray(3.0),
                        "initial": xp.asarray(2.0),
                    },
                    ForwardDesign(expected.design, None),
                    RandomKey(seed=1, stream="simulation"),
                )["value"]
            )

        rate = xp.asarray(0.7)
        assert all(
            eq.primitive.name != "scan" for eq in jax.make_jaxpr(value)(rate).jaxpr.eqns
        )
        result, gradient = jax.jit(jax.value_and_grad(value))(rate)
        times = np.asarray(settings.times)
        np.testing.assert_allclose(
            result, np.sum(6 * np.exp(-0.7 * times)), atol=3e-7, rtol=0
        )
        np.testing.assert_allclose(
            gradient, -np.sum(6 * times * np.exp(-0.7 * times)), atol=3e-7, rtol=0
        )


@pytest.mark.parametrize(
    "mutation", ["model", "solver", "mesh", "observation", "parameter", "output"]
)
def test_constructor_refuses_mismatched_identity_or_layout(mutation: str) -> None:
    """Changing a compiled output or numerical policy blocks the old contract."""
    compiled, settings, expected = fixture()
    parameters: tuple[str, ...] = ("rate", "scale", "initial")
    outputs = ("value",)
    if mutation == "model":
        expected = expected.with_structural_model(reference("different model"))
    elif mutation in {"solver", "mesh", "observation"}:
        assert expected.numerical is not None
        key = {"solver": "solver", "mesh": "time_mesh", "observation": "observation"}[
            mutation
        ]
        expected = expected.model_copy(
            update={
                "numerical": expected.numerical.model_copy(
                    update={key: reference("different contract")}
                )
            }
        )
    elif mutation == "parameter":
        parameters = ("rate", "initial")
    else:
        outputs = ("missing",)
    with pytest.raises(
        ValueError, match=r"identity mismatch|layout omits|latent observables"
    ):
        compiled_forward(
            compiled,
            settings=settings,
            initial_state=lambda _p, _d: np.ones(1),
            parameter_names=parameters,
            output_names=outputs,
            provenance=expected,
        )


def test_explicit_work_bound_and_unavailable_scan_are_not_ignored() -> None:
    """An oversized setup is rejected before the engine allocates its partition."""
    too_many = FixedForwardSettings(
        times=(0.0, 1e9), max_step=1e-9, max_internal_steps=10
    )
    with pytest.raises(ValueError, match="max_internal_steps"):
        fixed_forward_mesh_reference(too_many)
    compiled, settings, expected = fixture(loop="scan")
    forward = adapter(compiled, settings, expected, np)
    with pytest.raises(ValueError, match="no namespace loop adapter"):
        forward(
            {
                "rate": np.asarray(0.7),
                "scale": np.asarray(3.0),
                "initial": np.asarray(2.0),
            },
            ForwardDesign(expected.design, None),
            RandomKey(seed=1, stream="simulation"),
        )


def test_state_precision_is_part_of_numerical_identity_and_checked() -> None:
    """A precision change preserves model structure but invalidates numerical reuse."""
    compiled, settings, expected = fixture()
    _, settings32, expected32 = fixture(dtype="float32")
    assert expected.structural_model == expected32.structural_model
    assert fixed_forward_settings_reference(
        settings
    ) != fixed_forward_settings_reference(settings32)
    assert expected.identity != expected32.identity
    forward = adapter(compiled, settings, expected, np)
    with pytest.raises(ValueError, match="dtype differs"):
        forward(
            {
                "rate": np.asarray(0.7, dtype=np.float32),
                "scale": np.asarray(3.0, dtype=np.float32),
                "initial": np.asarray(2.0, dtype=np.float32),
            },
            ForwardDesign(expected.design, None),
            RandomKey(seed=1, stream="simulation"),
        )


def test_unqualified_operator_composition_is_rejected() -> None:
    """An engine operator cannot silently disappear from the adapted equation."""
    compiled, settings, expected = fixture()
    compiled = replace(compiled, operators=(object(),))
    with pytest.raises(NotImplementedError, match="operators/history"):
        adapter(compiled, settings, expected, np)


def test_shaped_reduction_retains_only_declared_outputs() -> None:
    """A many-state model returns a scalar observed sum at each requested time."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "site", "coords": ["west", "east"]}],
        "state": ["x[site]", "unused[site]"],
        "equations": {"x[site]": "-rate*x[site]", "unused[site]": "0.0"},
        "observables": {
            "value": {
                "expr": "scale*x[site]",
                "reduction": {"kind": "sum", "axes": ["site"]},
            }
        },
    }
    compiled, settings, expected = fixture(spec=spec)
    initial = np.array([1.0, 2.0, 100.0, 200.0])
    forward = compiled_forward(
        compiled,
        settings=settings,
        initial_state=lambda _p, _d: initial,
        parameter_names=("rate", "scale"),
        output_names=("value",),
        provenance=expected,
    )
    result = forward(
        {"rate": np.asarray(0.7), "scale": np.asarray(3.0)},
        ForwardDesign(expected.design, None),
        RandomKey(seed=1, stream="simulation"),
    )
    np.testing.assert_allclose(
        host(result["value"]),
        9 * np.exp(-0.7 * np.array(settings.times)),
        atol=3e-7,
        rtol=0,
    )
    np.testing.assert_array_equal(initial, [1, 2, 100, 200])
    assert host(result["value"]).shape == (3,)


class _ForwardWorld:
    def __init__(self, forward: CallableForward, design: ForwardDesign) -> None:
        self.forward = forward
        self.design = design

    def generate(
        self, config: dict[str, Any]
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """Use explicit stable replicate addresses and separate random roles."""
        streams = SimulationStreams(seed=47, index=config["replicate"])
        theta = {
            "rate": np.asarray(streams.parameters.numpy_generator().uniform(0.2, 1.0)),
            "scale": np.asarray(3.0),
            "initial": np.asarray(2.0),
        }
        return theta, self.forward(theta, self.design, streams.simulation)


class _ForwardScorer:
    @staticmethod
    def score(
        truth: Mapping[str, object],
        observed: Mapping[str, object],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Compare actual orchestrated outputs to independent continuous decay."""
        del config
        expected = 6 * np.exp(-float(host(truth["rate"])) * np.array([0.0, 0.5, 1.0]))
        return {"error": float(np.max(np.abs(host(observed["value"]) - expected)))}


def test_real_trade_study_simulator_and_guarded_checkpoint(tmp_path: Path) -> None:
    """The delivered forward adapter runs through the real Simulator contract."""
    pytest.importorskip("trade_study")
    trade = importlib.import_module("trade_study")
    compiled, settings, expected = fixture()
    world = _ForwardWorld(
        adapter(compiled, settings, expected, np), ForwardDesign(expected.design, None)
    )
    arguments = {
        "world": world,
        "scorer": _ForwardScorer(),
        "grid": [{"replicate": i} for i in [7, 2, 9]],
        "observables": [trade.Observable("error", trade.Direction.MINIMIZE)],
        "checkpoint_path": tmp_path / "grid.sqlite",
        "checkpoint_key": expected.identity,
    }
    table = trade.run_grid(**arguments)
    assert table.scores.shape == (3, 1)
    assert np.max(table.scores) < 3e-7
    replay = trade.run_grid(**arguments)
    np.testing.assert_array_equal(replay.scores, table.scores)
    assert all(row["recovered"] for row in replay.metadata)
    arguments["checkpoint_key"] = expected.with_structural_model(
        reference("changed model")
    ).identity
    with pytest.raises(ValueError, match="Incompatible grid checkpoint"):
        trade.run_grid(**arguments)


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
def test_grouped_scan_keeps_ragged_mesh_and_time_observations(method: str) -> None:
    """Irregular intervals preserve the eager discrete map and its gradients."""
    compiled, original, expected = fixture(method=method, loop="scan")  # type: ignore[arg-type]
    settings = FixedForwardSettings(
        times=(0.0, 0.1, 0.35, 0.55, 0.85),
        max_step=0.1,
        method=original.method,
        fixed_loop="scan",
    )
    assert expected.numerical is not None
    numerical = expected.numerical.model_copy(
        update={
            "solver": fixed_forward_settings_reference(settings),
            "time_mesh": fixed_forward_mesh_reference(settings),
        }
    )
    expected = Provenance.model_validate(
        {**expected.model_dump(), "numerical": numerical}
    )
    with backend("jax") as (xp, jax):
        scanned = adapter(compiled, settings, expected, xp)
        eager_settings = settings.model_copy(update={"fixed_loop": "unroll"})
        eager_numerical = numerical.model_copy(
            update={"solver": fixed_forward_settings_reference(eager_settings)}
        )
        eager_expected = Provenance.model_validate(
            {**expected.model_dump(), "numerical": eager_numerical}
        )
        eager = adapter(compiled, eager_settings, eager_expected, xp)
        design = ForwardDesign(expected.design, None)
        key = RandomKey(seed=73, stream="simulation")

        def values(forward: CallableForward, rate: Any) -> Any:  # noqa: ANN401
            return forward(
                {"rate": rate, "scale": xp.asarray(3.0), "initial": xp.asarray(2.0)},
                design,
                key,
            )["value"]

        parameter = xp.asarray(0.7)
        np.testing.assert_allclose(
            jax.jit(lambda r: values(scanned, r))(parameter),
            values(eager, parameter),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            jax.jit(jax.grad(lambda r: xp.sum(values(scanned, r))))(parameter),
            jax.grad(lambda r: xp.sum(values(eager, r)))(parameter),
            rtol=1e-12,
            atol=1e-12,
        )


def test_uniform_output_grid_has_bounded_top_level_scan_count() -> None:
    """Adding output times does not clone one differentiated solver per interval."""
    compiled, _original, expected = fixture(loop="scan", checkpoint=True)
    settings = FixedForwardSettings(
        times=tuple(float(t) for t in np.linspace(0, 1, 17)),
        max_step=1 / 64,
        fixed_loop="scan",
        checkpoint=True,
    )
    assert expected.numerical is not None
    numerical = expected.numerical.model_copy(
        update={
            "solver": fixed_forward_settings_reference(settings),
            "time_mesh": fixed_forward_mesh_reference(settings),
        }
    )
    expected = Provenance.model_validate(
        {**expected.model_dump(), "numerical": numerical}
    )
    with backend("jax") as (xp, jax):
        forward = adapter(compiled, settings, expected, xp)
        design = ForwardDesign(expected.design, None)
        key = RandomKey(seed=73, stream="simulation")

        def value(rate: Any) -> Any:  # noqa: ANN401
            return forward(
                {"rate": rate, "scale": xp.asarray(3.0), "initial": xp.asarray(2.0)},
                design,
                key,
            )["value"]

        graph = jax.make_jaxpr(value)(xp.asarray(0.7))
        scans = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == "scan"]
        assert len(scans) == 2
        result, gradient = jax.jit(jax.value_and_grad(lambda r: xp.sum(value(r))))(
            xp.asarray(0.7)
        )
        times = np.asarray(settings.times)
        np.testing.assert_allclose(
            result, np.sum(6 * np.exp(-0.7 * times)), rtol=0, atol=3e-7
        )
        np.testing.assert_allclose(
            gradient, -np.sum(6 * times * np.exp(-0.7 * times)), rtol=0, atol=3e-7
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_grouped_scan_preserves_vector_observables_and_dynamic_time(
    *, checkpoint: bool, dtype: Literal["float32", "float64"]
) -> None:
    """A clock-dependent vector output retains its declared spatial shape and AD."""
    compiler = pytest.importorskip("op_system")
    compiled = compiler.compile_spec(
        {
            "kind": "expr",
            "axes": [{"name": "site", "coords": ["a", "b"]}],
            "state": ["x[site]"],
            "equations": {"x[site]": "-rate*x[site]"},
            "observables": {"value": {"expr": "scale*x[site]+t", "axes": ["site"]}},
        }
    )
    _, _, expected = fixture(loop="scan")
    settings = FixedForwardSettings(
        times=tuple(float(t) for t in np.linspace(0, 1, 9)),
        max_step=1 / 32,
        fixed_loop="scan",
        checkpoint=checkpoint,
        dtype=dtype,
    )
    assert expected.numerical is not None
    numerical = expected.numerical.model_copy(
        update={
            "solver": fixed_forward_settings_reference(settings),
            "time_mesh": fixed_forward_mesh_reference(settings),
            "observation": compiled_observation_reference(compiled, ("value",)),
        }
    )
    expected = Provenance.model_validate(
        {
            **expected.model_dump(),
            "structural_model": structural_model_reference(compiled),
            "numerical": numerical,
        }
    )
    with backend("jax") as (xp, jax):

        def initial(theta: Mapping[str, object], _design: object) -> object:
            value: Any = theta["initial"]
            return xp.stack((value, 2 * value))

        forward = compiled_forward(
            compiled,
            settings=settings,
            initial_state=initial,
            parameter_names=("rate", "scale", "initial"),
            output_names=("value",),
            provenance=expected,
        )

        def values(rate: Any) -> Any:  # noqa: ANN401
            return forward(
                {
                    "rate": rate,
                    "scale": xp.asarray(3.0, dtype=getattr(xp, dtype)),
                    "initial": xp.asarray(1.0, dtype=getattr(xp, dtype)),
                },
                ForwardDesign(expected.design, None),
                RandomKey(seed=73, stream="simulation"),
            )["value"]

        times = np.asarray(settings.times)
        actual = jax.jit(values)(xp.asarray(0.7, dtype=getattr(xp, dtype)))
        np.testing.assert_allclose(
            actual,
            3 * np.array([1.0, 2.0])[None, :] * np.exp(-0.7 * times[:, None])
            + times[:, None],
            rtol=0,
            atol=2e-5 if dtype == "float32" else 3e-7,
        )
        gradient = jax.jit(jax.grad(lambda r: xp.sum(values(r))))(
            xp.asarray(0.7, dtype=getattr(xp, dtype))
        )
        np.testing.assert_allclose(
            gradient,
            -9 * np.sum(times * np.exp(-0.7 * times)),
            rtol=0,
            atol=2e-5 if dtype == "float32" else 3e-7,
        )
