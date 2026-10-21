"""Native CPU replay/JIT qualification; mandatory when qualify-rng is selected."""

import importlib

import numpy as np
import pytest

from op_infer.adapters.rng import jax_key, torch_generators
from op_infer.rng import RandomKey, SimulationStreams


def test_jax_explicit_algorithm_and_native_key_vector() -> None:
    """Typed Threefry bits are invariant to JAX's global PRNG default."""
    jax = pytest.importorskip("jax")
    with jax.default_prng_impl("rbg"):
        key = jax_key(RandomKey(seed=2026, stream="simulation", index=8))
        assert jax.random.key_impl(key) == jax.random.key_impl(
            jax.random.key(0, impl="threefry2x32")
        )
        np.testing.assert_array_equal(jax.random.key_data(key), [398470335, 3880169948])


def test_jax_jit_vmap_replay_and_order() -> None:
    """Task keys work as native inputs and remain stable after batch extension."""
    jax = pytest.importorskip("jax")
    xp = importlib.import_module("jax.numpy")
    keys = [
        jax_key(RandomKey(seed=2026, stream="simulation", index=i))
        for i in [8, 2, 19, 0, 99]
    ]
    draw = jax.jit(lambda k: jax.random.normal(k, shape=(4,)))
    batch = jax.jit(jax.vmap(draw))
    original = batch(xp.stack(keys[:4]))
    extended = batch(xp.stack(keys))
    reverse = batch(xp.stack(keys[:4][::-1]))
    np.testing.assert_array_equal(original, extended[:4])
    np.testing.assert_array_equal(original, reverse[::-1])
    np.testing.assert_array_equal(original[0], draw(keys[0]))
    np.testing.assert_array_equal(original[0], draw(keys[0]))
    assert len({tuple(row) for row in np.asarray(original)}) == 4


def test_jax_prior_consumption_and_high_bits() -> None:
    """Role separation and complete host entropy survive optional native lowering."""
    jax = pytest.importorskip("jax")
    streams = SimulationStreams(seed=2026, index=8)
    parameter_key = jax_key(streams.parameters)
    simulation_key = jax_key(streams.simulation)
    expected = jax.random.uniform(simulation_key, shape=(8,))
    jax.random.normal(parameter_key, shape=(1000,))
    np.testing.assert_array_equal(
        expected, jax.random.uniform(jax_key(streams.simulation), shape=(8,))
    )
    assert not np.array_equal(
        jax.random.key_data(parameter_key), jax.random.key_data(simulation_key)
    )
    altered = jax_key(
        RandomKey(seed=2026 + 2**127, stream="simulation", index=8 + 2**63)
    )
    assert not np.array_equal(
        jax.random.key_data(simulation_key), jax.random.key_data(altered)
    )


def test_torch_replay_order_and_global_state() -> None:
    """Local generators replay without mutating Torch's process-global RNG."""
    torch = pytest.importorskip("torch")
    keys = tuple(
        RandomKey(seed=2026, stream="simulation", index=i) for i in [8, 2, 19, 0]
    )
    before = torch.get_rng_state().clone()
    first = torch_generators(keys)
    assert first[0].initial_seed() == 398470335
    assert str(first[0].device) == "cpu"
    expected = [torch.randn(4, generator=g) for g in first]
    replay = [torch.randn(4, generator=g) for g in torch_generators(keys)]
    reverse = [torch.randn(4, generator=g) for g in torch_generators(keys[::-1])]
    extended = torch_generators(
        (*keys, RandomKey(seed=2026, stream="simulation", index=99))
    )
    for i, draws in enumerate(expected):
        assert torch.equal(draws, replay[i])
        assert torch.equal(draws, reverse[-i - 1])
        assert torch.equal(draws, torch.randn(4, generator=extended[i]))
    assert torch.equal(before, torch.get_rng_state())
    assert torch_generators(()) == ()


def test_torch_parameter_simulation_separation_and_gradient() -> None:
    """Parameter sampling cannot consume simulation state or detach model values."""
    torch = pytest.importorskip("torch")
    streams = SimulationStreams(seed=2026, index=8)
    parameter_rng, simulation_rng = torch_generators(
        (streams.parameters, streams.simulation)
    )
    torch.randn(1000, generator=parameter_rng)
    noise = torch.randn(8, generator=simulation_rng, dtype=torch.float64)
    (_, replay_rng) = torch_generators((streams.parameters, streams.simulation))
    assert torch.equal(noise, torch.randn(8, generator=replay_rng, dtype=torch.float64))
    rate = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
    result = torch.sum((rate * noise) ** 2)
    result.backward()
    assert float(rate.grad) == pytest.approx(4 * float(torch.sum(noise**2)))


def test_torch_rejects_real_effective_seed_collision() -> None:
    """The two distinct task addresses really alias under CPU manual_seed."""
    torch = pytest.importorskip("torch")
    keys = (
        RandomKey(seed=2026, stream="simulation", index=21297),
        RandomKey(seed=2026, stream="parameters", index=68078),
    )
    native = [
        torch.Generator(device="cpu").manual_seed(
            int(key.seed_sequence().generate_state(1)[0])
        )
        for key in keys
    ]
    assert torch.equal(
        torch.randn(8, generator=native[0]), torch.randn(8, generator=native[1])
    )
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="effective seeds collide"):
        torch_generators(keys)
    assert torch.equal(before, torch.get_rng_state())


def test_torch_explicit_cpu_device_keeps_existing_addressed_stream() -> None:
    """The optional device argument preserves the established CPU seed lowering."""
    torch = pytest.importorskip("torch")
    keys = (RandomKey(seed=2026, stream="simulation", index=8),)
    default = torch_generators(keys)[0]
    explicit = torch_generators(keys, device=torch.device("cpu"))[0]
    assert torch.equal(
        torch.randn(16, generator=default), torch.randn(16, generator=explicit)
    )
    with pytest.raises(ValueError, match="CPU and CUDA"):
        torch_generators((), device="meta")
