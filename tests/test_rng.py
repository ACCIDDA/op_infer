"""Replayable host addresses, independent derivation vectors and fail-closed banks."""

from __future__ import annotations

import importlib
import pickle
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from op_infer.adapters.rng import jax_key, torch_generators
from op_infer.rng import RandomKey, SimulationStreams

if TYPE_CHECKING:
    from types import ModuleType


def test_independent_derivation_vector() -> None:
    """Pin SHA-256 framing, word order and native state against a separate oracle."""
    key = RandomKey(seed=2026, stream="simulation", index=8)
    expected_path = (
        0x4F504931,
        45933618,
        1307823271,
        3073206642,
        1168739056,
        187248144,
        3576842475,
        1889024871,
        1338759189,
        8,
        0,
    )
    assert key.seed_sequence().spawn_key == expected_path
    np.testing.assert_array_equal(
        key.seed_sequence().generate_state(4),
        [398470335, 3880169948, 3660260011, 4115849871],
    )
    np.testing.assert_array_equal(
        key.numpy_generator().bit_generator.random_raw(4),
        [
            13920657339022771101,
            813882185005463392,
            4381985055935216273,
            5180042362490322184,
        ],
    )
    assert key.seed_sequence().entropy == 2026
    assert key.seed_sequence().n_children_spawned == 0
    assert isinstance(key.numpy_generator().bit_generator, np.random.PCG64DXSM)


@pytest.mark.parametrize("seed", [0, 2026, 2**128 - 1])
@pytest.mark.parametrize("index", [0, 8, 2**64 - 1])
def test_serialization_and_replay(seed: int, index: int) -> None:
    """Maximum-width addresses survive JSON/pickle and replay fresh native state."""
    key = RandomKey(seed=seed, stream="fit/chain-0.draw:1", index=index)
    restored = RandomKey.model_validate_json(key.model_dump_json())
    assert restored == key
    assert pickle.loads(pickle.dumps(key)) == key  # noqa: S301 -- own trusted bytes
    expected = key.numpy_generator().normal(size=20)
    np.testing.assert_array_equal(expected, restored.numpy_generator().normal(size=20))
    assert key.seed_sequence() is not key.seed_sequence()


def test_default_roles_and_consumption_are_separate() -> None:
    """Parameter consumption cannot change simulation draws, or advance metadata."""
    streams = SimulationStreams(seed=2026, index=8)
    assert streams.parameters != streams.simulation
    parameter_rng = streams.parameters.numpy_generator()
    simulation_rng = streams.simulation.numpy_generator()
    parameter_rng.normal(size=10_000)
    expected = streams.simulation.numpy_generator().normal(size=16)
    np.testing.assert_array_equal(simulation_rng.normal(size=16), expected)
    assert not np.array_equal(
        streams.parameters.numpy_generator().normal(size=16), expected
    )
    assert streams == SimulationStreams.model_validate_json(streams.model_dump_json())


def test_indexed_replay_does_not_depend_on_order_or_batch_length() -> None:
    """Extending/reordering/chunking task banks does not change a recorded task."""
    indices = [8, 2, 19, 0]
    initial = {
        i: RandomKey(seed=2026, stream="simulation", index=i)
        .numpy_generator()
        .normal(size=3)
        for i in indices
    }
    for i in [19, 99, 2, 8, 100, 0]:
        if i in initial:
            replay = RandomKey(seed=2026, stream="simulation", index=i)
            np.testing.assert_array_equal(
                initial[i], replay.numpy_generator().normal(size=3)
            )


def test_native_generator_progresses_but_relowering_restarts() -> None:
    """A key records a stream start; the caller owns its mutable generator state."""
    key = RandomKey(seed=2026, stream="parameters")
    generator = key.numpy_generator()
    first = generator.integers(0, 2**31, size=5)
    second = generator.integers(0, 2**31, size=5)
    joined = key.numpy_generator().integers(0, 2**31, size=10)
    np.testing.assert_array_equal(np.concatenate((first, second)), joined)
    assert not np.array_equal(first, second)


def test_complete_seed_and_index_bits_contribute() -> None:
    """High bits are mixed before backend key-width reduction, never truncated."""
    keys = [
        RandomKey(seed=9, stream="simulation", index=3),
        RandomKey(seed=9 + 2**32, stream="simulation", index=3),
        RandomKey(seed=9 + 2**64, stream="simulation", index=3),
        RandomKey(seed=9 + 2**127, stream="simulation", index=3),
        RandomKey(seed=9, stream="simulation", index=3 + 2**32),
        RandomKey(seed=9, stream="simulation", index=3 + 2**63),
        RandomKey(seed=9, stream="Simulation", index=3),
    ]
    states = {tuple(key.seed_sequence().generate_state(4)) for key in keys}
    assert len(states) == len(keys)


def test_numpy_lowering_leaves_global_state_untouched() -> None:
    """Snapshot the legacy global state used by unrelated consumers."""
    before = np.random.get_state(legacy=False)  # noqa: NPY002 -- audit global state
    RandomKey(seed=2026, stream="simulation").numpy_generator().normal(size=20)
    after = np.random.get_state(legacy=False)  # noqa: NPY002 -- audit global state
    assert pickle.dumps(before) == pickle.dumps(after)


@pytest.mark.parametrize("field", ["seed", "index"])
@pytest.mark.parametrize("value", [True, 1.5, "1", -1, None, np.int64(1)])
def test_invalid_integers_are_not_coerced(field: str, value: object) -> None:
    """Reject bool/fractional/foreign integer metadata rather than changing identity."""
    payload: dict[str, object] = {"seed": 2026, "stream": "simulation", field: value}
    with pytest.raises(ValidationError):
        RandomKey.model_validate(payload)


@pytest.mark.parametrize("stream", ["", " ", "a\n", "a b", "1a", "é", "a\x00b"])
def test_invalid_stream_names(stream: str) -> None:
    """Stream encoding is explicit ASCII with complete anchored validation."""
    with pytest.raises(ValidationError):
        RandomKey(seed=2026, stream=stream)


@pytest.mark.parametrize(
    "patch",
    [
        {"seed": 2**128},
        {"index": 2**64},
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"unexpected": "silently ignored"},
    ],
)
def test_overflow_unknown_schema_and_fields_fail(patch: dict[str, object]) -> None:
    """Fail before native lowering when a saved address cannot be interpreted."""
    payload = {"seed": 2026, "stream": "simulation", **patch}
    with pytest.raises(ValidationError):
        RandomKey.model_validate(payload)


def test_frozen_and_revalidated_records() -> None:
    """Even unchecked model_construct/model_copy values are checked before lowering."""
    key = RandomKey(seed=2026, stream="simulation")
    with pytest.raises(ValidationError):
        key.index = 3
    invalid = key.model_copy(update={"index": -1})
    with pytest.raises(ValidationError):
        invalid.numpy_generator()
    streams = SimulationStreams.model_construct(seed=2026, index=-1)
    with pytest.raises(ValidationError):
        _ = streams.parameters
    with pytest.raises(ValidationError):
        _ = streams.simulation


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_missing_extras_are_actionable(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Optional native packages are imported only by the requested lowering."""
    original = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> ModuleType:
        if name.split(".", maxsplit=1)[0] == backend:
            raise ModuleNotFoundError(name)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    key = RandomKey(seed=2026, stream="simulation")
    lower = (
        partial(jax_key, key) if backend == "jax" else partial(torch_generators, (key,))
    )
    with pytest.raises(ImportError, match=rf"op_infer\[{backend}\]"):
        lower()


def test_torch_collision_is_rejected_before_optional_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated task cannot become two supposedly independent replicates."""

    def unexpected_import(name: str) -> ModuleType:
        pytest.fail(f"Backend import occurred before validation: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    key = RandomKey(seed=2026, stream="simulation")
    with pytest.raises(ValueError, match="effective seeds collide"):
        torch_generators((key, key))


def test_distinct_addresses_with_a_real_torch_seed_collision() -> None:
    """Retain a counterexample found by an independent two-role SeedSequence scan."""
    keys = (
        RandomKey(seed=2026, stream="simulation", index=21297),
        RandomKey(seed=2026, stream="parameters", index=68078),
    )
    assert keys[0] != keys[1]
    assert [int(key.seed_sequence().generate_state(1)[0]) for key in keys] == [
        1998271142,
        1998271142,
    ]
    assert not np.array_equal(
        keys[0].seed_sequence().generate_state(4),
        keys[1].seed_sequence().generate_state(4),
    )
    with pytest.raises(ValueError, match="effective seeds collide"):
        torch_generators(keys)


@pytest.mark.parametrize("bank", [[1], (1,), "simulation", None])
def test_invalid_torch_banks(bank: object) -> None:
    """Mutable or invalid key collections cannot reach a native adapter."""
    with pytest.raises(TypeError, match="tuple of RandomKey"):
        torch_generators(bank)  # type: ignore[arg-type]
