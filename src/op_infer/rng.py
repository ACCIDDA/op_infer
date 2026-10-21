"""Explicit, backend-lowered random streams.

Scope: immutable seed/stream/index addresses, separate parameter and simulation
streams, and fresh NumPy generators. Optional JAX/Torch lowerings live in
``op_infer.adapters.rng``. Addresses are host metadata, not traced array inputs.

Estimator adapters and simulation-based calibration consume these task addresses.
This module wraps native RNGs; it does not implement a sampler or promise
bitwise-equivalent samples across backends, devices or dependency versions.
"""

import hashlib
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

__all__ = ["RandomKey", "SimulationStreams"]

type _Seed = Annotated[int, Field(ge=0, lt=2**128)]
type _Index = Annotated[int, Field(ge=0, lt=2**64)]
type _Stream = Annotated[
    str, StringConstraints(pattern=r"\A[A-Za-z][A-Za-z0-9_.:/-]*\z")
]

_SCHEME_WORD = 0x4F504931  # OPI1; changing derivation requires a new schema.
_WORD_MASK = 2**32 - 1


class _Address(BaseModel):
    """Validate versioned, immutable host metadata without coercing indices."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    schema_version: Literal[1] = 1
    seed: _Seed
    index: _Index = 0

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "schema_version must be an integer"
            raise ValueError(msg)
        return value


class RandomKey(_Address):
    """Address one replayable task stream.

    Attributes:
        schema_version: Derivation scheme; unknown versions are refused.
        seed: Explicit unsigned 128-bit root entropy, recorded by the caller.
        stream: Case-sensitive ASCII namespace, e.g. ``parameters`` or ``simulation``.
        index: Unsigned 64-bit task/attempt counter, independent of worker order.

    Examples:
        >>> key = RandomKey(seed=2026, stream="simulation", index=8)
        >>> a = key.numpy_generator().normal(size=3)
        >>> np.array_equal(a, key.numpy_generator().normal(size=3))
        True
        >>> RandomKey.model_validate_json(key.model_dump_json()) == key
        True
    """

    stream: _Stream

    def seed_sequence(self) -> np.random.SeedSequence:
        """Create a fresh SeedSequence for native adapters.

        Version 1 hashes ASCII stream bytes with SHA-256, interprets the digest
        as eight little-endian uint32 words and uses the spawn path
        ``(0x4F504931, *stream_words, index_low32, index_high32)``. The root
        entropy is the complete seed; the pool has four words. No Python hash,
        worker number, batch length or mutable spawn counter enters the identity.

        Returns:
            A fresh native SeedSequence with the recorded entropy/path.

        Raises:
            ValueError: The address was constructed with invalid metadata.
        """
        key = RandomKey.model_validate(self)
        digest = hashlib.sha256(key.stream.encode("ascii")).digest()
        words = tuple(
            int.from_bytes(digest[offset : offset + 4], "little")
            for offset in range(0, len(digest), 4)
        )
        return np.random.SeedSequence(
            key.seed,
            spawn_key=(
                _SCHEME_WORD,
                *words,
                key.index & _WORD_MASK,
                key.index >> 32,
            ),
            pool_size=4,
        )

    def numpy_generator(self) -> np.random.Generator:
        """Start a fresh PCG64DXSM generator for prior/simulation consumers.

        Repeated lowering deliberately replays the start; retain the mutable
        generator to continue a stream, or change the address for a fresh task.
        NumPy's global random state is never read or seeded.

        Returns:
            A local generator using the explicitly named PCG64DXSM algorithm.
        """
        return np.random.Generator(np.random.PCG64DXSM(self.seed_sequence()))


class SimulationStreams(_Address):
    """Supply distinct default stream addresses to prior and forward consumers.

    Rejections/retries need a new explicit
    index or namespace; the object does not silently advance or retry anything.

    Attributes:
        schema_version: Derivation scheme shared by both keys.
        seed: Explicit unsigned 128-bit root entropy.
        index: Unsigned 64-bit task/attempt identity shared by both roles.

    Examples:
        >>> streams = SimulationStreams(seed=2026, index=12)
        >>> streams.parameters.stream, streams.simulation.stream
        ('parameters', 'simulation')
        >>> streams.parameters.index == streams.simulation.index
        True
    """

    @property
    def parameters(self) -> RandomKey:
        """Return the parameter key  without mutable shared state."""
        validated = SimulationStreams.model_validate(self)
        return RandomKey(
            seed=validated.seed, stream="parameters", index=validated.index
        )

    @property
    def simulation(self) -> RandomKey:
        """Return the separate simulation key for forward-model calls."""
        validated = SimulationStreams.model_validate(self)
        return RandomKey(
            seed=validated.seed, stream="simulation", index=validated.index
        )
