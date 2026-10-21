"""Lazy native RNG lowerings for explicit task addresses.

The shared RandomKey supplies the
host identity; arrays and generators stay native after lowering. CPU sampling is
qualified; this module does not implement distributions or migrate consumers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import numpy as np

from op_infer.rng import RandomKey

if TYPE_CHECKING:
    from jax import Array
    from torch import Generator

__all__ = ["jax_key", "torch_generators"]


def jax_key(key: RandomKey) -> Array:
    """Lower a task address to an explicit typed Threefry key.

    The first two uint32 SeedSequence state words become Threefry2x32 key data.
    Lower on the host before tracing; the resulting key is a native JIT/vmap
    input. No global default PRNG implementation is consulted.

    Args:
        key: Validated, recorded stream address.

    Returns:
        A scalar typed JAX key using ``threefry2x32``.

    Raises:
        ImportError: The optional JAX dependency is unavailable.
    """
    words = key.seed_sequence().generate_state(2, dtype=np.uint32)
    try:
        random = importlib.import_module("jax.random")
        xp = importlib.import_module("jax.numpy")
    except ImportError as exc:
        msg = "Install op_infer[jax] to lower a JAX random key"
        raise ImportError(msg) from exc
    return cast(
        "Array",
        random.wrap_key_data(xp.asarray(words, dtype=xp.uint32), impl="threefry2x32"),
    )


def torch_generators(keys: tuple[RandomKey, ...]) -> tuple[Generator, ...]:
    """Lower a complete task bank to local Torch CPU generators.

    Torch's CPU MT19937 seed uses only 32 effective bits. Version 1 uses the
    first uint32 SeedSequence word explicitly and refuses duplicate effective
    seeds, including repeated addresses, before creating any generator. Pass
    all simultaneously independent tasks together; separate calls cannot check
    collisions against earlier banks. Deliberate replay uses a separate call.

    Args:
        keys: Immutable ordered bank of addresses; an empty bank is permitted.

    Returns:
        Fresh CPU generators in the same order, without seeding global state.

    Raises:
        TypeError: The bank is not a tuple of RandomKey records.
        ValueError: Addresses are invalid or effective seeds collide.
        ImportError: The optional Torch dependency is unavailable.
    """
    if not isinstance(keys, tuple) or any(
        not isinstance(key, RandomKey) for key in keys
    ):
        msg = "Torch generator banks require a tuple of RandomKey records"
        raise TypeError(msg)
    seeds = [
        int(key.seed_sequence().generate_state(1, dtype=np.uint32)[0]) for key in keys
    ]
    if len(set(seeds)) != len(seeds):
        msg = (
            "Torch CPU effective seeds collide; use a new recorded root seed or "
            "stream namespace for the complete bank"
        )
        raise ValueError(msg)
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        msg = "Install op_infer[torch] to lower Torch CPU generators"
        raise ImportError(msg) from exc
    return tuple(
        cast("Generator", torch.Generator(device="cpu").manual_seed(seed))
        for seed in seeds
    )
