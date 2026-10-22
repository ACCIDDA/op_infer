"""Atomic, immutable local registration shared by saved decision contracts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_immutable_json(path: str | Path, payload: bytes, *, conflict: str) -> None:
    """Install complete bytes atomically, accepting only an identical prior file.

    Raises:
        ValueError: Existing bytes differ from the proposed contract.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(conflict) from None
    finally:
        temporary.unlink()
