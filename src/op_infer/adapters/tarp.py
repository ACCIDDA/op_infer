"""Optional TARP curves from the public reference implementation.

This array reporting boundary takes an
already sampled bank and explicit reference points. Sampling, coordinate
transformation, MCMC adequacy and scientific thresholds belong to the caller.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import platform
import threading
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from op_infer.posterior import ArtifactReference, Provenance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

__all__ = ["tarp_coverage"]

_REFERENCE_LOCK = threading.RLock()


class _Reference(Protocol):
    def __call__(
        self,
        samples: NDArray[np.float64],
        theta: NDArray[np.float64],
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Call the reference package's public NumPy array API."""
        ...


def _host(value: NDArray[np.generic]) -> NDArray[np.float64]:
    if (
        not isinstance(value, np.ndarray)
        or isinstance(value, np.ma.MaskedArray)
        or value.dtype.kind != "f"
        or value.dtype.itemsize not in {4, 8}
    ):
        msg = "Explicitly transfer to unmasked NumPy float32/float64 for TARP"
        raise TypeError(msg)
    return np.array(value, dtype=np.float64, copy=True, order="C")


def _content(value: NDArray[np.float64]) -> dict[str, Any]:
    header = str((value.shape, "<f8")).encode()
    return ArtifactReference.from_bytes(
        header + value.astype("<f8", copy=False).tobytes(order="C"),
        kind="tarp-numpy-f64-shape-c-little-endian-v1",
    ).model_dump(mode="json")


def _curve(
    reference: _Reference,
    samples: NDArray[np.float64],
    truth: NDArray[np.float64],
    points: NDArray[np.float64],
    bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # The reference uses strict distance ranks without atom randomization.
    # Refuse ties rather than present their curve as a continuous-rank diagnostic.
    with np.errstate(over="raise", invalid="raise"):
        draw_distances = np.sqrt(np.sum((samples - points[None]) ** 2, axis=-1))
        truth_distances = np.sqrt(np.sum((truth - points) ** 2, axis=-1))
    if np.any(draw_distances == truth_distances):
        msg = "TARP strict distance ranks contain ties; use atom-aware PIT diagnostics"
        raise ValueError(msg)
    # TARP 0.1.1 seeds the legacy global RNG even with explicit references and no
    # bootstrap. Serialize our calls and restore the complete state on failure too.
    with _REFERENCE_LOCK:
        state = copy.deepcopy(np.random.get_state(legacy=False))  # noqa: NPY002 - reference compatibility
        try:
            with np.errstate(over="raise", invalid="raise"):
                ecp, alpha = reference(
                    samples,
                    truth,
                    references=points,
                    metric="euclidean",
                    num_alpha_bins=bins,
                    norm=False,
                    bootstrap=False,
                    seed=0,
                )
        finally:
            np.random.set_state(state)  # noqa: NPY002 - restore caller's complete state
    if (
        ecp.shape != (bins + 1,)
        or alpha.shape != (bins + 1,)
        or not np.all(np.isfinite(ecp))
        or not np.all(np.isfinite(alpha))
        or np.any((ecp < -1e-12) | (ecp > 1 + 1e-12))
        or np.any(np.diff(ecp) < -1e-12)
        or not np.allclose(ecp[[0, -1]], (0, 1))
        or np.any((alpha < -1e-12) | (alpha > 1 + 1e-12))
        or np.any(np.diff(alpha) <= 0)
        or not np.allclose(alpha, np.linspace(alpha[0], alpha[-1], bins + 1))
    ):
        msg = "TARP returned an invalid coverage curve"
        raise ValueError(msg)
    return ecp, alpha


def tarp_coverage(  # noqa: PLR0913 - explicit bank, coordinates and provenance boundary
    samples: NDArray[np.generic],
    truth: NDArray[np.generic],
    *,
    references: NDArray[np.generic],
    replicate_ids: Sequence[int],
    quantities: Sequence[str],
    definition: ArtifactReference,
    provenance: Provenance,
    num_bins: int,
    unavailable: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Report a TARP curve with explicit coordinates and retained failed rows.

    Wraps ``tarp.get_tarp_coverage`` with fixed
    Euclidean distance, no automatic normalization and no bootstrap. All three
    arrays must already use the same declared coordinates. Reference points
    are supplied explicitly; their distribution determines diagnostic scope.

    Args:
        samples: Unmasked NumPy float32/64 bank, ``(draw, replicate, quantity)``.
            Draws must be appropriate for calibration; this does not check MCMC.
        truth: True reported quantities, ``(replicate, quantity)``.
        references: Explicit reference points, with the same shape as truth.
        replicate_ids: Complete planned row IDs, including failed-fit placeholders.
        quantities: Ordered, distinct names for the selected coordinates.
        definition: Reference to coordinate transforms/alignment, reference-point
            construction and sample-selection rules. Retain its payload separately.
        provenance: Shared bank/estimator context, not proof of scientific validity.
        num_bins: Explicit positive credibility-grid bin count.
        unavailable: Explicit failed/inadequate fits, keyed by planned row ID.
            Nonfinite rows are additionally counted; neither outcome is hidden.

    Returns:
        JSON-compatible curve, complete row statuses/counts, successful-input
        conditioning, input/source digests, method settings and dependency versions.
        All-failed banks or reference errors have an unavailable curve. No pass,
        p-value, nominal confidence band or automatic adequacy decision is supplied.

    Raises:
        ImportError: Install ``op_infer[tarp]`` for the reference implementation.
        TypeError: Arrays have not crossed the explicit supported host boundary.
        ValueError: Shapes, coordinate IDs, exclusions or bin count are invalid.
    """
    Provenance.model_validate(provenance)
    ArtifactReference.model_validate(definition)
    bank, targets, points = (_host(value) for value in (samples, truth, references))
    ids = tuple(replicate_ids)
    names = tuple(quantities)
    if (
        bank.ndim != 3
        or targets.ndim != 2
        or targets.shape != bank.shape[1:]
        or points.shape != targets.shape
        or bank.shape[0] < 2
        or min(targets.shape) < 1
        or len(ids) != bank.shape[1]
        or len(names) != bank.shape[2]
        or len(set(ids)) != len(ids)
        or len(set(names)) != len(names)
        or isinstance(quantities, str)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids)
    ):
        msg = "TARP requires original sample shapes and distinct row/quantity IDs"
        raise ValueError(msg)
    if isinstance(num_bins, bool) or not isinstance(num_bins, Integral) or num_bins < 1:
        msg = "num_bins must be a positive integer"
        raise ValueError(msg)
    ids = tuple(int(i) for i in ids)
    excluded = dict(unavailable or {})
    if not set(excluded) <= set(ids) or any(
        isinstance(i, bool)
        or not isinstance(i, Integral)
        or not isinstance(reason, str)
        or not reason.strip()
        for i, reason in excluded.items()
    ):
        msg = "Unavailable rows need declared IDs and explicit reasons"
        raise ValueError(msg)
    try:
        reference = cast(
            "_Reference", importlib.import_module("tarp").get_tarp_coverage
        )
    except ImportError as error:
        msg = "Install op_infer[tarp] for the TARP reference implementation"
        raise ImportError(msg) from error
    bad = (
        np.count_nonzero(~np.isfinite(bank), axis=(0, 2))
        + np.count_nonzero(~np.isfinite(targets), axis=1)
        + np.count_nonzero(~np.isfinite(points), axis=1)
    )
    keep = np.array(
        [bad[i] == 0 and label not in excluded for i, label in enumerate(ids)]
    )
    rows = [
        {
            "replicate_id": label,
            "status": "available" if keep[i] else "unavailable",
            "nonfinite_values": int(bad[i]),
            "reasons": ([excluded[label]] if label in excluded else [])
            + (["nonfinite_input"] if bad[i] else []),
        }
        for i, label in enumerate(ids)
    ]
    curve = None
    reason = "no_usable_replicates"
    if np.any(keep):
        try:
            ecp, alpha = _curve(
                reference, bank[:, keep], targets[keep], points[keep], int(num_bins)
            )
            curve = {
                "expected_coverage": ecp.tolist(),
                "credibility": alpha.tolist(),
                "max_abs_deviation": float(np.max(np.abs(ecp - alpha))),
            }
            reason = None
        except Exception as error:  # noqa: BLE001 - retain reference failure, not a pass
            reason = f"reference_error:{type(error).__name__}:{error}"
    usable = int(np.count_nonzero(keep))
    return {
        "schema_version": 1,
        "status": "available" if curve is not None else "unavailable",
        "reason": reason,
        "curve": curve,
        "replicates": rows,
        "n_total": len(ids),
        "n_usable": usable,
        "n_failed": len(ids) - usable,
        "failed_fraction": (len(ids) - usable) / len(ids),
        "n_posterior_draws": bank.shape[0],
        "quantities": list(names),
        "conditioning": "usable inputs; failed/inadequate rows excluded and counted",
        "settings": {
            "num_bins": int(num_bins),
            "metric": "euclidean",
            "norm": False,
            "bootstrap": False,
            "reference_seed": 0,
            "credibility_grid": "reference_histogram_edges",
        },
        "implementation": "tarp.get_tarp_coverage",
        "backend_versions": {
            name: importlib.metadata.version(name)
            for name in ("tarp", "numpy", "op_infer")
        }
        | {"python": platform.python_version()},
        "implementation_reference": ArtifactReference.from_bytes(
            Path(__file__).read_bytes(), kind="op-infer-tarp-adapter-source-v1"
        ).model_dump(mode="json"),
        "inputs": {
            "samples": _content(bank),
            "truth": _content(targets),
            "references": _content(points),
        },
        "definition": definition.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
        "adequacy_status": "not_assessed",
    }
