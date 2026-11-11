"""Robustness of an unchanged estimator under a declared perturbed generator.

SBC under the training generator checks the computation, not what happens when
the data come from a different generator (Cannon, Ward and Schmon 2022,
arXiv:2209.01845; Ward et al. 2022, arXiv:2210.06564). A stress run draws a
replicate bank from a declared perturbation while the estimator is unchanged; a
paired control run uses the same seed and replicate IDs without it. The
comparison is a robustness reading, not a calibration reading.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from op_infer.calibrate import coverage_summary

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class StressSpec:
    """A named, declared perturbation of the generator.

    The record enters the immutable SBC run contract, so a stress bank cannot
    share a directory with its control or with a different stress.

    Examples:
        >>> spec = StressSpec("noise_x2", "observation noise doubled", {"s": 2.0})
        >>> spec.record()["settings"]
        {'s': 2.0}
    """

    name: str
    description: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """Return the JSON-compatible record stored in the run contract.

        Raises:
            ValueError: The name is not a portable identifier.
        """
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            msg = "StressSpec names must be portable identifiers"
            raise ValueError(msg)
        return {
            "name": self.name,
            "description": self.description,
            "settings": dict(self.settings),
        }


def _replicates(directory: Path) -> dict[str, dict[str, Any]]:
    files = sorted(Path(directory).glob("rep_*.json"))
    if not files:
        msg = f"no rep_*.json replicate files in {directory}"
        raise ValueError(msg)
    return {f.name: json.loads(f.read_text()) for f in files}


def _ok(record: Mapping[str, Any]) -> bool:
    return record.get("_failed") is not True


def stress_verdict(  # noqa: PLR0913 - explicit reading rule, no defaults
    stress_directory: str | Path,
    control_directory: str | Path,
    estimands: Sequence[str],
    *,
    level: float,
    floor: float,
    drop: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare central coverage under stress with the paired control.

    Replicates are paired by file name (same seed and replicate ID) and used
    only if both succeeded; failures and unpaired replicates are counted and
    never dropped silently. An estimand passes when its stress coverage is at
    least ``max(floor, control_coverage - drop)``. The caller supplies ``floor``
    and ``drop`` (no defaults): they are reading-rule choices, not derived. This
    is a robustness reading; it does not certify calibration.

    Args:
        stress_directory: SBC bank from the perturbed generator.
        control_directory: SBC bank from the same seed and IDs without the stress.
        estimands: Estimand names to read.
        level: Central interval level in (0, 1).
        floor: Absolute coverage floor in [0, 1].
        drop: Allowed drop from the control coverage, in [0, 1].
        confidence: Wilson interval confidence.

    Returns:
        JSON-compatible counts and one row per estimand with both coverages,
        Wilson intervals, the requirement and the verdict; ``passes`` is true
        only if every estimand passes.

    Raises:
        ValueError: Settings are invalid, or no replicate is paired and successful.

    Examples:
        Requires the study extra; the required study job executes this example.

        >>> import json, tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> for sub, pit in (("s", 0.5), ("c", 0.5)):
        ...     (root / sub).mkdir()
        ...     for i in range(4):
        ...         _ = (root / sub / f"rep_{i:020d}.json").write_text(
        ...             json.dumps({"theta": {"pit": pit}})
        ...         )
        >>> v = stress_verdict(
        ...     root / "s", root / "c", ["theta"], level=0.9, floor=0.8, drop=0.1
        ... )  # doctest: +SKIP
        >>> v["passes"], v["n_paired"]  # doctest: +SKIP
        (True, 4)
    """
    if not (0 < level < 1) or not (0 <= floor <= 1) or not (0 <= drop <= 1):
        msg = "level must be in (0, 1); floor and drop in [0, 1]"
        raise ValueError(msg)
    names = list(estimands)
    if not names or len(set(names)) != len(names):
        msg = "provide distinct estimand names"
        raise ValueError(msg)
    stress, control = (
        _replicates(Path(stress_directory)),
        _replicates(Path(control_directory)),
    )
    shared = sorted(set(stress) & set(control))
    both = [n for n in shared if _ok(stress[n]) and _ok(control[n])]
    if not both:
        msg = "no replicate is paired and successful in both banks"
        raise ValueError(msg)
    pits = {
        which: {e: [float(records[n][e]["pit"]) for n in both] for e in names}
        for which, records in (("stress", stress), ("control", control))
    }
    covs = {
        which: coverage_summary(pits[which], levels=(level,), confidence=confidence)
        for which in pits
    }
    rows: dict[str, Any] = {}
    for e in names:
        s_row, c_row = (
            covs["stress"]["estimands"][e][0],
            covs["control"]["estimands"][e][0],
        )
        need = max(floor, c_row["estimate"] - drop)
        rows[e] = {
            "stress": {k: s_row[k] for k in ("estimate", "low", "high")},
            "control": {k: c_row[k] for k in ("estimate", "low", "high")},
            "required": float(need),
            "passes": bool(s_row["estimate"] >= need),
        }
    return {
        "reading": "robustness under a declared perturbation, not calibration",
        "settings": {
            "level": level,
            "floor": floor,
            "drop": drop,
            "confidence": confidence,
            "method": "wilson",
        },
        "n_paired": len(both),
        "n_stress": len(stress),
        "n_control": len(control),
        "n_failed_stress": sum(not _ok(r) for r in stress.values()),
        "n_failed_control": sum(not _ok(r) for r in control.values()),
        "n_unpaired": len(set(stress) ^ set(control)),
        "estimands": rows,
        "passes": bool(all(r["passes"] for r in rows.values())),
        "multiplicity": "marginal; no adjustment across estimands",
    }
