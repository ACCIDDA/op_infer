"""Robustness of an unchanged estimator under a declared perturbed generator.

SBC under the training generator checks the computation, not what happens when
the data come from a different generator (Cannon, Ward and Schmon 2022,
arXiv:2209.01845; Ward et al. 2022, arXiv:2210.06564). A stress run draws a
replicate bank from a declared perturbation while the estimator is unchanged; a
paired control run uses the same seed, replicate IDs, estimator and estimands
without it. The comparison is a robustness reading, not a calibration reading.
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

_MANIFEST = "sbc-manifest.meta"


@dataclass(frozen=True)
class StressSpec:
    """A named, declared perturbation of the generator.

    The record enters the immutable SBC run contract, so a stress bank cannot
    share a directory with its control or with a different stress. ``settings``
    must be JSON-serialisable (plain Python numbers, strings, lists and dicts; no
    NumPy scalars) and every field, including ``description``, is part of the
    contract identity.

    Examples:
        >>> spec = StressSpec("noise_x2", "observation noise doubled", {"s": 2.0})
        >>> spec.record()["settings"]
        {'s': 2.0}
    """

    name: str
    description: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject names that are not portable identifiers.

        Raises:
            ValueError: The name is not a portable identifier.
        """
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            msg = "StressSpec names must be portable identifiers"
            raise ValueError(msg)

    def record(self) -> dict[str, Any]:
        """Return the JSON-compatible record stored in the run contract."""
        return {
            "name": self.name,
            "description": self.description,
            "settings": dict(self.settings),
        }


def _manifest(directory: Path) -> dict[str, Any]:
    path = directory / _MANIFEST
    if not path.is_file():
        msg = f"{directory} has no {_MANIFEST}; a bank must come from run_sbc"
        raise ValueError(msg)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        msg = f"{path} is not a JSON object"
        raise ValueError(msg)  # noqa: TRY004 - schema violation
    return data


def _replicates(directory: Path) -> dict[str, dict[str, Any]]:
    files = sorted(directory.glob("rep_*.json"))
    if not files:
        msg = f"no rep_*.json replicate files in {directory}"
        raise ValueError(msg)
    return {f.name: json.loads(f.read_text()) for f in files}


def _ok(record: Mapping[str, Any]) -> bool:
    return record.get("_failed") is not True


def _pit(name: str, record: Mapping[str, Any], estimand: str) -> float:
    entry = record.get(estimand)
    if not isinstance(entry, dict) or "pit" not in entry:
        msg = f"{name}: missing estimand {estimand!r}"
        raise ValueError(msg)
    return float(entry["pit"])


def _check_pair(stress: Path, control: Path) -> None:
    """Require two banks that differ only by the declared stress record."""
    s, c = _manifest(stress), _manifest(control)
    if "stress" not in s or "stress" in c:
        msg = "the stress bank must carry a stress record and the control must not"
        raise ValueError(msg)
    if {k: v for k, v in s.items() if k != "stress"} != c:
        msg = "stress and control banks differ in more than the stress record"
        raise ValueError(msg)


def _number(name: str, value: float, low: float, high: float, *, open_: bool) -> None:
    ok = not isinstance(value, bool) and (
        low < value < high if open_ else low <= value <= high
    )
    if not ok:
        left, right = ("(", ")") if open_ else ("[", "]")
        msg = f"{name} must be a number in {left}{low}, {high}{right}"
        raise ValueError(msg)


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

    The two banks must come from ``run_sbc``: the stress bank's manifest must
    carry a ``stress`` record and otherwise equal the control's manifest (same
    seed, provenance, estimands and observations). Replicates are paired by file
    name (replicate ID) and used only if both succeeded; failures and unpaired
    replicates are counted, never dropped silently. An estimand passes when its
    stress coverage **point estimate** is at least ``max(floor, control_coverage -
    drop)``; the Wilson intervals are reported but play no part in the verdict,
    so small banks give noisy readings. The caller supplies ``floor`` and
    ``drop`` (no defaults): they are reading-rule choices, to be fixed before the
    banks are read. This is a robustness reading; it does not certify calibration
    and the intervals are marginal, without adjustment across estimands.

    Args:
        stress_directory: SBC bank from the perturbed generator.
        control_directory: SBC bank from the same seed and IDs without the stress.
        estimands: Distinct estimand names to read (a list, not a string).
        level: Central interval level in (0, 1).
        floor: Absolute coverage floor in [0, 1].
        drop: Allowed drop from the control coverage, in [0, 1].
        confidence: Wilson interval confidence in (0, 1).

    Returns:
        JSON-compatible result. ``n_stress`` and ``n_control`` count all
        replicate files of each bank, ``n_failed_*`` the failed ones,
        ``n_unpaired`` the replicate IDs present in only one bank and
        ``n_paired`` the pairs successful in both banks, on which every
        coverage is computed. One row per estimand holds both coverages with
        Wilson intervals, the requirement and the verdict; ``passes`` is true
        only if every estimand passes.

    Raises:
        ValueError: Settings are invalid, the banks are not a stress and control
            pair, a record lacks a requested estimand, or no replicate is paired
            and successful.

    Examples:
        Requires the study extra; the doctest is skipped, and
        ``tests/test_stress_sbc.py`` runs the full workflow.

        >>> v = stress_verdict(
        ...     "stress-bank",
        ...     "control-bank",
        ...     ["theta"],
        ...     level=0.9,
        ...     floor=0.8,
        ...     drop=0.1,
        ... )  # doctest: +SKIP
        >>> v["passes"]  # doctest: +SKIP
        True
    """
    _number("level", level, 0.0, 1.0, open_=True)
    _number("confidence", confidence, 0.0, 1.0, open_=True)
    _number("floor", floor, 0.0, 1.0, open_=False)
    _number("drop", drop, 0.0, 1.0, open_=False)
    if isinstance(estimands, str):
        msg = "estimands must be a sequence of names, not a string"
        raise TypeError(msg)
    names = list(estimands)
    if not names or len(set(names)) != len(names):
        msg = "provide distinct estimand names"
        raise ValueError(msg)
    s_dir, c_dir = Path(stress_directory), Path(control_directory)
    _check_pair(s_dir, c_dir)
    stress, control = _replicates(s_dir), _replicates(c_dir)
    both = [
        n
        for n in sorted(set(stress) & set(control))
        if _ok(stress[n]) and _ok(control[n])
    ]
    if not both:
        msg = "no replicate is paired and successful in both banks"
        raise ValueError(msg)
    covs = {
        which: coverage_summary(
            {e: [_pit(n, records[n], e) for n in both] for e in names},
            levels=(level,),
            confidence=confidence,
        )
        for which, records in (("stress", stress), ("control", control))
    }
    rows: dict[str, Any] = {}
    for e in names:
        s_row = covs["stress"]["estimands"][e][0]
        c_row = covs["control"]["estimands"][e][0]
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
            "decision": "point estimate",
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
