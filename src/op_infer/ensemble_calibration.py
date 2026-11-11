"""Per-member and mixture calibration report for an ensemble of posteriors.

For an equal-weight mixture of independently trained estimators, calibration of
the **mixture** is what is reported (Hermans et al. 2022, arXiv:2110.06581, on
ensembling; Talts et al. 2018, arXiv:1804.06788, on SBC). Whether the mixture is
wider than its members, how many members individually show rank flags, and how
far the members' locations disagree are separate and useful diagnostics: a
mixture of members that differ in location can be wider than every member.

This module only summarises records that the caller has already produced; it
reports and gates nothing. Rank flags are counted (``n_flagged`` per member and
for the mixture, Benjamini-Hochberg across estimands), never turned into a pass
or fail, and member agreement and validation-loss spread are reported only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from op_infer.calibrate import aggregate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

LEVELS = (0.5, 0.9)


@dataclass(frozen=True)
class MemberRecords:
    """Per-replicate summaries of every member and the mixture for one estimand.

    Shapes: ``truth``, ``mixture_*`` are ``(R,)``; ``member_*`` are ``(M, R)``.
    ``*_pit`` are the (randomised, if the estimand has atoms) PITs in [0, 1];
    ``*_mean`` and ``*_sd`` the posterior mean and SD of each replicate.
    """

    truth: ArrayLike
    member_mean: ArrayLike
    member_sd: ArrayLike
    member_pit: ArrayLike
    mixture_mean: ArrayLike
    mixture_sd: ArrayLike
    mixture_pit: ArrayLike


def _arrays(name: str, r: MemberRecords) -> dict[str, NDArray[np.float64]]:
    a = {
        k: np.asarray(getattr(r, k), dtype=float)
        for k in (
            "truth",
            "member_mean",
            "member_sd",
            "member_pit",
            "mixture_mean",
            "mixture_sd",
            "mixture_pit",
        )
    }
    n = a["truth"].shape
    if len(n) != 1 or n[0] < 2 or a["member_mean"].ndim != 2:
        msg = f"{name}: truth must be (R,) with R >= 2 and member arrays (M, R)"
        raise ValueError(msg)
    m = a["member_mean"].shape
    ok = (
        all(a[k].shape == m for k in ("member_sd", "member_pit"))
        and all(a[k].shape == n for k in ("mixture_mean", "mixture_sd", "mixture_pit"))
        and m[1] == n[0]
        and m[0] >= 1
    )
    if not ok:
        msg = f"{name}: inconsistent record shapes"
        raise ValueError(msg)
    if not all(np.all(np.isfinite(v)) for v in a.values()):
        msg = f"{name}: records must be finite; count failed replicates separately"
        raise ValueError(msg)
    if np.any(a["member_sd"] <= 0) or np.any(a["mixture_sd"] <= 0):
        msg = f"{name}: posterior SDs must be positive"
        raise ValueError(msg)
    return a


def _row(truth: Any, mean: Any, sd: Any, pit: Any) -> dict[str, float]:  # noqa: ANN401
    z = (truth - mean) / sd
    row = {"z_sd": float(np.std(z)), "z_mean": float(np.mean(z))}
    for level in LEVELS:
        tail = (1.0 - level) / 2.0
        row[f"coverage{round(level * 100)}"] = float(
            np.mean((pit >= tail) & (pit <= 1.0 - tail))
        )
    return row


def member_calibration(
    records: Mapping[str, MemberRecords],
    *,
    alpha: float = 0.05,
    bins: int = 20,
    validation_losses: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarise per-member and mixture calibration over named estimands.

    For each estimand: the SD and mean of the standardised error
    ``(truth - mean) / sd`` and the coverage of the central 50% and 90% intervals
    for every member and for the mixture; ``z_sd_gap`` (the smallest member SD
    minus the mixture SD, positive when the mixture is wider than every member);
    and the agreement of the members, the median over replicates of the range of
    member means in units of the mixture SD. Rank flags are counted per member
    and for the mixture across the estimands (Benjamini-Hochberg through
    ``calibrate.aggregate``); the spread of ``validation_losses`` is reported.
    Nothing is gated.

    Raises:
        ValueError: Records are empty, inconsistent or non-finite, or the member
            counts differ between estimands.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> u = rng.uniform(size=(2, 400))
        >>> rec = MemberRecords(
        ...     truth=rng.normal(size=400),
        ...     member_mean=np.zeros((2, 400)),
        ...     member_sd=np.ones((2, 400)),
        ...     member_pit=u,
        ...     mixture_mean=np.zeros(400),
        ...     mixture_sd=np.ones(400),
        ...     mixture_pit=u[0],
        ... )
        >>> out = member_calibration({"theta": rec})
        >>> sorted(out["estimands"]["theta"]), out["n_members"]
        (['agreement', 'members', 'mixture', 'z_sd_gap'], 2)
    """
    if not records:
        msg = "provide at least one estimand"
        raise ValueError(msg)
    arrays = {name: _arrays(name, r) for name, r in records.items()}
    sizes = {a["member_mean"].shape[0] for a in arrays.values()}
    if len(sizes) != 1:
        msg = "every estimand must carry the same number of members"
        raise ValueError(msg)
    n_members = sizes.pop()
    table: dict[str, Any] = {}
    for name, a in arrays.items():
        members = [
            _row(a["truth"], a["member_mean"][i], a["member_sd"][i], a["member_pit"][i])
            for i in range(n_members)
        ]
        mixture = _row(a["truth"], a["mixture_mean"], a["mixture_sd"], a["mixture_pit"])
        spread = a["member_mean"].max(axis=0) - a["member_mean"].min(axis=0)
        table[name] = {
            "members": members,
            "mixture": mixture,
            "z_sd_gap": float(min(m["z_sd"] for m in members) - mixture["z_sd"]),
            "agreement": float(np.median(spread / a["mixture_sd"])),
        }
    flags = [
        aggregate(
            {n: a["member_pit"][i] for n, a in arrays.items()}, bins=bins, alpha=alpha
        )
        for i in range(n_members)
    ]
    mix_flags = aggregate(
        {n: a["mixture_pit"] for n, a in arrays.items()}, bins=bins, alpha=alpha
    )
    losses = (
        None if validation_losses is None else [float(x) for x in validation_losses]
    )
    if losses is not None and (
        len(losses) != n_members or not all(np.isfinite(losses))
    ):
        msg = "validation_losses must be finite, one per member"
        raise ValueError(msg)
    return {
        "reading": "descriptive; no gate is applied",
        "n_members": n_members,
        "estimands": table,
        "member_n_flagged": [f["n_flagged"] for f in flags],
        "member_flagged": [f["flagged"] for f in flags],
        "mixture_n_flagged": mix_flags["n_flagged"],
        "mixture_flagged": mix_flags["flagged"],
        "members_with_flags": int(sum(f["n_flagged"] > 0 for f in flags)),
        "validation_loss_spread": None
        if losses is None
        else float(max(losses) - min(losses)),
        "settings": {"alpha": alpha, "bins": bins, "levels": list(LEVELS)},
    }
