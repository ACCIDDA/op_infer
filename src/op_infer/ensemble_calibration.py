"""Per-member and mixture calibration report for an ensemble of posteriors.

For an equal-weight mixture of independently trained estimators, calibration of
the **mixture** is what is reported (Hermans et al. 2022, arXiv:2110.06581, on
ensembling; Talts et al. 2018, arXiv:1804.06788, on SBC). Whether the mixture is
wider than its members, how many members individually show rank flags, and how
far the members' locations disagree are separate and useful diagnostics: a
mixture of members that differ in location can be wider than every member.

This module only summarises records that the caller has produced (or builds them
from posterior draws with ``MemberRecords.from_draws``); it reports and gates
nothing. Rank flags are counted (``n_flagged`` per member and for the mixture,
Benjamini-Hochberg across estimands), never turned into a pass or fail, and
member agreement and validation-loss spread are reported only.

The SBC runner stores PITs only, not truths, means or member structure, so the
records come from the caller's own loop over replicates (see the docs page).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from op_infer.calibrate import aggregate, pit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

LEVELS = (0.5, 0.9)
_FIELDS = (
    "truth",
    "member_mean",
    "member_sd",
    "member_pit",
    "mixture_mean",
    "mixture_sd",
    "mixture_pit",
)


@dataclass(frozen=True)
class MemberRecords:
    """Per-replicate summaries of every member and the mixture for one estimand.

    Shapes: ``truth`` and ``mixture_*`` are ``(R,)``; ``member_*`` are ``(M, R)``.
    ``*_pit`` are PITs in [0, 1], randomised when the estimand has atoms;
    ``*_mean`` and ``*_sd`` the posterior mean and SD of each replicate. For an
    estimand with ``atoms=True`` the standardised-error quantities are not
    defined and are reported as ``None``; SDs may then be zero.
    """

    truth: ArrayLike
    member_mean: ArrayLike
    member_sd: ArrayLike
    member_pit: ArrayLike
    mixture_mean: ArrayLike
    mixture_sd: ArrayLike
    mixture_pit: ArrayLike
    atoms: bool = False

    @classmethod
    def from_draws(
        cls,
        truth: ArrayLike,
        draws: ArrayLike,
        *,
        atoms: bool = False,
        rng: np.random.Generator | None = None,
    ) -> MemberRecords:
        """Build records from per-replicate posterior draws.

        ``truth`` is ``(R,)`` and ``draws`` is ``(R, M, S)``: ``S`` draws of the
        estimand from each of ``M`` members for every replicate. The mixture is
        the equal-weight pool of the members' draws. PITs use ``calibrate.pit``
        (randomised with ``rng`` when ``atoms`` is true).

        Raises:
            ValueError: Shapes are inconsistent, or ``atoms`` is set without ``rng``.

        Examples:
            >>> g = np.random.default_rng(0)
            >>> rec = MemberRecords.from_draws(
            ...     g.normal(size=20), g.normal(size=(20, 2, 50))
            ... )
            >>> np.asarray(rec.member_mean).shape, np.asarray(rec.mixture_pit).shape
            ((2, 20), (20,))
        """
        t = np.asarray(truth, dtype=float)
        d = np.asarray(draws, dtype=float)
        if t.ndim != 1 or d.ndim != 3 or d.shape[0] != t.shape[0]:
            msg = "truth must be (R,) and draws (R, M, S)"
            raise ValueError(msg)
        n, m, _ = d.shape
        pooled = d.reshape(n, -1)
        return cls(
            truth=t,
            member_mean=d.mean(axis=2).T,
            member_sd=d.std(axis=2).T,
            member_pit=np.array(
                [
                    [pit(d[r, i], t[r], atoms=atoms, rng=rng) for r in range(n)]
                    for i in range(m)
                ]
            ),
            mixture_mean=pooled.mean(axis=1),
            mixture_sd=pooled.std(axis=1),
            mixture_pit=np.array(
                [pit(pooled[r], t[r], atoms=atoms, rng=rng) for r in range(n)]
            ),
            atoms=atoms,
        )


def _arrays(name: str, r: MemberRecords) -> dict[str, NDArray[np.float64]]:
    a = {k: np.asarray(getattr(r, k), dtype=float) for k in _FIELDS}
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
    sds = (a["member_sd"], a["mixture_sd"])
    if (r.atoms and any(np.any(s < 0) for s in sds)) or (
        not r.atoms and any(np.any(s <= 0) for s in sds)
    ):
        msg = f"{name}: posterior SDs must be positive (nonnegative for atoms)"
        raise ValueError(msg)
    return a


def _row(
    truth: Any,  # noqa: ANN401
    mean: Any,  # noqa: ANN401
    sd: Any,  # noqa: ANN401
    pit_: Any,  # noqa: ANN401
    *,
    atoms: bool,
) -> dict[str, float | None]:
    row: dict[str, float | None] = {"z_sd": None, "z_mean": None, "z_rms": None}
    if not atoms:
        z = (truth - mean) / sd
        row = {
            "z_sd": float(np.std(z)),
            "z_mean": float(np.mean(z)),
            "z_rms": float(np.sqrt(np.mean(z**2))),
        }
    for level in LEVELS:
        tail = (1.0 - level) / 2.0
        row[f"coverage{round(level * 100)}"] = float(
            np.mean((pit_ >= tail) & (pit_ <= 1.0 - tail))
        )
    return row


def member_calibration(
    records: Mapping[str, MemberRecords],
    *,
    alpha: float = 0.05,
    bins: int = 20,
    validation_losses: Sequence[float] | None = None,
    n_failed: int = 0,
) -> dict[str, Any]:
    """Summarise per-member and mixture calibration over named estimands.

    For each estimand: the centred SD (``z_sd``), mean (``z_mean``) and root mean
    square (``z_rms``) of the standardised error ``(truth - mean) / sd``, and the
    coverage of the central 50% and 90% intervals taken from the PITs, for every
    member and for the mixture. ``z_sd`` is centred, so a member with the right
    width but a bias shows ``z_sd`` near 1 and a non-zero ``z_mean``; read the two
    together, or use ``z_rms``. The mixture's ``z_sd`` uses its Gaussian summary
    SD, while its coverage uses the mixture PIT; for non-Gaussian mixtures these
    are different notions.

    With two or more members, ``z_sd_gap`` is the smallest member ``z_sd`` minus
    the mixture ``z_sd``: positive when the mixture's standardised errors are
    smaller than every member's, that is when it is wider than its own error
    scale relative to the members (a positive gap does not by itself show that the
    mixture is wide enough). ``agreement`` is the median over replicates of the
    range of member means in units of the mixture SD; it grows with the number of
    members and is not comparable across ensemble sizes. With one member both are
    ``None``. For atom estimands the standardised-error fields are ``None``.

    Rank flags are counted per member and for the mixture across the estimands
    (Benjamini-Hochberg through ``calibrate.aggregate``); the per-bank
    ``underpowered`` flag, replicate count and failed count are passed through.
    ``members_with_flags`` has no multiplicity control across members: with ``M``
    calibrated members that share the same truths the chance that at least one is
    flagged is about ``1 - (1 - alpha)**M`` per estimand, so it is not evidence of
    member miscalibration by itself. Nothing is gated.

    Raises:
        ValueError: Records are empty, inconsistent or non-finite, the member
            counts differ between estimands, or ``validation_losses`` is not one
            finite value per member.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> rec = MemberRecords.from_draws(
        ...     rng.normal(size=60), rng.normal(size=(60, 3, 200))
        ... )
        >>> out = member_calibration({"theta": rec})
        >>> out["n_members"], sorted(out["estimands"]["theta"])
        (3, ['agreement', 'members', 'mixture', 'z_sd_gap'])
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
    losses = (
        None if validation_losses is None else [float(x) for x in validation_losses]
    )
    if losses is not None and (
        len(losses) != n_members or not np.all(np.isfinite(losses))
    ):
        msg = "validation_losses must be finite, one per member"
        raise ValueError(msg)
    table: dict[str, Any] = {}
    for name, a in arrays.items():
        atoms = records[name].atoms
        members = [
            _row(
                a["truth"],
                a["member_mean"][i],
                a["member_sd"][i],
                a["member_pit"][i],
                atoms=atoms,
            )
            for i in range(n_members)
        ]
        mixture = _row(
            a["truth"],
            a["mixture_mean"],
            a["mixture_sd"],
            a["mixture_pit"],
            atoms=atoms,
        )
        gap = agreement = None
        if n_members >= 2 and not atoms:
            gap = float(
                min(cast("float", m["z_sd"]) for m in members)
                - cast("float", mixture["z_sd"])
            )
            spread = a["member_mean"].max(axis=0) - a["member_mean"].min(axis=0)
            agreement = float(np.median(spread / a["mixture_sd"]))
        table[name] = {
            "members": members,
            "mixture": mixture,
            "z_sd_gap": gap,
            "agreement": agreement,
        }
    flags = [
        aggregate(
            {n: a["member_pit"][i] for n, a in arrays.items()},
            bins=bins,
            alpha=alpha,
            n_failed=n_failed,
        )
        for i in range(n_members)
    ]
    mix = aggregate(
        {n: a["mixture_pit"] for n, a in arrays.items()},
        bins=bins,
        alpha=alpha,
        n_failed=n_failed,
    )
    return {
        "reading": "descriptive; no gate is applied",
        "n_members": n_members,
        "estimands": table,
        "member_n_flagged": [f["n_flagged"] for f in flags],
        "member_flagged": [f["flagged"] for f in flags],
        "mixture_n_flagged": mix["n_flagged"],
        "mixture_flagged": mix["flagged"],
        "members_with_flags": int(sum(f["n_flagged"] > 0 for f in flags)),
        "n_replicates": mix["n_replicates"],
        "n_failed": mix["n_failed"],
        "underpowered": mix["underpowered"],
        "underpowered_banner": mix["underpowered_banner"],
        "validation_loss_spread": None
        if losses is None
        else float(max(losses) - min(losses)),
        "settings": {"alpha": alpha, "bins": bins, "levels": list(LEVELS)},
    }
