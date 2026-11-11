"""Per-member and mixture calibration: an analytic mixture-wider-than-members case."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_infer.ensemble_calibration import MemberRecords, member_calibration

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_HERMEGAUSS = cast(
    "Callable[[int], tuple[NDArray[np.float64], NDArray[np.float64]]]",
    np.polynomial.hermite_e.hermegauss,
)
_MEDIAN_ABS_NORMAL = 0.6744897501960817


def _phi(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(
        "NDArray[np.float64]",
        0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0))),
    )


def _bank(delta: float, n: int = 8000, seed: int = 0) -> MemberRecords:
    """Two members, each calibrated on its own, with displaced locations.

    Prior N(0, 1), unit noise, true posterior N(x / 2, 1 / 2). Member m has a
    location error delta * u_m (u_m standard normal, independent) and a width that
    already accounts for it, sd_m^2 = 1 / 2 + delta^2, so each member is
    calibrated. The mixture is their equal-weight average.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    x = theta + rng.normal(size=n)
    u = rng.normal(size=(2, n))
    means = x / 2.0 + delta * u
    sd_member = math.sqrt(0.5 + delta**2)
    member_pit = _phi((theta - means) / sd_member)
    mix_sd = np.sqrt(0.5 + delta**2 + delta**2 * (u[0] - u[1]) ** 2 / 4.0)
    return MemberRecords(
        truth=theta,
        member_mean=means,
        member_sd=np.full_like(means, sd_member),
        member_pit=member_pit,
        mixture_mean=means.mean(axis=0),
        mixture_sd=mix_sd,
        mixture_pit=member_pit.mean(axis=0),
    )


def _expected_mixture_z(delta: float) -> tuple[float, float]:
    """(z_sd, z_rms) of the mixture: sqrt(E[err^2] E[1 / sd_mix^2]); mean zero."""
    g, w = _HERMEGAUSS(80)
    inv = float(np.sum(w / (0.5 + delta**2 + delta**2 * g**2 / 2.0)) / np.sum(w))
    value = math.sqrt((0.5 + delta**2 / 2.0) * inv)
    return value, value


def test_calibrated_members_pooled_into_a_wider_mixture_match_the_analytic_value() -> (
    None
):
    """Members have SD 1 and mean 0; the mixture's SD is its exact expectation."""
    delta = 1.0
    out = member_calibration({"theta": _bank(delta)})
    row = out["estimands"]["theta"]
    for m in row["members"]:
        assert m["z_sd"] == pytest.approx(1.0, abs=0.03)
        assert m["z_mean"] == pytest.approx(0.0, abs=0.03)
        assert m["coverage90"] == pytest.approx(0.9, abs=0.02)
    expected, _ = _expected_mixture_z(delta)
    assert expected < 0.9
    assert row["mixture"]["z_sd"] == pytest.approx(expected, rel=0.03)
    assert row["mixture"]["z_rms"] == pytest.approx(expected, rel=0.03)
    assert row["mixture"]["coverage90"] > 0.92
    assert row["z_sd_gap"] > 0.1
    assert out["reading"].startswith("descriptive")


def test_agreement_matches_its_analytic_median() -> None:
    """Range of means is delta sqrt(2)|g|; the median is at |g| = 0.6745."""
    delta = 0.8
    g = _MEDIAN_ABS_NORMAL
    expected = (
        delta * math.sqrt(2.0) * g / math.sqrt(0.5 + delta**2 + delta**2 * g**2 / 2.0)
    )
    out = member_calibration({"theta": _bank(delta, n=20000, seed=2)})
    assert out["estimands"]["theta"]["agreement"] == pytest.approx(expected, rel=0.04)


def test_no_location_disagreement_gives_unit_sd_no_gap_and_zero_agreement() -> None:
    """With delta = 0 every SD is about 1, the gap about 0 and agreement exactly 0."""
    out = member_calibration({"theta": _bank(0.0)}, validation_losses=[1.0, 2.5])
    row = out["estimands"]["theta"]
    assert all(m["z_sd"] == pytest.approx(1.0, abs=0.03) for m in row["members"])
    assert abs(row["z_sd_gap"]) < 0.05
    assert row["agreement"] == pytest.approx(0.0, abs=1e-9)
    assert out["validation_loss_spread"] == 1.5
    assert out["mixture_n_flagged"] == 0
    assert out["underpowered"] is False
    assert out["n_replicates"] == 8000


def test_flags_are_counted_per_member_and_for_a_misspecified_mixture() -> None:
    """A too-narrow member and a too-narrow mixture are flagged; nothing is gated."""
    bank = _bank(0.5)
    pits = np.asarray(bank.member_pit).copy()
    pits[0] = np.clip(0.5 + 3.0 * (pits[0] - 0.5), 0.0, 1.0)
    bad = MemberRecords(
        bank.truth,
        bank.member_mean,
        bank.member_sd,
        pits,
        bank.mixture_mean,
        bank.mixture_sd,
        np.clip(0.5 + 3.0 * (np.asarray(bank.mixture_pit) - 0.5), 0.0, 1.0),
    )
    out = member_calibration({"theta": bad}, n_failed=7)
    assert out["member_n_flagged"][0] == 1
    assert out["member_n_flagged"][1] == 0
    assert out["mixture_n_flagged"] == 1
    assert out["members_with_flags"] == 1
    assert out["n_failed"] == 7
    assert "pass" not in out


def test_point_mass_estimand_with_randomised_pits_reads_as_calibrated() -> None:
    """A clipped estimand drawn through from_draws: PITs uniform, no flags, z None."""
    rng = np.random.default_rng(4)
    n, members, draws = 3000, 3, 400
    theta = rng.normal(size=n)
    x = theta + rng.normal(size=n)
    post = rng.normal(x[:, None, None] / 2.0, math.sqrt(0.5), size=(n, members, draws))
    rec = MemberRecords.from_draws(
        np.maximum(theta, 0.0),
        np.maximum(post, 0.0),
        atoms=True,
        rng=np.random.default_rng(9),
    )
    out = member_calibration({"positive_part": rec})
    row = out["estimands"]["positive_part"]
    assert row["z_sd_gap"] is None
    assert row["agreement"] is None
    assert row["mixture"]["z_sd"] is None
    assert row["mixture"]["coverage90"] == pytest.approx(0.9, abs=0.025)
    assert out["mixture_n_flagged"] == 0
    assert out["member_n_flagged"] == [0, 0, 0]


def test_single_member_has_no_gap_or_agreement_and_small_banks_are_underpowered() -> (
    None
):
    """M = 1 reports None, not zero; a small bank surfaces the power banner."""
    rng = np.random.default_rng(1)
    rec = MemberRecords.from_draws(rng.normal(size=30), rng.normal(size=(30, 1, 100)))
    out = member_calibration({"t": rec})
    assert out["estimands"]["t"]["z_sd_gap"] is None
    assert out["estimands"]["t"]["agreement"] is None
    assert out["underpowered"] is True
    assert "UNDERPOWERED" in out["underpowered_banner"]


def test_inputs_are_validated() -> None:
    """Empty, mismatched, non-finite and nonpositive records fail early."""
    good = _bank(0.1, n=50)
    with pytest.raises(ValueError, match="at least one"):
        member_calibration({})
    with pytest.raises(ValueError, match="one per member"):
        member_calibration({"t": good}, validation_losses=[1.0])
    bad_sd = MemberRecords(
        good.truth,
        good.member_mean,
        np.zeros((2, 50)),
        good.member_pit,
        good.mixture_mean,
        good.mixture_sd,
        good.mixture_pit,
    )
    with pytest.raises(ValueError, match="positive"):
        member_calibration({"t": bad_sd})
    atom_sd = MemberRecords(
        good.truth,
        good.member_mean,
        np.zeros((2, 50)),
        good.member_pit,
        good.mixture_mean,
        np.zeros(50),
        good.mixture_pit,
        atoms=True,
    )
    assert member_calibration({"t": atom_sd})["n_members"] == 2
    short = MemberRecords(
        good.truth,
        np.asarray(good.member_mean)[:, :10],
        good.member_sd,
        good.member_pit,
        good.mixture_mean,
        good.mixture_sd,
        good.mixture_pit,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        member_calibration({"t": short})
    fewer = MemberRecords(
        good.truth,
        np.asarray(good.member_mean)[:1],
        np.asarray(good.member_sd)[:1],
        np.asarray(good.member_pit)[:1],
        good.mixture_mean,
        good.mixture_sd,
        good.mixture_pit,
    )
    with pytest.raises(ValueError, match="same number of members"):
        member_calibration({"a": good, "b": fewer})
    nan = np.asarray(good.truth).copy()
    nan[0] = np.nan
    nonfinite = MemberRecords(
        nan,
        good.member_mean,
        good.member_sd,
        good.member_pit,
        good.mixture_mean,
        good.mixture_sd,
        good.mixture_pit,
    )
    with pytest.raises(ValueError, match="finite"):
        member_calibration({"t": nonfinite})
    with pytest.raises(ValueError, match="atoms"):
        MemberRecords.from_draws(np.zeros(3), np.zeros((3, 2, 5)), atoms=True)
    with pytest.raises(ValueError, match="truth must be"):
        MemberRecords.from_draws(np.zeros(3), np.zeros((4, 2, 5)))
