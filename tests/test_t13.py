"""The T13 analyser, checked against data whose answer is put in by hand.

The project's rule for a verdict script: build runs where the right answer is known in advance and
check the script returns it, both when the criteria should pass and when each should fail. The two
amendments of 23 September 2026 get their own checks, including the control that decided
amendment 1 — a windowed jump rule must not manufacture a jump in a curve that is smooth.
"""
import pytest

from scripts.analyse_t13 import (AGREE, JUMP, agreement, largest_move, read_e, read_p, verdict)

# 40 couplings from 12 to 1.5, geometric: the real protocol P grid, 5.3 % apart.
P_GRID = [12.0 * (1.5 / 12.0) ** (i / 39) for i in range(40)]
# 20 rungs from 9.0 to 2.2: the real protocol E ladder at N = 196, 7.2 % apart.
E_GRID = [9.0 * (2.2 / 9.0) ** (i / 19) for i in range(20)]


def descent(g):
    """A smooth equilibrium curve: phi = 0.25 in the random phase, 1.0 when settled."""
    return 0.25 + 0.75 * (12.0 / g - 1.0) / (12.0 / 1.5 - 1.0)


def seq_rows(n=196, replicas=4, collapse_at=3.5, steps=1, smooth=False):
    """Protocol P. Heating from the lattice, the ascent sits at phi = 1.0 while the lattice
    survives, and rejoins the descent over `steps` couplings once past `collapse_at`. steps=1 is a
    single-step collapse; larger values spread the same move, which is what dilutes it per step."""
    rows = []
    above = [g for g in P_GRID if g > collapse_at]      # descending in g
    trans = above[-steps:] if steps <= len(above) else above
    for rep in range(replicas):
        for g in P_GRID:
            desc = descent(g)
            if smooth or g > collapse_at and g not in trans:
                asc = desc
            elif g <= collapse_at:
                asc = 1.0
            else:
                frac = (trans.index(g) + 1) / len(trans)
                asc = desc + frac * (1.0 - desc)
            for leg, phi in (("cool", desc), ("heat", asc)):
                rows.append(dict(N=str(n), replica=str(rep), leg=leg, g=str(g), phi=str(phi)))
    return rows


def temper_rows(n=196, replicas=4, trips=8, swap=0.4, jump_at=None, offset=0.0, spread=0.0):
    rows = []
    for rep in range(replicas):
        for g in E_GRID:
            phi = max(0.25, min(1.0, 0.25 + 0.75 * (9.0 - g) / 6.8)) + offset
            if jump_at is not None and g < jump_at:
                phi = min(1.0, phi + 0.4)
            phi += spread * rep
            rows.append(dict(N=str(n), replica=str(rep), g=str(g), phi=str(phi),
                             round_trips=str(trips), swap_rate=str(swap)))
    return rows


def sized(n, **p):
    kw = dict(p)
    e_kw = {k: kw.pop(k) for k in list(kw) if k.startswith(("melt_", "torus_"))}
    melt = {k[5:]: v for k, v in e_kw.items() if k.startswith("melt_")}
    torus = {k[6:]: v for k, v in e_kw.items() if k.startswith("torus_")}
    return {"p": read_p(seq_rows(n=n, **kw)),
            "e": {"melt": read_e(temper_rows(n=n, **melt), n),
                  "torus": read_e(temper_rows(n=n, **torus), n)}}


# ---------------------------------------------------------------- the jump rule and its control

def test_a_single_step_collapse_is_a_jump_both_ways_of_reading_it():
    gs = [4.0, 3.8, 3.6]
    ph = [0.70, 0.70, 1.00]
    assert largest_move(gs, ph)[0] == pytest.approx(0.30)
    assert largest_move(gs, ph, window=None)[0] == pytest.approx(0.30)


def test_a_collapse_split_over_two_couplings_is_missed_per_step_and_caught_windowed():
    """Amendment 1's whole point: the same physics, diluted only by a finer grid."""
    gs = [4.0, 3.79, 3.59]                      # each neighbour ~5.3 % apart, ends 10.2 % apart
    ph = [0.70, 0.85, 1.00]
    assert largest_move(gs, ph, window=None)[0] == pytest.approx(0.15)   # no jump per step
    assert largest_move(gs, ph)[0] == pytest.approx(0.30)                # jump windowed
    assert largest_move(gs, ph)[0] > JUMP


def test_the_window_refuses_to_reach_beyond_twelve_percent():
    gs = [4.0, 3.6, 3.2]                        # 4.0 -> 3.2 is 20 % apart, outside the window
    ph = [0.70, 0.85, 1.00]
    assert largest_move(gs, ph)[0] == pytest.approx(0.15)


def test_the_control_a_smooth_curve_is_not_turned_into_a_jump():
    """The check that decided amendment 1. If this fails the amendment is wrong."""
    for rows, label in ((seq_rows(smooth=True), "P both legs"), (temper_rows(), "E")):
        if label == "E":
            gs, ph = read_e(rows, 196)["curve"]
        else:
            gs, ph = [float(r["g"]) for r in rows if r["leg"] == "cool" and r["replica"] == "0"], \
                     [float(r["phi"]) for r in rows if r["leg"] == "cool" and r["replica"] == "0"]
        assert largest_move(gs, ph)[0] < JUMP, label


# ---------------------------------------------------------------------------- protocol P reading

def test_hysteresis_is_found_when_the_ascent_rides_above_the_descent():
    p = read_p(seq_rows(collapse_at=3.0, steps=3))
    assert p["hysteresis_met"] and p["hysteresis"][0] > 0.15


def test_no_hysteresis_when_the_two_legs_coincide():
    p = read_p(seq_rows(smooth=True))
    assert not p["hysteresis_met"]


def test_replicas_collapsing_at_different_couplings_split_the_two_readings():
    """The ambiguity the script refuses to resolve: every replica jumps, the mean does not."""
    rows = []
    for rep, at in enumerate((5.0, 4.0, 3.2, 2.6)):
        for r in seq_rows(replicas=1, collapse_at=at, steps=1):
            r["replica"] = str(rep)
            rows.append(r)
    p = read_p(rows)
    assert p["legs"]["heat"]["replicas_jumping"] == 4
    assert not p["legs"]["heat"]["mean_jump"]


# ------------------------------------------------------------------- gates and amendment 2

def test_a_start_below_its_round_trip_gate_fails_the_gate():
    assert not read_e(temper_rows(trips=0), 196)["gate"]
    assert read_e(temper_rows(trips=5), 196)["gate"]
    assert not read_e(temper_rows(trips=4), 196)["gate"]      # 5 is the gate at this size
    assert read_e(temper_rows(trips=3), 484)["gate"]          # 3 is the gate at the larger ones


def test_a_swap_rate_outside_the_band_fails_the_gate():
    assert not read_e(temper_rows(swap=0.05), 196)["gate"]
    assert not read_e(temper_rows(swap=0.9), 196)["gate"]


def test_agreement_is_not_evaluable_when_only_one_start_passes():
    """Amendment 2: this must be None, not False. A stalled start is not a disagreement."""
    starts = {"melt": read_e(temper_rows(trips=8), 196),
              "torus": read_e(temper_rows(trips=0, offset=0.3), 196)}
    got, worst, rung = agreement(starts)
    assert got is None and worst is None


def test_two_passing_starts_that_differ_disagree():
    starts = {"melt": read_e(temper_rows(trips=8), 196),
              "torus": read_e(temper_rows(trips=8, offset=0.2), 196)}
    got, worst, _ = agreement(starts)
    assert got is False and worst > AGREE


def test_two_passing_starts_that_coincide_agree():
    starts = {"melt": read_e(temper_rows(trips=8), 196),
              "torus": read_e(temper_rows(trips=8), 196)}
    got, worst, _ = agreement(starts)
    assert got is True and worst <= AGREE


# ------------------------------------------------------------------------------- the verdicts

def test_one_size_however_clean_returns_no_verdict():
    lines, out = verdict({196: sized(196, collapse_at=3.0, steps=1, torus_trips=0)})
    assert out == "NO VERDICT", "\n".join(lines)


def test_two_sizes_of_hysteresis_and_ascent_jump_with_smooth_equilibrium_is_a_metastable_branch():
    sizes = {n: sized(n, collapse_at=3.0, steps=1, torus_trips=0) for n in (196, 484)}
    lines, out = verdict(sizes)
    assert out == "METASTABLE BRANCH", "\n".join(lines)


def test_a_jump_in_the_equilibrium_curve_from_a_passing_start_is_an_equilibrium_jump():
    sizes = {n: sized(n, collapse_at=3.0, steps=1,
                      melt_jump_at=6.0, torus_jump_at=6.0) for n in (196, 484)}
    lines, out = verdict(sizes)
    assert out == "EQUILIBRIUM JUMP", "\n".join(lines)


def test_everything_smooth_is_inconclusive():
    sizes = {n: sized(n, smooth=True, torus_trips=0) for n in (196, 484)}
    lines, out = verdict(sizes)
    assert out == "INCONCLUSIVE", "\n".join(lines)


def test_the_majority_decides_where_the_mean_would_erase_the_collapse():
    """Amendment 3. Every replica collapses in one step, at its own coupling, so the mean is
    smooth and the majority is unanimous. The majority must win, and the mean must still be said."""
    rows = []
    for rep, at in enumerate((5.0, 4.0, 3.2, 2.6)):
        for r in seq_rows(replicas=1, collapse_at=at, steps=1):
            r["replica"] = str(rep)
            rows.append(r)
    p = read_p(rows)
    assert p["legs"]["heat"]["replicas_jumping"] == 4 and not p["legs"]["heat"]["mean_jump"]

    sizes = {}
    for n in (196, 484):
        sizes[n] = {"p": p,
                    "e": {"melt": read_e(temper_rows(n=n), n),
                          "torus": read_e(temper_rows(n=n, trips=0), n)}}
    lines, out = verdict(sizes)
    assert out == "METASTABLE BRANCH", "\n".join(lines)
    assert any("replica-mean reading differs" in ln for ln in lines), "\n".join(lines)


def test_a_minority_of_replicas_jumping_is_not_a_jump():
    """One replica in four is not the protocol showing a jump."""
    rows = []
    for rep, at in enumerate((4.0, None, None, None)):
        for r in seq_rows(replicas=1, collapse_at=at or 1.0, steps=1, smooth=at is None):
            r["replica"] = str(rep)
            rows.append(r)
    p = read_p(rows)
    assert p["legs"]["heat"]["replicas_jumping"] == 1
    sizes = {n: {"p": p, "e": {"melt": read_e(temper_rows(n=n), n)}} for n in (196, 484)}
    lines, out = verdict(sizes)
    assert out == "INCONCLUSIVE", "\n".join(lines)
