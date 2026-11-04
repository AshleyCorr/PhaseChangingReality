"""Build one T6 tempering config per size, with a coupling ladder sized for that size.

WHY PER SIZE. The first attempt used a single 18-rung ladder for all three sizes. At N = 36 it
worked (about 50 round trips per replica). At N = 64 it gave ZERO round trips, and the reason is
visible in the run: swap acceptance along the ladder sat at 0.62 to 0.72 everywhere except at the
transition, where it collapsed to 0.27 as the order parameter jumped from 0.76 to 1.18 to 1.46.
That collapse is not a bug, it is the first-order transition itself -- adjacent rungs stop sharing
any energies in common, so no replica can cross. But a severed ladder cannot measure the thing,
so the rungs have to be placed with the gap in mind, and the gap grows with the size.

THE RULE. Two rungs exchange easily when the tilt between them times the energy gap between them
is of order one: d(1/g) * dH ~ 1. The latent heat measured at N = 36 is about 5.8 per point, so
dH ~ 5.8 N and the spacing must fall like 1/N. Taking the product to be 1.5:

    d(1/g) = 1.5 / (5.8 N)

Rungs are placed uniformly in 1/g, not in g, because 1/g is what multiplies the energy in the
Boltzmann factor -- uniform steps in g crowd the hot end and starve the cold one. The fine
spacing is spent in a window around the transition and the spacing is tripled outside it, since
away from the transition there is no gap to bridge.

WHERE THE TRANSITION IS. Measured at N = 36 (g = 7.45, four replicas agreeing to 1 per cent) and
read off the order-parameter jump at N = 64 (between 7.0 and 6.3, so about 6.7). For N = 100 it
is extrapolated: g_c falls by about 1.3 per unit of ln N between the two measured sizes, giving
about 6.1. An extrapolated centre is a guess, so the window is wide enough to be wrong by a good
margin, and the run reports where the humps actually balanced rather than assuming this number.

THE COLD TAIL IS SHORT ON PURPOSE. In the first attempt the coldest rungs had a move acceptance
of 0.0001 -- frozen, contributing nothing but extra length for a replica to diffuse across. The
cold phase is fully converted by about g = 5 at these sizes, so the ladder stops shortly after.
"""
import json
import sys
from pathlib import Path

LATENT_PER_POINT = 5.8     # measured at N = 36, lambda = 0
TILT_TIMES_GAP = 1.5       # target for d(1/g) * dH between neighbouring rungs
WINDOW = 0.035             # half-width in 1/g of the finely spaced region
COARSE = 3.0               # spacing multiplier outside that window
G_HOT, G_COLD = 14.0, 4.5  # ends of the ladder

SIZES = {36: 7.45, 64: 6.70, 100: 6.12}     # N -> where the transition is expected


def ladder(n, g_c):
    step = TILT_TIMES_GAP / (LATENT_PER_POINT * n)
    lo, hi = 1.0 / g_c - WINDOW, 1.0 / g_c + WINDOW
    xs, x = [], 1.0 / G_HOT
    while x <= 1.0 / G_COLD + 1e-12:
        xs.append(x)
        x += step if lo - 1e-12 <= x <= hi else step * COARSE
    return [round(1.0 / x, 4) for x in xs]


def main(lam, tag, out=Path("configs")):
    for n, g_c in SIZES.items():
        side = int(round(n ** 0.5))
        assert side * side == n, n
        gs = ladder(n, g_c)
        # "t6b" marks T6 run under amendment 1 (phi as the reaction coordinate, joint (S, X)
        # storage). The pre-amendment results keep their own names and are not overwritten.
        name = "t6b_%s_n%d" % (tag, n)
        cfg = {
            "_purpose": ("T6 at lambda = %g, size %d, one size per config so the coupling ladder "
                         "can be sized for the gap at THIS size. Ladder built by "
                         "scripts/make_t6_ladder.py; see that file for why a single shared ladder "
                         "gave zero round trips at N = 64. %d rungs, spaced %.5f in 1/g through "
                         "the window around g_c = %g and three times that outside it. Everything "
                         "else matches t6_lam0_control so the three sizes and the three lambdas "
                         "stay comparable at matched sensitivity."
                         % (lam, n, len(gs), TILT_TIMES_GAP / (LATENT_PER_POINT * n), g_c)),
            "name": name,
            "sides": [side],
            "couplings": gs,
            "cap": None,
            "lambda": lam,
            "acceptance": "metropolis",
            "start": "melt",
            "n_melt": 1000,
            "sweeps_per_round": 5,
            "rounds_equil": 4000,
            "rounds_meas": 12000,
            "replicas": 4,
            "seed": 20260921 + n,
        }
        p = out / (name + ".json")
        p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print("%-18s %2d rungs  g %.2f .. %.2f  step 1/g %.5f" % (name, len(gs), gs[0], gs[-1],
                                                                 TILT_TIMES_GAP / (LATENT_PER_POINT * n)))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0,
         sys.argv[2] if len(sys.argv) > 2 else "lam0")
