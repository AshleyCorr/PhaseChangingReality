"""The pre-registered T6 observables, read along phi instead of along the energy. Usage:

    python scripts/analyse_t6_phi.py t6_lam0_n36

Implements PREREGISTRATION.md section T6 as amended on 21 September 2026 (amendment 1), and
makes no choices of its own.

WHY PHI AND NOT THE ENERGY. The energy H = 16(N - S) + 4*lambda*X is a sum of two integers with
two quanta, so which energies are reachable, and how many ways each is reached, is arithmetic
between 16 and 4*lambda. Where 4*lambda does not divide 16 the histogram grows teeth at the
period of that arithmetic, and a pair of teeth reads exactly like a pair of humps. S is an
integer with unit spacing at every lambda, so a histogram in S has no comb in it. This is a
property of the two coordinates, not of any result.

WHAT IS MEASURED, AND IN WHICH COORDINATE.

  The transition coupling and the free-energy barrier are read along phi = S/N: the coupling
  where the two humps of P(phi) carry equal weight, and the depth of the valley between them.

  The latent heat stays an ENERGY difference, because that is what a latent heat is. phi only
  says which sweeps belong to which phase; their energies are then compared directly. This is
  the one place where mixing the two coordinates is deliberate rather than sloppy.

REWEIGHTING. From the joint (S, X) histogram at coupling g, the distribution at g' follows by
weighting each (S, X) by exp(-H(1/g' - 1/g)). The guards from the energy analysis carry over
unchanged and for unchanged reasons: an unvisited S is not a deep S, a valley must beat the
counting error on the bins it is built from, and a reweight may not concentrate the answer onto
a handful of sweeps. The coarse-graining veto does NOT carry over -- it existed only to catch
the comb, and under phi there is no comb to catch.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t6_hist as H          # noqa: E402  (shared guards; see its header)

ENERGY_PER_SQUARE = 16.0
ENERGY_PER_SURPLUS = 4.0


def energy(s, x, n, lam):
    return ENERGY_PER_SQUARE * (n - s) + ENERGY_PER_SURPLUS * lam * x


def profile(d, g_to):
    """P(S) and the mean energy in each S, reweighted from the run's coupling to g_to."""
    n, lam, g0 = int(d["N"]), float(d["lam"]), float(d["g"])
    s, x, c = d["s_bin"].astype(int), d["x_bin"].astype(int), d["sx_counts"].astype(float)
    h = energy(s, x, n, lam)
    lw = -h * (1.0 / g_to - 1.0 / g0)
    lw -= lw.max()
    w = np.exp(lw) * c
    lo, hi = s.min(), s.max()
    grid = np.arange(lo, hi + 1)
    wsum = np.bincount(s - lo, weights=w, minlength=len(grid))
    esum = np.bincount(s - lo, weights=w * h, minlength=len(grid))
    raw = np.bincount(s - lo, weights=c, minlength=len(grid))
    ess = (w.sum() ** 2 / (w * w / np.maximum(c, 1e-300)).sum()) if w.sum() > 0 else 0.0
    return grid, wsum, esum, raw, ess


def observables(d, rows_total):
    """Scan for the coupling where the two phases balance; return the pre-registered four."""
    n, g0 = int(d["N"]), float(d["g"])
    floor = max(H.MIN_ESS, H.MIN_ESS_FRAC * float(d["sx_counts"].sum()))
    best = None
    for g in np.linspace(g0 * (1 - H.MAX_SHIFT), g0 * (1 + H.MAX_SHIFT), 241):
        grid, wsum, esum, raw, ess = profile(d, g)
        if ess < floor:
            continue
        keep = H.visited_stretch(grid.astype(float), raw)
        if len(keep[0]) < 2 * H.MIN_LEVELS_APART:
            continue
        i0 = int(np.searchsorted(grid, keep[0][0]))
        i1 = i0 + len(keep[0])
        gs, ws, es, rs = grid[i0:i1], wsum[i0:i1], esum[i0:i1], raw[i0:i1]
        if ws.sum() <= 0:
            continue
        p = ws / ws.sum()
        hh = H.two_humps(p, rs)
        if hh is None:
            continue
        a, b, v, ps = hh
        below = ps[:v + 1].sum()
        imbalance = abs(np.log(max(below, 1e-300) / max(1 - below, 1e-300)))
        if best is None or imbalance < best[0]:
            best = (imbalance, g, a, b, v, ps, gs, ws, es, n)
    if best is None:
        return None
    imbalance, g, a, b, v, ps, gs, ws, es, n = best
    # the latent heat is an energy difference between the two phases phi has just separated
    w_lo, w_hi = ws[:v + 1].sum(), ws[v + 1:].sum()
    e_lo = es[:v + 1].sum() / w_lo if w_lo > 0 else np.nan
    e_hi = es[v + 1:].sum() / w_hi if w_hi > 0 else np.nan
    return dict(g_c=g, shift=abs(g - g0) / g0, imbalance=imbalance,
                phi_lo=float(gs[a]) / n, phi_hi=float(gs[b]) / n,
                d_phi=float(gs[b] - gs[a]) / n,
                latent=abs(float(e_hi - e_lo)) / n,
                barrier=float(np.log(min(ps[a], ps[b])) - np.log(ps[v])),
                levels_apart=int(b - a))


def main(name, out_dir="results"):
    base = Path(out_dir) / (name + ".csv")
    if not base.exists():
        base = Path(out_dir) / (name + ".csv.partial")
    rows = [r for r in csv.DictReader(open(base, newline="")) if r.get("round_trips")]
    store = Path(out_dir) / (name + "_hist")
    per_size, per_rung = defaultdict(list), defaultdict(list)

    for r in rows:
        n, rep, k = int(r["N"]), int(r["replica"]), int(r["k"])
        f = store / ("N%d_rep%d_k%d.npz" % (n, rep, k))
        if not f.exists():
            continue
        d = np.load(f)
        if "s_bin" not in d:
            print("  %s has no joint (S, X) histogram -- it predates amendment 1; re-run it."
                  % f.name)
            return
        got = observables(d, len(rows))
        if got:
            got.update(N=n, rep=rep, k=k, g0=float(r["g"]), trips=int(r["round_trips"]),
                       binder=float(r["binder"]))
            per_size[n].append(got)
            per_rung[(n, k)].append(got)

    print("T6 along phi, %s. PREREGISTRATION.md T6 as amended 21 Sep 2026.\n" % name)

    # The falsification clause: a transition too weak to see is reported as a BOUND, not an
    # absence. Two phases separated in energy by less than about twice the width of one hump
    # do not produce a visible dip, so the largest latent heat that could hide in a single hump
    # is about 2 sigma_E / N, taken where the energy fluctuations are largest (the specific-heat
    # peak). Reported for every size; where two humps ARE found it should sit below the latent
    # heat measured, which is a consistency check on both.
    bound = defaultdict(float)
    for r in rows:
        if r.get("e_var"):
            n = int(r["N"])
            bound[n] = max(bound[n], 2.0 * float(r["e_var"]) ** 0.5 / n)
    print("Largest latent heat that could hide in a single hump, per point, by size:")
    for n in sorted(bound):
        print("   N=%-4d  < %.3f   (lambda = 0 measured 5.8 at N = 36, 9.5 at N = 64, for scale)"
              % (n, bound[n]))
    print()

    if not per_size:
        print("  no two-hump structure in phi at any coupling, size or replica.")
        return
    print("%-6s %-5s %-9s %-8s %-9s %-9s %-9s %-9s %-7s"
          % ("N", "reps", "g_c", "shift", "phi lo", "phi hi", "latent", "barrier", "trips"))
    for n in sorted(per_size):
        by_rep = defaultdict(list)
        for g in per_size[n]:
            by_rep[g["rep"]].append(g)
        picks = [min(v, key=lambda d: d["imbalance"]) for v in by_rep.values()]
        print("%-6d %-5d %-9.3f %-8.3f %-9.3f %-9.3f %-9.4f %-9.3f %-7d"
              % (n, len(picks), np.mean([p["g_c"] for p in picks]),
                 np.mean([p["shift"] for p in picks]),
                 np.mean([p["phi_lo"] for p in picks]), np.mean([p["phi_hi"] for p in picks]),
                 np.mean([p["latent"] for p in picks]), np.mean([p["barrier"] for p in picks]),
                 min(p["trips"] for p in picks)))

    # The check that mattered most last time: do different rungs agree about the same number?
    print("\nAgreement between ladder rungs (a barrier is a property of the system, so the rung"
          "\nit is approached from must not matter). Spread is (max - min) / mean:")
    for n in sorted(per_size):
        for label in ("g_c", "latent", "barrier"):
            vals = np.array([np.mean([g[label] for g in per_rung[(nn, k)]])
                             for (nn, k) in per_rung if nn == n])
            if len(vals) > 1:
                print("   N=%-4d %-8s %d rungs   min %.4f  max %.4f   spread %.1f%%"
                      % (n, label, len(vals), vals.min(), vals.max(),
                         100 * (vals.max() - vals.min()) / abs(vals.mean())))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "t6_lam0_n36")
