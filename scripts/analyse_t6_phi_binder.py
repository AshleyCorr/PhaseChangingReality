"""T6 criterion 3 at lambda >= 1, read from the cumulant of phi (amendment 5, option (a)). Usage:

    python scripts/analyse_t6_phi_binder.py lam1      # also lam125, lam15

Implements the enacted wording of PREREGISTRATION.md T6 amendment 5 and makes no choices of its
own. It reads N = 36 from the t6b run (which already met gate 4) and N = 64, 100 from the t6c
reruns, all from the joint (S, X) histograms the runs saved.

WHY PHI. The Binder energy cumulant 1 - <H^4>/3<H^2>^2 measures the shape of the energy
distribution relative to the energy zero. At lambda >= 1 the cold phase, the flat sheet, sits at
exactly H = 0, so the cumulant collapses whatever the distribution looks like (ASSUMPTIONS O17).
phi = S/N stays well away from zero in every phase (about 0.12 hot, 1 on the sheet), so its
cumulant keeps the meaning the pre-registration gave the energy one: a minimum that approaches
2/3 as N grows for one hump, and stays below 2/3 for two.

WHAT IS COMPUTED. For every rung's histogram, P(S) is reweighted to couplings within the φ
analysis's usual range (analyse_t6_phi.profile, the same guards), and at each admissible coupling
U = 1 - <phi^4>/3<phi^2>^2. Per replica, U_min is the minimum over every rung and coupling; per
size, the mean over replicas with their spread. Criterion 3 reads as approaching 2/3 exactly when
the gap 2/3 - U_min shrinks at each step from the smallest size to the largest.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t6_hist as H          # noqa: E402  (shared guards)
import analyse_t6_phi as P           # noqa: E402  (the reweighted profile along S)

TWO_THIRDS = 2.0 / 3.0
SCAN_POINTS = 121


def phi_cumulant(s_values, weights, n):
    """U = 1 - <phi^4> / 3<phi^2>^2 for phi = S/N distributed with the given weights."""
    w = np.asarray(weights, dtype=float)
    phi = np.asarray(s_values, dtype=float) / n
    m2 = (w * phi ** 2).sum() / w.sum()
    m4 = (w * phi ** 4).sum() / w.sum()
    return 1.0 - m4 / (3.0 * m2 * m2)


def rung_minimum(d):
    """The smallest U over the couplings this rung's histogram may be reweighted to."""
    g0 = float(d["g"])
    n = int(d["N"])
    floor = max(H.MIN_ESS, H.MIN_ESS_FRAC * float(d["sx_counts"].sum()))
    best = None
    for g in np.linspace(g0 * (1 - H.MAX_SHIFT), g0 * (1 + H.MAX_SHIFT), SCAN_POINTS):
        grid, wsum, _, _, ess = P.profile(d, g)
        if ess < floor or wsum.sum() <= 0:
            continue
        u = phi_cumulant(grid, wsum, n)
        if best is None or u < best[0]:
            best = (u, g)
    return best


def approaches_two_thirds(sizes, u_means):
    """Criterion 3 as enacted: the gap 2/3 - U_min shrinks at each step, smallest size to largest."""
    order = np.argsort(sizes)
    gaps = [TWO_THIRDS - u_means[i] for i in order]
    return len(gaps) >= 2 and all(b < a for a, b in zip(gaps, gaps[1:]))


def minima_from_run(name, out_dir="results", keep_sizes=None):
    """{N: {replica: (U_min, g at the minimum)}} for one run's saved histograms."""
    base = Path(out_dir) / (name + ".csv")
    rows = [r for r in csv.DictReader(open(base, newline="")) if r.get("round_trips")]
    store = Path(out_dir) / (name + "_hist")
    got = defaultdict(dict)
    for r in rows:
        n, rep, k = int(r["N"]), int(r["replica"]), int(r["k"])
        if keep_sizes is not None and n not in keep_sizes:
            continue
        f = store / ("N%d_rep%d_k%d.npz" % (n, rep, k))
        if not f.exists():
            continue
        m = rung_minimum(np.load(f))
        if m is None:
            continue
        if rep not in got[n] or m[0] < got[n][rep][0]:
            got[n][rep] = m
    return got


def main(tag, out_dir="results"):
    per_size = {}
    for name, sizes in (("t6b_" + tag, {36}), ("t6c_%s_n64" % tag, {64}), ("t6c_%s_n100" % tag, {100})):
        per_size.update(minima_from_run(name, out_dir, sizes))
    print("T6 amendment 5 (a), %s: criterion 3 from U_phi = 1 - <phi^4>/3<phi^2>^2.\n" % tag)
    print("%-6s %-5s %-10s %-9s %-10s %s" % ("N", "reps", "U_min", "spread", "2/3 - U", "g at min"))
    sizes, means = [], []
    for n in sorted(per_size):
        vals = np.array([v[0] for v in per_size[n].values()])
        gs = np.array([v[1] for v in per_size[n].values()])
        sizes.append(n)
        means.append(vals.mean())
        print("%-6d %-5d %-10.5f %-9.5f %-10.5f %.2f-%.2f"
              % (n, len(vals), vals.mean(), vals.max() - vals.min(), TWO_THIRDS - vals.mean(),
                 gs.min(), gs.max()))
    ok = approaches_two_thirds(np.array(sizes), np.array(means))
    print("\nCriterion 3 (gap to 2/3 shrinks at each step, %s): %s"
          % (" -> ".join(str(s) for s in sizes), "MET" if ok else "NOT MET"))
    print("Verdict under amendment 5 (a), criteria 1 and 2 as stated in PREREGISTRATION T6: %s"
          % ("NO EVIDENCE OF FIRST ORDER AT THESE SIZES" if ok else "INCONCLUSIVE"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lam1")
