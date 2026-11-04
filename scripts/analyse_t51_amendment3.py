"""T51, Amendment 3's reported-not-scored items: the shape of the fall, the size of what is left, and the pieces. Usage:

    python scripts/analyse_t51_amendment3.py [results_dir]

Implements the first item of PREREGISTRATION.md section T51, Amendment 3 (written 2026-10-06, before any result was
read), which `scripts/analyse_t51.py` (the scoring, Amendments 1 and 2) does not cover. Nothing here is scored; the
verdict, P1 and P2 are analyse_t51's. The cells and replicas are read with analyse_t51's own loaders, so what is
kept and what is paired is the same here as there.

  The shape of the fall     At the verdict's length (L = 256) over the cells from 10,000 fair sweeps upward, the mean
                            columns per replica C(t) and the mean frozen share F(t) are each fitted to a power law,
                            ln y = alpha ln t + beta, by least squares on the cell means. The error of alpha comes from
                            resampling replica ids with replacement (BOOT_N resamples, seeded with BOOT_SEED): a
                            resample keeps the pairing, since one drawn id is used in every cell that holds it. A cell
                            whose resampled mean is zero cannot enter a log and that resample is left out, counted.
  The extrapolation         The cooling time at which the frozen-share fit would reach O89's share at 1 MeV,
                            SHARE_1MEV = 7e-7 (ASSUMPTIONS O18, O89): t* = exp((ln SHARE_1MEV - beta) / alpha). It is an
                            extrapolation over many decades from four cells and is printed as one.
  The size distribution     From every saved final graph (t51_*_adj/*.npz): the points in each connected piece of
                            points not at d = 2 at the end of the hold, as a histogram per (L, t_cool), with the
                            columns (four points all at d = 1) told from the rest, as analyse_t37.leftovers tells them.
  The pieces                The connected pieces of the whole graph at the end of the hold (does the opening split
                            the space, O100), per (L, t_cool).
  Every decade              R(t) = C(10 t) / C(t) on shared replicas at every t with a 10 t cell, not only the two the
                            verdict uses: t = 1,000, 10,000, 30,000 at L = 256.

Pure functions of parsed rows and loaded graphs, tested in tests/test_t51_amendment3.py.
"""
import glob
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.connectivity import component_labels                       # noqa: E402
from graphity.dimension import local_dimension, piece_labels             # noqa: E402
from analyse_t51 import cells, decade, kept, load, VERDICT_L             # noqa: E402

FIT_FROM = 10000                     # the fit runs over the cells from this cooling time upward (Amendment 3)
SHARE_1MEV = 7e-7                    # dark matter's share of the energy at 1 MeV, O89 (settled physics applied by us)
BOOT_N, BOOT_SEED = 2000, 351
DECADE_STARTS = (1000, 10000, 30000)


def frozen_share(r):
    """The energy left per point at the end of the hold over the release per point, 4(lambda - 1)."""
    return r["h1"] / (r["N"] * 4.0 * (r["lam"] - 1.0))


def power_law(ts, ys):
    """(alpha, beta) of ln y = alpha ln t + beta by least squares; nan if any y is not positive or fewer than two."""
    ts, ys = np.asarray(ts, dtype=float), np.asarray(ys, dtype=float)
    if len(ts) < 2 or (ys <= 0).any():
        return math.nan, math.nan
    alpha, beta = np.polyfit(np.log(ts), np.log(ys), 1)
    return float(alpha), float(beta)


def fit_with_error(by_t, value):
    """Fit the cell means of `value` (a function of a replica record) to a power law in t_cool, with alpha's error.

    `by_t` maps t_cool to the kept records of its cell. Returns dict(alpha, beta, se, n_left_out, means): the fit on
    the cell means, alpha's standard error over BOOT_N resamples of replica ids (a drawn id is used in every cell
    that holds it), the resamples left out because a cell's mean was not positive, and the means fitted.
    """
    ts = sorted(by_t)
    means = {t: float(np.mean([value(r) for r in by_t[t]])) for t in ts}
    alpha, beta = power_law(ts, [means[t] for t in ts])
    ids = sorted({r["replica"] for t in ts for r in by_t[t]})
    table = {t: {r["replica"]: value(r) for r in by_t[t]} for t in ts}
    rng = np.random.default_rng(BOOT_SEED)
    alphas, left_out = [], 0
    for _ in range(BOOT_N):
        draw = rng.choice(ids, size=len(ids), replace=True)
        resampled = []
        for t in ts:
            vals = [table[t][i] for i in draw if i in table[t]]
            resampled.append(float(np.mean(vals)) if vals else math.nan)
        a, _ = power_law(ts, resampled) if not any(math.isnan(v) for v in resampled) else (math.nan, math.nan)
        if math.isnan(a):
            left_out += 1
        else:
            alphas.append(a)
    se = float(np.std(alphas, ddof=1)) if len(alphas) > 1 else math.nan
    return dict(alpha=alpha, beta=beta, se=se, n_left_out=left_out, means=means)


def time_to_reach(fit, target):
    """The t at which exp(beta) t^alpha = target; nan unless the fit falls (alpha < 0)."""
    if math.isnan(fit["alpha"]) or fit["alpha"] >= 0:
        return math.nan
    return math.exp((math.log(target) - fit["beta"]) / fit["alpha"])


def piece_sizes(adj):
    """(sizes of the connected pieces of points not at d = 2, as a sorted list; how many of them are columns)."""
    d = local_dimension(adj)
    labels = piece_labels(adj, d != 2)
    sizes, columns = [], 0
    for k in range(labels.max() + 1):
        members = np.flatnonzero(labels == k)
        sizes.append(int(len(members)))
        if len(members) == 4 and (d[members] == 1).all():
            columns += 1
    return sorted(sizes), columns


def whole_graph_pieces(adj):
    return int(component_labels(adj).max()) + 1


def read_saved(results):
    """{(L, t_cool): [(replica, sizes, columns, pieces of the whole graph), ...]} from every saved final graph."""
    out = defaultdict(list)
    for f in sorted(glob.glob(str(Path(results) / "t51_*_adj" / "*.npz"))):
        z = np.load(f)
        adj = z["adj"]
        sizes, columns = piece_sizes(adj)
        out[(adj.shape[0] // 4, int(z["t_cool"]))].append((int(z["replica"]), sizes, columns, whole_graph_pieces(adj)))
    return dict(out)


def size_histogram(records):
    """Counter of piece size over every saved graph of a cell, and the share of graphs that are one piece."""
    hist = Counter()
    for _, sizes, _, _ in records:
        hist.update(sizes)
    one = sum(1 for _, _, _, p in records if p == 1)
    return hist, one


def main(results="results"):
    rows = load(results)
    by_cell = cells(rows)
    print("T51, Amendment 3's reported-not-scored items (PREREGISTRATION T51, Amendment 3, written 2026-10-06 before any "
          "result was read). Nothing here is scored.")
    at_l = {t: kept(recs) for (length, t), recs in by_cell.items() if length == VERDICT_L and t >= FIT_FROM}
    print("\nThe shape of the fall, L = %d, cells from %d fair sweeps upward: %s" % (VERDICT_L, FIT_FROM, sorted(at_l)))
    fit_c = fit_with_error(at_l, lambda r: float(r["c1"]))
    fit_f = fit_with_error(at_l, frozen_share)
    for name, fit in (("columns per replica C(t)", fit_c), ("frozen share F(t)", fit_f)):
        print("  %-26s cell means %s" % (name, ", ".join("%d: %.4g" % (t, m) for t, m in sorted(fit["means"].items()))))
        print("  %-26s power law exponent alpha = %.3f +- %.3f (resamples left out: %d of %d); "
              "y = %.3g * t^alpha" % ("", fit["alpha"], fit["se"], fit["n_left_out"], BOOT_N, math.exp(fit["beta"])
                                      if not math.isnan(fit["beta"]) else math.nan))
    t_star = time_to_reach(fit_f, SHARE_1MEV)
    print("  EXTRAPOLATION, marked as one: the frozen-share fit reaches O89's share at 1 MeV (%.0e) at t_cool = %.2g "
          "fair sweeps (%.1f decades beyond the last cell)" % (SHARE_1MEV, t_star,
                                                              math.log10(t_star / max(at_l)) if t_star > 0 else math.nan))

    print("\nEvery decade available, L = %d, R(t) = C(10 t) / C(t) on shared replicas:" % VERDICT_L)
    for t in DECADE_STARTS:
        a, b = by_cell.get((VERDICT_L, t)), by_cell.get((VERDICT_L, 10 * t))
        if a is None or b is None:
            print("  R(%d): a cell is missing" % t)
            continue
        d = decade(a, b)
        print("  R(%-6d) = %s (columns %d over %d, on %d shared replicas)"
              % (t, "undefined (C(t) = 0)" if d["R"] is None else "%.3f +- %.3f" % (d["R"], d["se"]), d["c_10t"], d["c_t"], d["n"]))

    saved = read_saved(results)
    print("\nThe size distribution of what is left at the end of the hold, from the saved final graphs "
          "(points per connected piece not at d = 2; columns are pieces of four all at d = 1), and the pieces of the "
          "whole graph:")
    for (length, t), recs in sorted(saved.items()):
        hist, one = size_histogram(recs)
        total = sum(hist.values())
        columns = sum(c for _, _, c, _ in recs)
        print("  L=%-4d t_cool=%-7d graphs %3d | pieces %4d, of them columns %4d | sizes: %s | whole graph one piece in "
              "%d of %d" % (length, t, len(recs), total, columns,
                            ", ".join("%d:%d" % (s, k) for s, k in sorted(hist.items())) or "none", one, len(recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
