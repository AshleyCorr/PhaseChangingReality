"""T54: the true barrier in three directions. The energy of a slab (4 x L x L, one direction curled) with its curled
direction opened over a patch, exact, against the patch's size. Usage:

    python scripts/exact_t54_patch.py configs/t54_patch_stage1.json [out_dir]

Implements PREREGISTRATION.md section T54 with its Amendments 1 and 2 (the draft of 2026-10-06; the owner's prediction
CRITICAL PATCH; the held-out size L = 16; the tied case). Exact; no run; nothing random.

THE PATCH (the exact definition the draft left to this script). The slab is cqg_d.torus([4, L, L]): axis 0 is the
curled direction (a ring of four at every column (x, y) of the L x L sheet). The chain opens such a ring direction by
re-threading: in T48's saved end states every link an opening created joins a point to a point two steps round the
ring in the neighbouring column (O91; read on 9 October), so that the rings become helices. One switch does it for a
pair of rings in neighbouring columns (x, y) and (x + 1, y):

    remove (3, x, y)-(2, x, y) and (0, x+1, y)-(1, x+1, y);  add (0, x+1, y)-(2, x, y) and (3, x, y)-(1, x+1, y).

A patch of size k is that switch applied to every pair (x, x + 1) for x = x0 .. x0 + k - 1 and every row
y = y0 .. y0 + k - 1: k^2 switches over k + 1 columns and k rows, the rings inside cut twice (opened, d = 3), the
rings at the two ends of each row cut once (the seam). Applied to every pair of every row it gives flat space exactly
(H = 0, every point at d = 3; tested). dH(k) = E(patch) - E(slab), with E = H untied and H + T_f under the tie.

WHAT IS REPORTED, per (L, k, lambda, tie): dH(k); the points at each d; the growth dH(k) - dH(k - 1); and the cheapest
single switch out of the patch state (every switch whose first point lies within NEAR links of a point the patch
changed, priced from scratch by exact_walls_tie_d.kinds), as its cost and kind. Per (L, lambda, tie): the least-squares
fit dH(k) = a k - b k^2 + c over k <= L/2 - 1 (the draft's range), k* = a / (2b), dH* = c + a^2 / (4b), and the verdict
word at that L: FIXED WALL if dH(k) falls for every k >= 1; CRITICAL PATCH if it rises then falls within the k
computed; NO FINITE PATCH if it rises for every k computed. The held-out scoring (L = 16) is done by hand from the
amendment that names the prediction, as the pre-registration says.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                             # noqa: E402
from graphity.cqg_d import _switch, hamiltonian, is_valid, torus             # noqa: E402
from graphity.dimension import local_dimension_d                             # noqa: E402
from graphity.results import ResultWriter                                    # noqa: E402
from graphity.sealed_tie_d import table_total                                # noqa: E402
from exact_hidden_relics import ball                                         # noqa: E402
from exact_walls_tie_d import kinds                                          # noqa: E402

NEAR = 3


def index(length, a, x, y):
    """The vertex number of (a, x, y) in cqg_d.torus([4, length, length]) (last axis fastest)."""
    return ((a % 4) * length + (x % length)) * length + (y % length)


def helix_switch(adj, length, x, y):
    """Open the pair of rings at columns (x, y) and (x + 1, y). Modifies adj in place; returns the four points."""
    u1, v1 = index(length, 0, x + 1, y), index(length, 1, x + 1, y)
    u2, v2 = index(length, 3, x, y), index(length, 2, x, y)
    _switch(adj, u1, v1, u2, v2)
    return u1, v1, u2, v2


def patch(adj0, length, k, x0=0, y0=0):
    """(the slab with a patch of size k opened, the points touched)."""
    adj = adj0.copy()
    touched = set()
    for x in range(x0, x0 + k):
        for y in range(y0, y0 + k):
            touched.update(helix_switch(adj, length, x, y))
    return adj, sorted(touched)


def tie_table(tie, lam, dim=3):
    a = 4.0 * (lam - 1.0)
    if tie == "all_at_the_last":
        return np.array([0.0] + [j * a for j in range(1, dim)] + [0.0])
    return np.zeros(dim + 1)


def energy(adj, lam, tie):
    return float(hamiltonian(adj, lam)) + float(table_total(adj, tie_table(tie, lam)))


def cheapest_move(adj, part, touched, lams, ties, near=NEAR):
    """{(lam, tie): (cost, dS, dX)} of the cheapest single switch whose first point is within `near` of the patch."""
    starts = [v for v in ball(adj, touched, near) if part[v] == 0]
    found = {}
    for u1 in starts:
        for (ds, dx, _, _, change), _ in kinds(adj, part, u1).items():
            if ds == 0 and dx == 0 and not change:
                continue
            for lam in lams:
                for tie in ties:
                    tab = tie_table(tie, lam)
                    dt = sum(float(tab[d]) * c for d, c in change if d < len(tab))
                    cost = -16.0 * ds + 4.0 * lam * dx + dt
                    if (lam, tie) not in found or cost < found[(lam, tie)][0] - 1e-9:
                        found[(lam, tie)] = (cost, int(ds), int(dx))
    return found


def fit(ks, dhs):
    """(a, b, c) of dH = a k - b k^2 + c by least squares; nan if fewer than three points."""
    if len(ks) < 3:
        return (np.nan,) * 3
    m = np.vstack([ks, -np.square(ks, dtype=float), np.ones(len(ks))]).T
    a, b, c = np.linalg.lstsq(m, np.asarray(dhs, dtype=float), rcond=None)[0]
    return float(a), float(b), float(c)


def critical(a, b, c):
    """(k*, dH*) from the fit; (inf, inf) if b <= 0 (no finite patch in the fit)."""
    if not b > 0:
        return float("inf"), float("inf")
    return a / (2.0 * b), c + a * a / (4.0 * b)


def verdict(dhs):
    """FIXED WALL, CRITICAL PATCH or NO FINITE PATCH from dH(k) in order of k (k = 1 first)."""
    rises = [dhs[i + 1] > dhs[i] + 1e-9 for i in range(len(dhs) - 1)]
    if not any(rises):
        return "FIXED WALL"
    if all(rises):
        return "NO FINITE PATCH"
    first_fall = rises.index(False)
    return "CRITICAL PATCH" if not any(rises[first_fall:]) else "RISES AGAIN"


def held_out_fit(rows, lengths=(8, 12)):
    """The held-out prediction by the pre-registration's rule, from the stage-1 rows (dicts with L, k, lam, tie, dH,
    in_fit). Per (lam, tie): the fit to both stage-1 sizes together over their in_fit points, and a range for k* and
    dH* set as the larger of +-10 % and the spread between the fits to each size alone. Returns
    {(lam, tie): dict(a, b, c, k_star, dh_star, k_range, dh_range, verdict, alone)}; a verdict NO FINITE PATCH where
    the joint fit has b <= 0."""
    out = {}
    keys = sorted({(float(r["lam"]), r["tie"]) for r in rows})
    for lam, tie in keys:
        pts = {length: [(int(r["k"]), float(r["dH"])) for r in rows
                        if float(r["lam"]) == lam and r["tie"] == tie and int(r["L"]) == length
                        and str(r["in_fit"]) == "True"] for length in lengths}
        joint = [p for length in lengths for p in pts[length]]
        a, b, c = fit([k for k, _ in joint], [dh for _, dh in joint])
        k_star, dh_star = critical(a, b, c)
        alone = {}
        for length in lengths:
            aa, bb, cc = fit([k for k, _ in pts[length]], [dh for _, dh in pts[length]])
            alone[length] = critical(aa, bb, cc)
        finite = [v for v in alone.values() if all(np.isfinite(v))]

        def span(j, centre):
            if not np.isfinite(centre):
                return (float("inf"), float("inf"))
            spread = (max(v[j] for v in finite) - min(v[j] for v in finite)) / 2.0 if len(finite) > 1 else 0.0
            half = max(0.1 * abs(centre), spread)
            return (centre - half, centre + half)
        out[(lam, tie)] = dict(a=a, b=b, c=c, k_star=k_star, dh_star=dh_star, k_range=span(0, k_star),
                               dh_range=span(1, dh_star), alone=alone,
                               verdict="NO FINITE PATCH" if not b > 0 else "CRITICAL PATCH")
    return out


def main(path, out_dir="results"):
    if path.endswith(".csv"):                       # the held-out prediction from a finished stage-1 CSV
        import csv
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        for (lam, tie), f in held_out_fit(rows).items():
            print("lambda=%.2f %-15s joint fit a=%.3f b=%.4f c=%.3f -> k*=%.2f in [%.2f, %.2f], dH*=%.1f in [%.1f, %.1f]"
                  " | alone: %s | fit says %s"
                  % (lam, tie, f["a"], f["b"], f["c"], f["k_star"], *f["k_range"], f["dh_star"], *f["dh_range"],
                     {L: ("%.2f, %.1f" % v) for L, v in f["alone"].items()}, f["verdict"]))
        return 0
    cfg = json.loads(Path(path).read_text())
    lengths, lams, ties = [int(x) for x in cfg["lengths"]], [float(x) for x in cfg["lambdas"]], list(cfg["ties"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T54 with Amendments 1 and 2, written 2026-10-06 and 2026-10-09")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for length in lengths:
            adj0, part = torus([4, length, length])
            base = {(lam, tie): energy(adj0, lam, tie) for lam in lams for tie in ties}
            k_max = int(cfg.get("k_max", length - 1))
            series = {(lam, tie): [] for lam in lams for tie in ties}
            previous = {}
            for k in range(1, k_max + 1):
                adj, touched = patch(adj0, length, k)
                valid = bool(is_valid(adj))
                d = local_dimension_d(adj)
                census = {j: int((d == j).sum()) for j in range(7)}
                moves = cheapest_move(adj, part, touched, lams, ties) if valid else {}
                for lam in lams:
                    for tie in ties:
                        dh = energy(adj, lam, tie) - base[(lam, tie)]
                        cost, ds, dx = moves.get((lam, tie), (np.nan, 0, 0))
                        row = dict(L=length, k=k, lam=lam, tie=tie, valid=valid, dH=round(dh, 6),
                                   growth=("" if (lam, tie) not in previous else round(dh - previous[(lam, tie)], 6)),
                                   cheapest=round(cost, 6) if not np.isnan(cost) else "", cheapest_dS=ds, cheapest_dX=dx,
                                   in_fit=(k <= length // 2 - 1),
                                   **{"d%d" % j: census[j] for j in range(7)})
                        out.write(row)
                        series[(lam, tie)].append((k, dh))
                        previous[(lam, tie)] = dh
                print("L=%d k=%d valid=%s d=%s | dH: %s" % (length, k, valid, {j: census[j] for j in range(2, 6)},
                      ", ".join("%.2f/%s %.1f" % (lam, tie[:4], series[(lam, tie)][-1][1]) for lam in lams for tie in ties)),
                      flush=True)
            for (lam, tie), pts in series.items():
                ks = [k for k, _ in pts if k <= length // 2 - 1]
                dhs = [dh for k, dh in pts if k <= length // 2 - 1]
                a, b, c = fit(ks, dhs)
                k_star, dh_star = critical(a, b, c)
                print("FIT L=%d lambda=%.2f %-15s over k<=%d: a=%.3f b=%.3f c=%.3f -> k*=%.2f dH*=%.1f | verdict over k "
                      "computed: %s" % (length, lam, tie, length // 2 - 1, a, b, c, k_star, dh_star,
                                        verdict([dh for _, dh in pts])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results"))
