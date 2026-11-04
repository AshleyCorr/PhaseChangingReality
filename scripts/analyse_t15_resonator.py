"""T15 rung 0b: is the closed loop of four a resonator inside the sheet? Usage:

    python scripts/analyse_t15_resonator.py configs/t15_resonator.json
    python scripts/analyse_t15_resonator.py --report results/t15_resonator.csv

Implements PREREGISTRATION.md section T15 rung 0b (written 2026-09-23, before this ran) and makes no
choices of its own. Everything is exact; the only random numbers choose the control point sets.

WHAT IS COMPUTED. For each saved resting state that holds a closed loop of four: the Laplacian
L = D - A of the whole arrangement; its eigenvalues grouped into eigenspaces; and for each eigenspace
its localisation on a point set S, the largest eigenvalue of the eigenspace projector restricted to
S, which is the most weight any unit vector of the eigenspace can carry on S (1 when a mode lives on
S alone; |S|/N for a mode spread evenly). S is the union of the state's loop-of-four points; beside
it, each loop alone, the loops with their collars, and the 3-cube's points where a cube is present.

GATES. The isolated loop gives localisation 1 at eigenvalues 2 and 4; a perfect sheet with an
eight-point set gives less than 0.5 everywhere; in every state twenty random eight-point sets give
less than 0.5 everywhere. Otherwise the threshold does not discriminate and nothing else is read.
"""
import csv
import glob
import json
import platform
import sys
import time
from pathlib import Path

import networkx as nx
import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, torus                                    # noqa: E402
from graphity.dimension import local_dimension                            # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from scripts.analyse_t7_states import pieces_with_members                 # noqa: E402

SHEET_D = 2


# ------------------------------------------------------------------------------ the operator

def dense(adj):
    """(N, N) adjacency matrix from the kernel's (N, 4) neighbour array."""
    n = adj.shape[0]
    a = np.zeros((n, n))
    for v in range(n):
        for w in adj[v]:
            if w >= 0:
                a[v, w] = 1.0
    return a


def laplacian(a):
    return np.diag(a.sum(axis=1)) - a


def eigenspaces(lap, tol=1e-9):
    """[(eigenvalue, (N, m) orthonormal basis)] with eigenvalues grouped within tol, ascending."""
    lam, vec = np.linalg.eigh(lap)
    out, start = [], 0
    for i in range(1, len(lam) + 1):
        if i == len(lam) or lam[i] - lam[start] > tol:
            out.append((float(np.mean(lam[start:i])), vec[:, start:i]))
            start = i
    return out


def localisation(basis, points):
    """Most weight any unit vector in the eigenspace can carry on `points`: the top eigenvalue of
    the projector restricted there. Basis-independent."""
    sub = basis[np.asarray(points), :]
    m = sub @ sub.T
    return float(np.linalg.eigvalsh(m)[-1]) if len(points) else 0.0


def best_mode(basis, points):
    """The unit vector of the eigenspace that carries the most weight on `points`."""
    sub = basis[np.asarray(points), :]
    w, u = np.linalg.eigh(sub @ sub.T)
    vec = basis @ (sub.T @ u[:, -1])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def participation_ratio(vec):
    return float(1.0 / np.sum(vec ** 4))


# ------------------------------------------------------------------------------- the objects

def loops_and_cubes(adj):
    """Closed loops of four and 3-cubes among the pieces off the sheet, read by isomorphism."""
    d = local_dimension(adj)
    g = nx.Graph()
    for v in range(adj.shape[0]):
        for w in adj[v]:
            if w >= 0 and v < w:
                g.add_edge(int(v), int(w))
    loops, cubes = [], []
    for p in pieces_with_members(adj, d != SHEET_D):
        sub = g.subgraph([int(v) for v in p])
        if len(p) == 4 and nx.is_isomorphic(sub, nx.cycle_graph(4)):
            loops.append([int(v) for v in p])
        elif len(p) == 8 and nx.is_isomorphic(sub, nx.hypercube_graph(3)):
            cubes.append([int(v) for v in p])
    return loops, cubes


def collar_of(adj, loop_points, all_loop_points):
    keep = set(all_loop_points)
    return sorted({int(w) for v in loop_points for w in adj[v] if w >= 0 and int(w) not in keep})


# --------------------------------------------------------------------------------- reading

def read_state(a, loops, cubes, cfg, rng=None, adj=None):
    """Rows for every eigenspace, plus the state's summary. `a` is the dense adjacency."""
    n = a.shape[0]
    spaces = eigenspaces(laplacian(a), cfg["group_tolerance"])
    union = sorted(v for loop in loops for v in loop)
    collar = sorted(set(union) | {v for loop in loops for v in collar_of(adj, loop, union)}) if adj is not None else union
    cube_pts = sorted(v for cube in cubes for v in cube)
    random_max = 0.0
    random_sets = []
    if rng is not None and union:
        for _ in range(int(cfg["controls"]["random_sets"])):
            random_sets.append(sorted(int(v) for v in rng.choice(n, size=len(union), replace=False)))
    rows = []
    for lam, basis in spaces:
        if lam <= cfg["group_tolerance"]:
            continue                                             # the constant mode, the sheet's own
        loc_union = localisation(basis, union) if union else 0.0
        loc_each = [localisation(basis, loop) for loop in loops]
        loc_collar = localisation(basis, collar) if union else 0.0
        loc_cube = localisation(basis, cube_pts) if cube_pts else None
        r_max = max((localisation(basis, s) for s in random_sets), default=0.0)
        random_max = max(random_max, r_max)
        pr = participation_ratio(best_mode(basis, union)) if union else None
        rows.append(dict(eigenvalue=lam, multiplicity=basis.shape[1], loc_loops=loc_union,
                         loc_each_loop=loc_each, loc_loops_collar=loc_collar, loc_cube=loc_cube,
                         random_max_here=r_max, pr_best=pr))
    return rows, dict(N=n, n_loops=len(loops), n_cubes=len(cubes), random_max=random_max,
                      spread_even=(len(union) / n if union else None))


def state_class(rows, cfg):
    """ISOLATED, RETUNED or DISSOLVED for one state, by the pre-registered definitions."""
    thr, win = cfg["localised_at_least"], cfg["window"]
    localised = [r["eigenvalue"] for r in rows if r["loc_loops"] >= thr]
    if not localised:
        return "DISSOLVED", localised
    near = [any(abs(lam - iso) <= win for lam in localised) for iso in cfg["isolated_eigenvalues"]]
    return ("ISOLATED" if all(near) else "RETUNED"), localised


def verdict(summaries, gates, cfg):
    """summaries: [(source, rows, meta)] for the loop-of-four states. gates: dict of booleans."""
    lines = ["Gates: isolated loop %s; perfect sheet %s; random sets %s"
             % tuple("pass" if gates[k] else "FAIL" for k in ("loop", "sheet", "random"))]
    if not all(gates.values()):
        lines.append("")
        lines.append("PRE-REGISTERED VERDICT: INCONCLUSIVE (a gate failed; the threshold does not discriminate)")
        return lines, "INCONCLUSIVE"
    classes = []
    for source, rows, meta in summaries:
        cls, localised = state_class(rows, cfg)
        classes.append(cls)
        top = sorted(rows, key=lambda r: -r["loc_loops"])[:4]
        lines.append("%-24s N=%-4d loops %d cubes %d  -> %-9s localised modes at %s; random sets max %.3f; even spread %.3f"
                     % (source, meta["N"], meta["n_loops"], meta["n_cubes"], cls,
                        ", ".join("%.3f" % x for x in localised) or "none", meta["random_max"],
                        meta["spread_even"] or 0.0))
        for r in top:
            lines.append("      eigenvalue %.4f (x%d)  on loops %.3f  each %s  with collar %.3f  best-mode participation %.1f"
                         % (r["eigenvalue"], r["multiplicity"], r["loc_loops"],
                            "/".join("%.2f" % x for x in r["loc_each_loop"]), r["loc_loops_collar"],
                            r["pr_best"] or 0.0))
    lines.append("")
    if not classes:
        out = "INCONCLUSIVE"
    elif all(c == "ISOLATED" for c in classes):
        out = "RESONATOR AT THE ISOLATED FREQUENCIES"
    elif all(c == "DISSOLVED" for c in classes):
        out = "NOT A RESONATOR"
    elif all(c != "DISSOLVED" for c in classes):
        out = "RESONATOR, RETUNED"
    else:
        out = "INCONCLUSIVE"
    lines.append("States: %s" % ", ".join(classes))
    lines.append("PRE-REGISTERED VERDICT: %s" % out)
    return lines, out


# ------------------------------------------------------------------------------------ main

def cycle_adjacency(n):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1.0
    return a


def run_gates(cfg):
    thr = cfg["localised_at_least"]
    # the isolated loop: localisation 1 at 2 and at 4
    spaces = eigenspaces(laplacian(cycle_adjacency(4)), cfg["group_tolerance"])
    got = {round(lam, 6): localisation(basis, [0, 1, 2, 3]) for lam, basis in spaces if lam > 1e-9}
    loop_ok = all(abs(got.get(round(iso, 6), 0.0) - 1.0) < 1e-9 for iso in cfg["isolated_eigenvalues"])
    # the perfect sheet with two disjoint squares
    lx, ly = cfg["controls"]["sheet"]
    adj, _ = torus(lx, ly, NO_CAP)
    pts = [x * ly + y for x, y in ((0, 0), (0, 1), (1, 0), (1, 1), (6, 6), (6, 7), (7, 6), (7, 7))]
    sheet_max = max(localisation(basis, pts) for lam, basis in
                    eigenspaces(laplacian(dense(adj)), cfg["group_tolerance"]) if lam > 1e-9)
    return dict(loop=loop_ok, sheet=sheet_max < thr), dict(loop=got, sheet_max=sheet_max)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    gates, gate_numbers = run_gates(cfg)
    print("Gates: isolated loop %r -> %s; sheet max localisation %.3f -> %s"
          % (gate_numbers["loop"], "pass" if gates["loop"] else "FAIL",
             gate_numbers["sheet_max"], "pass" if gates["sheet"] else "FAIL"), flush=True)
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                networkx=nx.__version__, gates=gate_numbers,
                preregistration="PREREGISTRATION.md section T15 rung 0b, written 2026-09-23")
    rng = np.random.default_rng(int(cfg["controls"]["random_seed"]))
    summaries, cube_only = [], []
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for f in sorted(glob.glob(str(ROOT / cfg["states"]))):
            z = np.load(f)
            adj = z["adj"]
            loops, cubes = loops_and_cubes(adj)
            source = Path(f).parent.name.replace("t7d_lam125_", "") + "/" + Path(f).stem
            if not loops and not cubes:
                continue
            t = time.time()
            rows, m = read_state(dense(adj), loops, cubes, cfg, rng, adj)
            for r in rows:
                out.write({"source": source, "N": m["N"], "n_loops": m["n_loops"], "n_cubes": m["n_cubes"],
                           "eigenvalue": "%.10g" % r["eigenvalue"], "multiplicity": r["multiplicity"],
                           "loc_loops": "%.6f" % r["loc_loops"],
                           "loc_each_loop": "|".join("%.6f" % x for x in r["loc_each_loop"]),
                           "loc_loops_collar": "%.6f" % r["loc_loops_collar"],
                           "loc_cube": "" if r["loc_cube"] is None else "%.6f" % r["loc_cube"],
                           "random_max_here": "%.6f" % r["random_max_here"],
                           "pr_best": "" if r["pr_best"] is None else "%.3f" % r["pr_best"]})
            if loops:
                summaries.append((source, rows, m))
            else:
                cube_only.append((source, rows, m))
            print("   %-24s N=%-4d loops %d cubes %d  %d eigenspaces  (%.1f s)"
                  % (source, m["N"], m["n_loops"], m["n_cubes"], len(rows), time.time() - t), flush=True)
    gates["random"] = all(m["random_max"] < cfg["localised_at_least"] for _, _, m in summaries)
    print()
    lines, out = verdict(summaries, gates, cfg)
    print("\n".join(lines))
    if cube_only:
        print("\nBeside, not judged: the 3-cube's modes")
        for source, rows, m in cube_only:
            top = sorted((r for r in rows if r["loc_cube"] is not None), key=lambda r: -r["loc_cube"])[:4]
            print("   %-24s " % source + "; ".join("%.3f at %.3f" % (r["loc_cube"], r["eigenvalue"]) for r in top))
    return out


def report(csv_path, cfg_path="configs/t15_resonator.json"):
    cfg = json.loads(Path(cfg_path).read_text())
    by = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            row = dict(eigenvalue=float(r["eigenvalue"]), multiplicity=int(r["multiplicity"]),
                       loc_loops=float(r["loc_loops"]),
                       loc_each_loop=[float(x) for x in r["loc_each_loop"].split("|") if x],
                       loc_loops_collar=float(r["loc_loops_collar"]),
                       loc_cube=float(r["loc_cube"]) if r["loc_cube"] else None,
                       random_max_here=float(r["random_max_here"]),
                       pr_best=float(r["pr_best"]) if r["pr_best"] else None)
            by.setdefault(r["source"], ([], dict(N=int(r["N"]), n_loops=int(r["n_loops"]),
                                                  n_cubes=int(r["n_cubes"]), random_max=0.0, spread_even=None)))
            by[r["source"]][0].append(row)
            by[r["source"]][1]["random_max"] = max(by[r["source"]][1]["random_max"], row["random_max_here"])
    summaries = [(s, rows, m) for s, (rows, m) in by.items() if m["n_loops"] > 0]
    for s, rows, m in summaries:
        m["spread_even"] = 4 * m["n_loops"] / m["N"]
    gates, _ = run_gates(cfg)
    gates["random"] = all(m["random_max"] < cfg["localised_at_least"] for _, _, m in summaries)
    lines, out = verdict(summaries, gates, cfg)
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        main(sys.argv[1])
