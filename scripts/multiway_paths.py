"""T95: do the paths of the toy's move count the same thing as its renamings? Exact, on the complete class spaces.

Usage:
    python scripts/multiway_paths.py

THE TWO COUNTS. Update 17's versions are an arrangement's renamings, the side-preserving symmetries A of its class.
Gorard's multiway system ([Gorard20]) counts paths: every order of applying the rule, followed at once, with probabilities
from path weights. On the complete class spaces at N = 16 and 18 (T4; graphity.small_graphs) both are exact.

WHAT IS COMPUTED. From the circulant start the switch move is explored to every class. M[k][j] is the number of labelled
switches from a state of class k that land in class j (the same for every state of the class, by symmetry). The number of
k-step paths from the start state ending in class j is the start row of M^k; normalised over classes it is the path share
at k. The long-path share is the mean of the shares at K-1 and K (the move graph may be two-periodic). Beside it, per
class: A_j, L_j = (n!)^2 / A_j (labelled states in the class) and deg_j = sum_i M[j][i] (moves from a state).

Pre-registered as PREREGISTRATION.md section T95.
"""
import csv
import json
import platform
import sys
from math import factorial
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, is_valid                                   # noqa: E402
from graphity.small_graphs import circulant, explore, one_switch_away       # noqa: E402

K = 40
TOLERANCE = 0.10


def move_counts(classes, cap=NO_CAP):
    """M[k][j]: labelled switches from a state of class k into class j."""
    m = np.zeros((len(classes.reps), len(classes.reps)), dtype=float)
    for k, (adj, _) in enumerate(classes.reps):
        for other in one_switch_away(adj, cap):
            m[k, classes.index_of(other)] += 1
    return m


def path_shares(m, start, steps):
    """Row `start` of M^k normalised, for k = 1..steps (floats; the counts themselves overflow nothing at these sizes)."""
    row = np.zeros(len(m))
    row[start] = 1.0
    shares = []
    for _ in range(steps):
        row = row @ m
        row = row / row.sum()
        shares.append(row.copy())
    return shares


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def within(share, target, tol=TOLERANCE):
    return bool(np.all(np.abs(share / target - 1) <= tol))


def run(n, cap=NO_CAP, steps=K):
    start = circulant(n)
    assert is_valid(start, cap)
    classes, _ = explore(start, cap)
    m = move_counts(classes, cap)
    k0 = classes.index_of(start)
    shares = path_shares(m, k0, steps)
    long_share = (shares[-1] + shares[-2]) / 2
    a = np.array([classes.symmetries(j) for j in range(len(classes.reps))], dtype=float)
    labelled = factorial(n) ** 2 / a
    deg = m.sum(axis=1)
    versions_target = a / a.sum()
    labelled_target = labelled * deg / (labelled * deg).sum()
    word = "VERSIONS" if within(long_share, versions_target) else ("LABELLED" if within(long_share, labelled_target) else "NEITHER")
    rows = []
    for j in range(len(classes.reps)):
        rows.append(dict(n=n, N=2 * n, class_id=j, is_start=int(j == k0), renamings=int(a[j]), labelled_states=int(round(labelled[j])),
                         moves=int(deg[j]), share_k1=float(shares[0][j]), share_k5=float(shares[4][j]), share_k20=float(shares[19][j]),
                         long_path_share=float(long_share[j]), versions_target=float(versions_target[j]),
                         labelled_target=float(labelled_target[j]), word=word,
                         spearman_share_vs_renamings=spearman(long_share, a)))
    return rows, word


def main(out_dir="results"):
    name = "t95_multiway_paths"
    path = Path(out_dir) / (name + ".csv")
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    rows = []
    for n in (8, 9):
        r, word = run(n)
        rows += r
        print("n =", n, "classes", len(r), "word", word, "spearman", r[0]["spearman_share_vs_renamings"], flush=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta = dict(script="scripts/multiway_paths.py", sizes=[8, 9], cap=None, steps=K, tolerance=TOLERANCE, package=__version__,
                python=platform.python_version(), numpy=np.__version__, networkx=nx.__version__,
                preregistration="PREREGISTRATION.md section T95")
    (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
