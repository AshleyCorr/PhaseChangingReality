"""T102: the toy's Klein relative as the owner's Klein store, exact. Usage:

    python scripts/t102_klein_store.py [out_dir]

PREREGISTRATION.md section T102 (written 2026-11-04 before this ran). For the Klein relative of the 4 x L tube
(read_flat_kind.klein_bottle; O150) at L = 8, 12, 16: every valid state one switch away (small_graphs.one_switch_away),
the change of energy to each at lambda = 1.10, 1.25 and 1.40, whether the state is a dip (every single move out costs),
its wall (the cheapest move out), its energy above the flat sheet (what it would release if it opened all the way), and
the same for the 4 x L tube as the control with a known answer (a dip with wall 12 at lambda = 1.25; paper 1).
"""
import csv
import json
import platform
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                   # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, is_valid, torus        # noqa: E402
from graphity.small_graphs import one_switch_away, sides_first        # noqa: E402
from read_flat_kind import klein_bottle                              # noqa: E402

LAMS = (1.10, 1.25, 1.40)


def part_of(adj):
    g = nx.Graph([(u, int(v)) for u in range(len(adj)) for v in adj[u] if v >= 0])
    col = nx.bipartite.color(g)
    return np.array([col[i] for i in range(len(adj))])


def read_state(name, adj, part, L):
    adj_sf = sides_first(adj, part)
    assert is_valid(adj_sf, NO_CAP)
    neighbours = one_switch_away(adj_sf, NO_CAP)
    rows = []
    for lam in LAMS:
        h0 = hamiltonian(adj_sf, lam)
        dh = np.array([hamiltonian(nb, lam) - h0 for nb in neighbours])
        rows.append(dict(state=name, L=L, N=len(adj), lam=lam, h=float(h0), above_flat=float(h0),
                         neighbours=len(neighbours), dip=bool((dh > 0).all()), wall=float(dh.min()),
                         downhill=int((dh < 0).sum()), level=int((dh == 0).sum())))
    return rows


def main(out_dir="results"):
    path = Path(out_dir) / "t102_klein_store.csv"
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    rows = []
    for L in (8, 12, 16):
        k = klein_bottle(4, L)
        k_adj = k[0] if isinstance(k, tuple) else k
        rows += read_state("klein", k_adj, part_of(k_adj), L)
        t_adj, t_part = torus(4, L, NO_CAP)
        rows += read_state("tube", t_adj, t_part, L)
        print(L, "done", flush=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta = dict(script="scripts/t102_klein_store.py", lams=LAMS, sizes=(8, 12, 16), package=__version__,
                python=platform.python_version(), numpy=np.__version__, networkx=nx.__version__,
                preregistration="PREREGISTRATION.md section T102")
    (Path(out_dir) / "t102_klein_store.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
