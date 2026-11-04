"""Find the four-point relics in a saved T37 end state and run the hidden count on a window around one. Usage:

    python scripts/relic_region.py results/t37_lam125_g125_L1024_a_adj/N4096_rep0.npz

Written in a chat session on 5 October 2026 (docs/HANDOFF_2026-10-05_chat.md, section 2.3) and kept as it was run
there, with the sandbox paths replaced. A window is a set of whole original columns of the 4 x L tube: `torus(L, 4)`
numbers its vertices x * 4 + y, so vertex v sits in column v // 4. The wiring has changed since the tube was built
but the names have not, so a window is simply a set of names; whether the relic lies inside it is printed.

EXPLORATORY. The count uses `hidden_count.py` as it stood in chat, which does not restrict the interior wirings to
the model's own (two-sided graphs obeying the hard-core rule); `graphity.hidden` (src/graphity/hidden.py) is the
version that does, and is the one to quote (ASSUMPTIONS O87).
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.cqg import hamiltonian                          # noqa: E402
from graphity.dimension import local_dimension, piece_labels  # noqa: E402
import hidden_count as hc                                     # noqa: E402
import repo_energy                                            # noqa: E402


def col(v):
    return v // 4


def main(path, widths=(2,)):
    adj = np.load(path)["adj"]
    n = adj.shape[0]
    d = local_dimension(adj)
    lab = piece_labels(adj, d != 2)
    cols = [np.flatnonzero(lab == k) for k in range(lab.max() + 1)]
    relics = [c for c in cols if len(c) == 4 and (d[c] == 1).all()]
    print("4-point relics:", len(relics), "other pieces:", len(cols) - len(relics), "H =", hamiltonian(adj, repo_energy.LAM))
    r = relics[0]
    print("relic vertices", r, "columns", sorted(set(col(v) for v in r)))
    adjd = {v: set(int(x) for x in adj[v]) for v in range(n)}
    for width in widths:
        rc = sorted(set(col(v) for v in r))
        c0 = rc[0] - (width - len(rc)) // 2 if width >= len(rc) else rc[0]
        region = [v for v in range(n) if col(v) in {c0 + i for i in range(width)}]
        inside = all(v in region for v in r)
        t = time.time()
        count, _, _ = hc.hidden_count(adjd, region, 2, repo_energy.energy)
        cut = sum(1 for v in region for w in adjd[v] if w not in region)
        print("width %d: points %d relic inside %s cut %d hidden count %d ln %.3f (%.0fs)"
              % (width, len(region), inside, cut, count, math.log(count) if count else float("-inf"), time.time() - t))


if __name__ == "__main__":
    main(sys.argv[1], tuple(int(w) for w in sys.argv[2:]) or (2,))
