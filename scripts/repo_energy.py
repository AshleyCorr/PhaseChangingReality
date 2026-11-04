"""The repository's energy for `scripts/hidden_count.py --energy repo_energy:energy`.

`hidden_count.py` passes a wiring as a dict {vertex: set of neighbours}; this turns it into the (N, 2D) neighbour
array of `graphity.cqg` and returns H = 16(N - S) + 4 lambda X. Four links only (D = 2). A wiring in which some
vertex has lost or gained a link is given infinite energy, so it can never tie with the original.

lambda is 1.25 unless the environment variable HIDDEN_COUNT_LAMBDA says otherwise (the hidden count compares
energies for exact equality, so the value matters only through which wirings tie).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import hamiltonian    # noqa: E402

LAM = float(os.environ.get("HIDDEN_COUNT_LAMBDA", "1.25"))


def energy(adj_dict, D):
    n = len(adj_dict)
    arr = np.full((n, 2 * D), -1, dtype=np.int64)
    for v, s in adj_dict.items():
        nb = sorted(s)
        if len(nb) != 2 * D:
            return float("inf")
        arr[v] = nb
    return float(hamiltonian(arr, LAM))
