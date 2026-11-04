"""T7 amendment 4 (a): read each resting state from its saved wiring, and check its energy. Usage:

    python scripts/analyse_t7_states.py

Implements the enacted wording of PREREGISTRATION.md T7 amendment 4 and makes no choices of its
own. For every decay replayed under it (results/t7d_lam125_n*.csv, with the final adjacency in
results/t7d_lam125_n*_adj/), it

  1. checks the replay is the same decay as its original in t7b (same waiting time and the same
     first-window release), as amendment 3 required;
  2. reads the resting state from the graph: the histogram of local dimension d(v), the connected
     pieces of the vertices that are not sheet (d != 2) with what each is made of, the extra
     squares S - N and the surplus edges X;
  3. computes the state's exact energy from that wiring, H = 16(N - S) + 4*lambda*X, and passes
     gate 3 exactly when the recorded release equals the release this implies, to within 1 % of
     the lump 4(lambda - 1).

Pieces are named only where the graph says what they are: four vertices at d = 1 is the
four-point remnant of ASSUMPTIONS O16; two at d = 1 and two at d = 3 is the twist of O15.
Anything else is reported by its composition and left unnamed.
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, surplus, total_squares  # noqa: E402
from graphity.dimension import local_dimension                                        # noqa: E402

TOLERANCE = 0.01        # of the lump 4(lambda - 1), as in amendment 2
D_BINS = 7


def pieces_with_members(adj, members):
    """Connected pieces of the subgraph on `members`, as arrays of vertex indices, largest first."""
    members = np.asarray(members, dtype=bool)
    seen = np.zeros(adj.shape[0], dtype=bool)
    out = []
    for start in np.flatnonzero(members):
        if seen[start]:
            continue
        stack, got = [start], []
        seen[start] = True
        while stack:
            v = stack.pop()
            got.append(v)
            for w in adj[v]:
                if w >= 0 and members[w] and not seen[w]:
                    seen[w] = True
                    stack.append(w)
        out.append(np.array(got))
    return sorted(out, key=len, reverse=True)


def describe(adj, lam):
    """The resting state, read from its wiring."""
    n = adj.shape[0]
    s, x = int(total_squares(adj)), int(surplus(adj))
    d = local_dimension(adj)
    comp = [tuple(int(c) for c in np.bincount(d[p], minlength=D_BINS)[:D_BINS])
            for p in pieces_with_members(adj, d != 2)]
    return dict(n=n, extra_squares=s - n, surplus=x,
                h=ENERGY_PER_SQUARE * (n - s) + ENERGY_PER_SURPLUS * lam * x,
                hist=tuple(int(c) for c in np.bincount(d, minlength=D_BINS)[:D_BINS]),
                pieces=comp)


def name_piece(counts):
    """A name where the graph gives one, otherwise the composition."""
    counts = tuple(counts)
    if sum(counts) == 4 and counts[1] == 4:
        return "four-point remnant"
    if sum(counts) == 4 and counts[1] == 2 and counts[3] == 2:
        return "twist"
    return "unnamed: " + ", ".join("%d at d=%d" % (c, k) for k, c in enumerate(counts) if c)


def gate(recorded_release, h0_total, h_total, n, lam):
    """Released energy per point implied by the structure, and whether it matches the record."""
    implied = (h0_total - h_total) / n
    return implied, abs(implied - recorded_release) <= TOLERANCE * 4.0 * (lam - 1.0)


def originals(n, out_dir):
    got = {}
    for f in glob.glob(str(Path(out_dir) / ("t7b_lam125_n%d*.csv" % n))):
        for r in csv.DictReader(open(f, newline="")):
            got[int(r["replica"])] = r
    return got


def check_all(out_dir="results", prefix="t7d", tag="lam125"):
    """Every decay replayed under amendment 4: {(N, replica): record}, with record['ok'] the gate."""
    got = {}
    for f in sorted(glob.glob(str(Path(out_dir) / ("%s_%s_n*.csv" % (prefix, tag))))):
        name = Path(f).stem
        for r in csv.DictReader(open(f, newline="")):
            n, rep, lam = int(r["N"]), int(r["replica"]), float(r["lam"])
            z = np.load(Path(out_dir) / (name + "_adj") / ("N%d_rep%d.npz" % (n, rep)))
            state = describe(z["adj"], lam)
            orig = originals(n, out_dir).get(rep)
            same = (orig is not None and orig["waiting"] == r["waiting"]
                    and abs(float(orig["released_first_window"]) - float(r["released_first_window"])) < 1e-9)
            implied, match = gate(float(r["released"]), float(z["h0"]), state["h"], n, lam)
            got[(n, rep)] = dict(row=r, state=state, same=same, implied=implied, ok=bool(match and same))
    return got


def main(out_dir="results"):
    print("T7 amendment 4 (a): resting states read from their wiring.\n")
    recs = check_all(out_dir)
    for (n, rep), c in sorted(recs.items()):
        st, r = c["state"], c["row"]
        print("N=%-4d rep=%-3d same decay: %-3s  held %5.1f units (S-N=%d, X=%d)  release recorded %.4f,"
              " implied %.4f  gate 3: %s"
              % (n, rep, "yes" if c["same"] else "NO", st["h"], st["extra_squares"], st["surplus"],
                 float(r["released"]), c["implied"], "pass" if c["ok"] else "FAIL"))
        print("        d-histogram %s; pieces off the sheet: %s"
              % (list(st["hist"]),
                 "; ".join("%d points (%s)" % (sum(p), name_piece(p)) for p in st["pieces"]) or "none"))
    print("\n%d of %d replayed decays pass gate 3 under amendment 4 (a)."
          % (sum(c["ok"] for c in recs.values()), len(recs)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
