"""T15 rung 0: do the versions exist where the hypothesis says? Usage:

    python scripts/analyse_t15_rung0.py configs/t15_rung0.json
    python scripts/analyse_t15_rung0.py --report results/t15_rung0.csv     # re-read a finished run

Implements PREREGISTRATION.md section T15 rung 0 (written 2026-09-23, before this ran) and makes
no choices of its own. Everything is exact. The only random numbers make the melts, from seeds in
the config, and the counting is done on the melt that comes out, not sampled.

WHAT IS COUNTED. A renaming is a permutation of the points that keeps every point on its own side
of the bipartition and changes no relationship: a graph automorphism, in the convention of
ASSUMPTIONS Q15 and `small_graphs.log_automorphisms`. Two counts for every object, side by side,
because the author chose both and the choice changes the answer:

  whole graph   renamings of the entire arrangement, sheet and defect together;
  defect only   renamings of the subgraph induced on the vertices whose local dimension is not 2.

And the Laplacian spectrum (eigenvalues of D - A) of each connected piece of that defect subgraph.
A piece is named from its wiring by `analyse_t7_states.name_piece` where the graph gives a name,
and by its composition otherwise. Two spectra differ when their sorted non-zero eigenvalues differ
by more than 1e-9 in any position or in length.

THE GATE. `log_automorphisms` must return ln 320 for the 16 x 10 sheet and ln 1 for the melt of
Q15 (seed 2024), and the helper used on defect subgraphs, which the named tool cannot take because
it expects a 4-regular array, must return 320 on the same sheet. Otherwise nothing else is read.

OBJECTS. The twelve saved resting states of T7 amendment 4 (results/t7d_lam125_n*_adj/*.npz); at
each size that appears there, a perfect sheet, the tube those states started from, and a melt. The
tube is not in the pre-registration's table; it is the start of every decay and is reported as a
baseline only. It takes no part in any verdict.

One column beyond the pre-registration, `renamings_defect_any_side`, drops the same-side rule for
the defect subgraph alone. It costs nothing on a graph of four points and is there to help read
the pre-registered number, not to replace it.
"""
import csv
import glob
import json
import platform
import sys
import time
from collections import defaultdict
from math import exp, factorial
from pathlib import Path

import networkx as nx
import numba
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, is_valid, run_chain, torus                # noqa: E402
from graphity.dimension import local_dimension                            # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.small_graphs import log_automorphisms, sides_first          # noqa: E402
from scripts.analyse_t7_states import name_piece, pieces_with_members     # noqa: E402

SAME_SIDE = categorical_node_match("side", 0)
TOL = 1e-9              # "Definitions, fixed now": floating point only; the arithmetic is on integers
SHEET_D = 2             # local dimension of a point that is sheet
D_BINS = 7
EXACT_BELOW = 1e15      # a count is reported as an integer below this and as its log otherwise


# ------------------------------------------------------------------------------ the counting

def graph_of(adj, side, members=None):
    """networkx graph of the vertices in `members` (all when None), each carrying its side."""
    n = adj.shape[0]
    keep = np.ones(n, dtype=bool) if members is None else np.asarray(members, dtype=bool)
    g = nx.Graph()
    g.add_nodes_from((int(v), dict(side=int(side[v]))) for v in np.flatnonzero(keep))
    for v in np.flatnonzero(keep):
        for w in adj[v]:
            if w >= 0 and keep[w] and v < w:
                g.add_edge(int(v), int(w))
    return g


def renamings(g, same_side=True):
    """The exact number of renamings of `g` that change no relationship, as an integer.

    Piece by piece, as `log_automorphisms` does it: isomorphic pieces may be permuted among
    themselves, so k identical pieces with A renamings each contribute A^k k!. The empty graph
    has one renaming, the empty one. `same_side=False` drops the bipartition rule.
    """
    match = SAME_SIDE if same_side else None
    kinds = []                                       # [representative, how many, own count]
    for part in nx.connected_components(g):
        piece = g.subgraph(part).copy()
        for kind in kinds:
            rep = kind[0]
            if (len(rep) == len(piece) and rep.number_of_edges() == piece.number_of_edges()
                    and GraphMatcher(rep, piece, node_match=match).is_isomorphic()):
                kind[1] += 1
                break
        else:
            own = sum(1 for _ in GraphMatcher(piece, piece, node_match=match).isomorphisms_iter())
            kinds.append([piece, 1, own])
    total = 1
    for _, k, own in kinds:
        total *= own ** k * factorial(k)
    return total


def laplacian_spectrum(g):
    """Sorted eigenvalues of D - A. Built by hand so that nothing beyond numpy is needed."""
    nodes = sorted(g)
    if not nodes:
        return []
    idx = {v: i for i, v in enumerate(nodes)}
    lap = np.zeros((len(nodes), len(nodes)))
    for u, v in g.edges():
        i, j = idx[u], idx[v]
        lap[i, j] -= 1.0
        lap[j, i] -= 1.0
        lap[i, i] += 1.0
        lap[j, j] += 1.0
    return sorted(float(x) for x in np.linalg.eigvalsh(lap))


def nonzero(spectrum):
    return [x for x in spectrum if x > TOL]


def spectra_differ(a, b):
    """The pre-registered rule: sorted non-zero eigenvalues differ in length or in any position."""
    a, b = nonzero(a), nonzero(b)
    return len(a) != len(b) or any(abs(x - y) > TOL for x, y in zip(a, b))


def as_count(ln):
    """An integer where the count is small enough to be exact, else None (the log is kept)."""
    value = exp(ln)
    if value < EXACT_BELOW:
        rounded = int(round(value))
        if abs(value - rounded) > 1e-6 * max(1.0, rounded):
            raise ValueError("ln %r is not the log of an integer" % ln)
        return rounded
    return None


# ---------------------------------------------------------------------------- one object

def read(adj, part):
    """Both counts, the pieces off the sheet with their spectra, for one arrangement."""
    adj = sides_first(adj, part)                     # the numbering the named tool expects
    n = adj.shape[0]
    side = (np.arange(n) >= n // 2).astype(int)
    if not is_valid(adj, NO_CAP):
        raise ValueError("not a valid state")

    t = time.time()
    ln_whole = float(log_automorphisms(adj))
    seconds = time.time() - t

    d = local_dimension(adj)
    members = d != SHEET_D
    g_def = graph_of(adj, side, members)
    pieces = []
    for p in pieces_with_members(adj, members):
        comp = tuple(int(c) for c in np.bincount(d[p], minlength=D_BINS)[:D_BINS])
        sub = g_def.subgraph([int(v) for v in p]).copy()
        pieces.append(dict(name=name_piece(comp), size=int(len(p)),
                           edges=int(sub.number_of_edges()), spectrum=laplacian_spectrum(sub)))
    return dict(N=n, ln_whole=ln_whole, whole=as_count(ln_whole),
                defect_vertices=int(members.sum()),
                defect=renamings(g_def), defect_any_side=renamings(g_def, same_side=False),
                d_hist=[int(c) for c in np.bincount(d, minlength=D_BINS)[:D_BINS]],
                pieces=pieces, seconds=seconds)


def melt(lx, ly, seed, sweeps):
    adj, part = torus(lx, ly, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    run_chain(adj, side_u, 0.0, int(sweeps), 1, int(seed), 1.0, NO_CAP)   # infinite temperature
    return adj, part


def fmt_spec(spectrum):
    return "|".join("%.9g" % x for x in spectrum)


def parse_spec(text):
    return [float(x) for x in text.split("|")] if text else []


def to_row(obj, source, got):
    return {
        "object": obj, "source": source, "N": got["N"],
        "d_hist": "|".join(str(c) for c in got["d_hist"]),
        "defect_vertices": got["defect_vertices"],
        "pieces": ";".join("%s(%d)" % (p["name"], p["size"]) for p in got["pieces"]),
        "renamings_whole": "" if got["whole"] is None else got["whole"],
        "ln_renamings_whole": "%.12g" % got["ln_whole"],
        "renamings_defect": got["defect"],
        "ln_renamings_defect": "%.12g" % np.log(got["defect"]),
        "renamings_defect_any_side": got["defect_any_side"],
        "spectra": ";".join("%s:%s" % (p["name"], fmt_spec(p["spectrum"])) for p in got["pieces"]),
        "seconds_whole": "%.1f" % got["seconds"],
    }


def from_row(r):
    """A CSV row back into what `verdict` reads."""
    pieces = []
    if r["spectra"]:
        for item in r["spectra"].split(";"):
            name, spec = item.rsplit(":", 1)
            pieces.append((name, parse_spec(spec)))
    whole = int(r["renamings_whole"]) if r["renamings_whole"] not in ("", None) else None
    return dict(object=r["object"], source=r["source"], N=int(r["N"]),
                defect_vertices=int(r["defect_vertices"]), whole=whole,
                defect=int(r["renamings_defect"]), pieces=pieces)


# --------------------------------------------------------------------------------- verdict

def verdict(rows):
    """The three predictions and the four verdicts of PREREGISTRATION T15 rung 0."""
    lines = []
    left = [r for r in rows if r["object"] == "leftover"]
    with_defect = [r for r in left if r["defect_vertices"] > 0]
    melts = [r for r in rows if r["object"] == "melt"]

    # Prediction 1: a remnant in a sheet has more than one renaming, on both counts.
    whole_ok = [r for r in with_defect if r["whole"] is None or r["whole"] > 1]
    defect_ok = [r for r in with_defect if r["defect"] > 1]
    p1 = bool(with_defect) and len(whole_ok) == len(with_defect) and len(defect_ok) == len(with_defect)
    lines.append("Prediction 1 (a remnant in a sheet has more than one renaming, both counts): %s"
                 % ("HOLDS" if p1 else "FAILS"))
    lines.append("   whole graph > 1 in %d of %d leftovers with a defect; defect only > 1 in %d of %d"
                 % (len(whole_ok), len(with_defect), len(defect_ok), len(with_defect)))
    for r in with_defect:
        lines.append("   %-28s whole %-6s defect %-6s %s"
                     % (r["source"], r["whole"] if r["whole"] is not None else "huge",
                        r["defect"], "" if (r["whole"] != 1 and r["defect"] > 1) else "<-- exactly 1"))
    if len(left) != len(with_defect):
        lines.append("   %d leftover state(s) have no defect at all and take no part: %s"
                     % (len(left) - len(with_defect),
                        ", ".join(r["source"] for r in left if r["defect_vertices"] == 0)))

    # Prediction 2: a melt has exactly one, at every size.
    p2 = bool(melts) and all(r["whole"] == 1 for r in melts)
    lines.append("Prediction 2 (a melt has exactly one renaming, every size): %s"
                 % ("HOLDS" if p2 else "FAILS"))
    for r in melts:
        lines.append("   N = %-4d whole %s" % (r["N"], r["whole"]))

    # Prediction 3: distinct leftover types have distinct Laplacian spectra.
    by_type = defaultdict(list)                      # type name -> the distinct spectra seen
    for r in with_defect:
        for name, spec in r["pieces"]:
            if not any(not spectra_differ(spec, s) for s in by_type[name]):
                by_type[name].append(spec)
    names = sorted(by_type)
    clashes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in by_type[names[i]]:
                for b in by_type[names[j]]:
                    if not spectra_differ(a, b):
                        clashes.append((names[i], names[j], a))
    p3 = None if len(names) < 2 else not clashes
    lines.append("Prediction 3 (distinct leftover types have distinct spectra): %s"
                 % ({True: "HOLDS", False: "FAILS", None: "NOT EVALUABLE, one type only"}[p3]))
    for name in names:
        for spec in by_type[name]:
            lines.append("   %-26s non-zero eigenvalues %s" % (name, fmt_spec(nonzero(spec)) or "(none)"))
        if len(by_type[name]) > 1:
            lines.append("   ^ one name, %d different spectra: the composition lumps different wirings"
                         % len(by_type[name]))
    for a, b, spec in clashes:
        lines.append("   CLASH: '%s' and '%s' share the spectrum %s" % (a, b, fmt_spec(nonzero(spec)) or "(none)"))

    lines.append("")
    if p1 and p2 and p3:
        out = "VERSIONS EXIST AND LABEL TYPE"
    elif p1 and p2:
        out = "VERSIONS EXIST"
    elif with_defect and all(r["whole"] == 1 for r in with_defect):
        out = "NO VERSIONS IN ANYTHING REAL"
    else:
        out = "INCONCLUSIVE"
    lines.append("PRE-REGISTERED VERDICT: %s" % out)
    return lines, out


# ------------------------------------------------------------------------------------ main

def gate(cfg):
    """Q15's two numbers, from the named tool and from the helper. Raises if either is off."""
    lx, ly = cfg["sheet"]
    adj, part = torus(lx, ly, NO_CAP)
    sheet = sides_first(adj, part)
    n = sheet.shape[0]
    side = (np.arange(n) >= n // 2).astype(int)
    got_tool = as_count(float(log_automorphisms(sheet)))
    got_helper = renamings(graph_of(sheet, side))
    adj, part = melt(lx, ly, cfg["melt_seed"], cfg["melt_sweeps"])
    got_melt = as_count(float(log_automorphisms(sides_first(adj, part))))
    result = dict(sheet_tool=got_tool, sheet_helper=got_helper, melt_tool=got_melt)
    want = (int(cfg["sheet_renamings"]), int(cfg["sheet_renamings"]), int(cfg["melt_renamings"]))
    if (got_tool, got_helper, got_melt) != want:
        raise RuntimeError("gate failed: %r, wanted %r. The tool is wrong; nothing else is read."
                           % (result, want))
    return result


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    print("Gate (ASSUMPTIONS Q15) ...", flush=True)
    t = time.time()
    gate_result = gate(cfg["gate"])
    print("   passed: %r in %.0f s" % (gate_result, time.time() - t), flush=True)

    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                networkx=nx.__version__, gate=gate_result,
                preregistration="PREREGISTRATION.md section T15 rung 0, written 2026-09-23")
    rows = []
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        def emit(obj, source, got):
            row = to_row(obj, source, got)
            out.write(row)
            rows.append(from_row({k: str(v) for k, v in row.items()}))
            print("   %-9s %-28s N=%-4d whole %-8s defect %-6s pieces: %s   (%.0f s)"
                  % (obj, source, got["N"], got["whole"] if got["whole"] is not None else "huge",
                     got["defect"], row["pieces"] or "none", got["seconds"]), flush=True)

        files = sorted(glob.glob(str(ROOT / cfg["leftovers"])))
        print("Leftovers (%d files) ..." % len(files), flush=True)
        sizes = set()
        for f in files:
            z = np.load(f)
            got = read(z["adj"], z["part"])
            sizes.add(got["N"])
            emit("leftover", Path(f).parent.name.replace("t7d_lam125_", "") + "/" + Path(f).stem, got)

        print("Sheets and tubes ...", flush=True)
        for n in sorted(sizes):
            lx, ly = cfg["sheets"][str(n)]
            adj, part = torus(lx, ly, NO_CAP)
            emit("sheet", "torus(%d,%d)" % (lx, ly), read(adj, part))
            lx, ly = cfg["tubes"][str(n)]
            adj, part = torus(lx, ly, NO_CAP)
            emit("tube", "torus(%d,%d)" % (lx, ly), read(adj, part))

        print("Melts ...", flush=True)
        for n in sorted(sizes):
            lx, ly = cfg["sheets"][str(n)]
            seed = int(np.random.SeedSequence([int(cfg["melt_seed"]), n]).generate_state(1)[0])
            adj, part = melt(lx, ly, seed, cfg["melt_sweeps"])
            emit("melt", "melt(%d,%d) seed %d" % (lx, ly, seed), read(adj, part))

    print()
    lines, out = verdict(rows)
    print("\n".join(lines))
    return out


def report(csv_path):
    with open(csv_path, newline="") as f:
        rows = [from_row(r) for r in csv.DictReader(f)]
    lines, out = verdict(rows)
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        main(sys.argv[1])
