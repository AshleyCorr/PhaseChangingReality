"""T41: does the new space need room for the burp in three directions? Usage:

    python scripts/analyse_t41.py

PREREGISTRATION T41 (written 2026-09-25, before any run). Reads `results/t41_*.csv` (scripts/run_sealed_curled_d.py) and
uses T39's per-replica rules (`analyse_t39.read_replica`) with D = 3. Starting from one curled direction, T39's MIDDLE
means the run is still on that rung: read here as STAYS. Per cell (lambda, C): the majority of FLAT, STAYS, MELTED,
OTHER (else MIXED). **C\\*** per lambda: the smallest C with a FLAT majority. **Verdict per lambda:** ROOM NEEDED if C\\*
exists and some smaller C has no FLAT majority; ALWAYS OPENS if every C has a FLAT majority; NEVER OPENS if none has.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t39 import read_replica, replicas   # noqa: E402

RENAME = {"MIDDLE": "STAYS", "STUCK": "OTHER"}


def summarize(rows):
    cells = defaultdict(list)
    for (dim, lam, n, c, rep), blocks in replicas(rows).items():
        out, _, _ = read_replica(blocks, dim, lam, n)
        cells[(lam, n, c)].append(RENAME.get(out, out))
    majority = {}
    for k, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        majority[k] = top if votes > len(outs) / 2 else "MIXED"
    return {k: dict(Counter(v)) for k, v in cells.items()}, majority


def verdicts(majority):
    by_lam = defaultdict(dict)
    for (lam, n, c), m in majority.items():
        by_lam[lam][c] = m
    out = {}
    for lam, row in by_lam.items():
        flat = sorted(c for c, m in row.items() if m == "FLAT")
        if not flat:
            out[lam] = (None, "NEVER OPENS")
        elif len(flat) == len(row):
            out[lam] = (flat[0], "ALWAYS OPENS")
        else:
            c_star = flat[0]
            smaller_fail = any(c < c_star for c in row)
            out[lam] = (c_star, "ROOM NEEDED" if smaller_fail else "ALWAYS OPENS")
    return out


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t41_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T41 results")
        return
    cells, majority = summarize(rows)
    for k in sorted(cells):
        print("lambda=%.2f N=%-4d C=%-5d %-40s -> %s" % (k[0], k[1], k[2], cells[k], majority[k]))
    for lam, (c_star, word) in sorted(verdicts(majority).items()):
        print("VERDICT T41 lambda=%.2f: %s (C* = %s)" % (lam, word, c_star))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
