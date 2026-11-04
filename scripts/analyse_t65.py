"""T65: does the owner's slab open into one flat space when given a push? Usage:

    python scripts/analyse_t65.py [results_dir]

PREREGISTRATION T65 (drafted 2026-10-15, revised 16 October before any run). Reads `results/t65_*.csv`
(scripts/run_sealed_curled_d.py with the table tie, a shared bath, and the "patch" start for arm B).

Per replica, from the final block: open = points at d = 3; curled = points at d <= 2; damaged = points above d = 3
(`damaged_d`). OPENS if open >= 0.9 N and damaged <= 0.02 N; FALLS BACK if curled >= 0.9 N; DAMAGED if damaged > 0.1 N;
else PARTIAL. Per cell (tie, lambda, k, C, L, E): the majority of at least six of eight, else MIXED. Per lambda over the
tied arm-A cells: OPENS IN ROOM if some cell reads OPENS; FALLS BACK if every cell reads FALLS BACK; DAMAGED if none
opens and some cell reads DAMAGED; else PARTIAL. Arm B and the untied control are reported by cell. Reported, not
scored: the release per point (the tied energy per point at the start less at the end), the share of the full release
3aN the stores hold at the end, and the first recorded sweep at which open passed N/2. The width of the product and the
leftover curled pieces are read from the saved graphs at reading time with `scripts/read_endpoint_width.py` and the
sandbox's piece reader, as the registration says. Every six-link result carries VISION Update 24's caveat.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_REPLICAS = 6


def load(results="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(results) / "t65_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return rows


def k_of(row):
    d = row["dims"].split()
    return int(d[-1]) if "patch" in d else 0


def replica_outcome(final_row):
    n = int(final_row["N"])
    open_ = int(final_row["d3"])
    curled = int(final_row["d0"]) + int(final_row["d1"]) + int(final_row["d2"])
    damaged = int(final_row["damaged_d"])
    if open_ >= 0.9 * n and damaged <= 0.02 * n:
        return "OPENS"
    if curled >= 0.9 * n:
        return "FALLS BACK"
    if damaged > 0.1 * n:
        return "DAMAGED"
    return "PARTIAL"


def by_replica(rows):
    out = defaultdict(list)
    for r in rows:
        key = (r.get("tie_label") or "none", float(r["lam"]), k_of(r), int(r["C"]), int(r["N"]), float(r["spark"]), int(r["replica"]))
        out[key].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def cell_majority(outcomes):
    counts = Counter(outcomes)
    top, votes = counts.most_common(1)[0]
    return top if (votes > len(outcomes) / 2 and len(outcomes) >= MIN_REPLICAS) else "MIXED"


def cells(rows):
    """{(tie, lam, k, C, N, E): (majority, Counter, mean release per point, mean store share, first-open sweeps)}"""
    grouped = defaultdict(list)
    for key, blocks in by_replica(rows).items():
        grouped[key[:6]].append(blocks)
    out = {}
    for cell, runs in grouped.items():
        outs, releases, shares, firsts = [], [], [], []
        tie, lam, k, c, n, e = cell
        a = 4.0 * (lam - 1.0)
        for blocks in runs:
            first, last = blocks[0], blocks[-1]
            outs.append(replica_outcome(last))
            col = "h_tie_per_vertex" if tie != "none" else "h_per_vertex"
            releases.append(float(first[col]) - float(last[col]))
            shares.append(float(last["total"]) / (3.0 * a * n) if a > 0 else float("nan"))
            crossed = [int(b["sweep"]) for b in blocks if int(b["d3"]) > n / 2]
            firsts.append(crossed[0] if crossed else None)
        out[cell] = (cell_majority(outs), Counter(outs), sum(releases) / len(releases), sum(shares) / len(shares), firsts)
    return out


def lambda_verdict(majorities):
    """Over the tied arm-A cells at one lambda."""
    if any(m == "OPENS" for m in majorities):
        return "OPENS IN ROOM"
    if majorities and all(m == "FALLS BACK" for m in majorities):
        return "FALLS BACK"
    if any(m == "DAMAGED" for m in majorities):
        return "DAMAGED"
    return "PARTIAL"


def main(results="results"):
    rows = load(results)
    if not rows:
        print("no t65 results in", results)
        return
    c = cells(rows)
    per_lambda = defaultdict(list)
    for cell in sorted(c):
        tie, lam, k, cap, n, e = cell
        maj, counts, rel, share, firsts = c[cell]
        print("%-14s lam=%.2f k=%d C=%-5d N=%-4d E=%-5g | %-10s %s | release/pt %+.3f | stores hold %.2f of 3aN | first open>N/2 at %s"
              % (tie, lam, k, cap, n, e, maj, dict(counts), rel, share, [f for f in firsts]))
        if tie != "none" and k == 0:
            per_lambda[lam].append(maj)
    for lam in sorted(per_lambda):
        print("VERDICT lambda=%.2f (tied, arm A, both sizes): %s" % (lam, lambda_verdict(per_lambda[lam])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
