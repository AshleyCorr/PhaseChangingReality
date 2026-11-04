"""T57: does too much concentrated energy curl flat six-link space under the owner's tie? Read by T34's rules. Usage:

    python scripts/analyse_t57.py [results_dir]

PREREGISTRATION T57 (written 2026-10-09, before any run). Reads `results/t57_*.csv` (scripts/run_sealed_curled_d.py,
T34's protocol with the table tie). The replica rule, the cell majority and the verdict words are T34's
(`analyse_t34`: FOLDED / MELTED / HEALED per replica from the final block; RE-CURLS / MELTS / HEALS / MIXED per
group), imported and not rewritten. What is new: the groups. The verdict is read per (tie, lambda), over that group's
cells (protocol, N, E); the tied groups are the test and the untied groups the control. Reported, not scored, as T34:
ONE DIRECTION and CASCADE replicas, the largest flat piece (*), melt-then-fold; and, new, the energy with the tie per
point at the end and the share of the spark left in the stores. Every six-link result carries VISION Update 24's
caveat.

(*) Corrected 11 October 2026, after the reading: this line was labeled "largest folded". The runner writes no
folded-piece column; what is read is `largest_flat`, the largest connected piece of points at d = 3. The line is
not scored and no rule is touched.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t34 import MIN_REPLICAS, melt_then_fold, outcome, reach   # noqa: E402


def load(results="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(results) / "t57_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return rows


def tie_of(row):
    return row.get("tie_label") or "none"


def by_replica(rows):
    out = defaultdict(list)
    for r in rows:
        proto = "packed" if r.get("local_heat", "False") == "True" else "spread"
        out[(tie_of(r), float(r["lam"]), proto, int(r["N"]), float(r["spark"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def cells(rows):
    """{(tie, lam, proto, N, E): (Counter of outcomes, majority or MIXED, [reach flags], largest flat piece, melt-then-fold)}"""
    reps = by_replica(rows)
    grouped = defaultdict(list)
    for key, blocks in reps.items():
        grouped[key[:5]].append(blocks)
    out = {}
    for cell, runs in grouped.items():
        outs = [outcome(b) for b in runs]
        counts = Counter(outs)
        top, votes = counts.most_common(1)[0]
        majority = top if (votes > len(outs) / 2 and len(outs) >= MIN_REPLICAS) else "MIXED"
        n = cell[3]
        reaches = [reach(b, n) for b in runs]
        largest = max(int(b[-1].get("largest_flat", 0) or 0) for b in runs)
        out[cell] = (dict(counts), majority, reaches, largest, sum(1 for b in runs if melt_then_fold(b)))
    return out


def group_verdict(majorities):
    """T34's verdict over one group's cell majorities."""
    if "FOLDED" in majorities:
        return "RE-CURLS"
    if "MELTED" in majorities:
        return "MELTS"
    if majorities and all(m == "HEALED" for m in majorities):
        return "HEALS"
    return "MIXED"


def summarize(rows):
    c = cells(rows)
    groups = defaultdict(list)
    for cell, (_, majority, _, _, _) in c.items():
        groups[cell[:2]].append(majority)
    return c, {g: group_verdict(m) for g, m in groups.items()}


def main(results="results"):
    rows = load(results)
    if not rows:
        print("no t57_*.csv in %s" % results)
        return 1
    c, verdicts = summarize(rows)
    print("T57, read by T34's rules (PREREGISTRATION T57, written 2026-10-09 before any run). Six links: Gate C open.")
    for cell, (counts, majority, reaches, largest, mtf) in sorted(c.items()):
        tie, lam, proto, n, e = cell
        print("%-16s lambda=%.2f %-6s N=%-3d E=%-6.0f %-42s -> %-7s | one direction %d, cascade %d of %d | largest flat piece %d | melt-then-fold %d"
              % (tie, lam, proto, n, e, counts, majority, sum(1 for a, _ in reaches if a), sum(1 for _, b in reaches if b),
                 len(reaches), largest, mtf))
    for (tie, lam), v in sorted(verdicts.items()):
        print("VERDICT T57 %-16s lambda=%.2f: %s" % (tie, lam, v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
