"""T42: does concentrated energy fold flat six-link space when the points are interchangeable, near lambda = 1.02? Usage:

    python scripts/analyse_t42.py

PREREGISTRATION T42 (written 2026-09-25, before any run). Reads `results/t42_*.csv` (scripts/run_sealed_curled_d.py with
"interchangeable": true) and applies T34's rules unchanged (`analyse_t34.cells` and `verdict`), separately to the runs
with interchangeable points (weighted) and to the control with named points (the same chain, weighted false). The
verdict for the owner's question is the weighted one; the control says whether counting changed anything.
"""
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t34 import cells, verdict   # noqa: E402


def split(rows):
    weighted = [r for r in rows if r.get("weighted", "True") == "True"]
    named = [r for r in rows if r.get("weighted", "True") == "False"]
    return weighted, named


def main(out_dir="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "t42_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    if not rows:
        print("no T42 results yet")
        return
    for label, subset in zip(("interchangeable", "named control"), split(rows)):
        cm = cells(subset)
        for k in sorted(cm):
            c = cm[k]
            print("  %-15s N=%-4d E=%-6g n=%d %s -> %s | one direction %d, cascade %d, melt then fold %d, largest folded %d"
                  % (label, k[1], k[2], c["n"], c["counts"], c["majority"], c["one_direction"], c["cascade"],
                     c["melt_then_fold"], c["largest_folded"]))
        print("VERDICT T42 (%s): %s" % (label, verdict(cm)))


if __name__ == "__main__":
    main(*sys.argv[1:])
