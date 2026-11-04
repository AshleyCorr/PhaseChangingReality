"""EXPLORATORY, read-only (12 October 2026): what did "melted" mean in the sealed-tube runs of T9? Usage:

    python scripts/read_t9_squares.py [results_dir]

Why. T9's registered classes call a run "melted" when at least 15 % of its vertices sit at d = 0 or d >= 3, and
the record and paper 1 went on to describe those runs as back in the random phase. The owner asked on 12 October
what follows if "melted" is not a thing. Exact, and overlooked until then: a sealed run holds only the energy it
started with, the tube's 4 (lambda - 1) per vertex plus the push, and a missing square costs 16. With no surplus
squares, H = 16 (N - S), so the squares per vertex can never fall below 1 - (4 (lambda - 1) N + push) / (16 N).
At lambda = 1.25 that is about fifteen squares in sixteen. The burp cannot melt its own product.

What is read. The final row of every T9 run (results/t9_n*_C*.csv), classified by scripts/analyse_t9.classify as
registered, with its squares per vertex. Nothing here is a registered rule and no verdict changes; it reads
results/ and writes nothing.
"""
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t9                                                 # noqa: E402

LAM, PUSH = 1.25, 12.0            # T9's curling cost and the push in one store (PREREGISTRATION T9)
RANDOM_PHASE = 0.126              # squares per vertex of the random phase at N = 160 ([T25]; ASSUMPTIONS O5)


def floor(n, lam=LAM, push=PUSH):
    """The fewest squares per vertex a sealed tube of n vertices can ever have: every unit of its energy spent on
    missing squares, at 16 each, with no surplus squares left."""
    return 1.0 - (4.0 * (lam - 1.0) * n + push) / (16.0 * n)


def final_rows(results="results"):
    for f in sorted(glob.glob(str(Path(results) / "t9_n*_C*.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["final"] in ("1", "True", "true"):
                    yield r


def by_class(results="results"):
    """{(N, class): [squares per vertex of each run's final state]}."""
    out = defaultdict(list)
    for r in final_rows(results):
        out[(int(r["N"]), analyse_t9.classify(r))].append(float(r["phi"]))
    return out


def main(results="results"):
    groups = by_class(results)
    print("T9's sealed tubes, final states: squares per vertex by registered class (exploratory reading).")
    print("The random phase has about %.2f squares per vertex; flat fabric has 1; the tube 1.25.\n" % RANDOM_PHASE)
    print("%-5s %-8s %5s  %s   %s" % ("N", "class", "runs", "squares per vertex: lowest, median, highest",
                                     "the energy's floor"))
    for (n, kind), phi in sorted(groups.items()):
        print("%-5d %-8s %5d  %.3f  %.3f  %.3f                         %.3f"
              % (n, kind, len(phi), min(phi), statistics.median(phi), max(phi), floor(n)))
    melted = [p for (n, kind), phi in groups.items() if kind == "melted" for p in phi]
    if melted:
        print("\nAll %d runs classed melted: %.3f to %.3f squares per vertex. None is near the random phase."
              % (len(melted), min(melted), max(melted)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
