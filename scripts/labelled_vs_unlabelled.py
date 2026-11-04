"""Named points against interchangeable ones, exactly, from the enumeration already on disk (T10).

    python scripts/labelled_vs_unlabelled.py [results/ergodicity_small.csv]

Prints to the screen and writes nothing: it is arithmetic on a committed result, not a new run.

THE TWO ENSEMBLES. `results/ergodicity_small.csv` lists every shape at N = 16 and 18, with the
number of renamings that leave it unchanged (A) and the number of distinct named versions it
has, which is (n!)^2 / A. So:

    named           weight a shape by (n!)^2 / A     every named arrangement counts once
    interchangeable weight a shape by 1              every shape counts once

and the second therefore favours a shape by exactly A. That is the whole of the difference, and
it is exact (ASSUMPTION Q15). The same ratio is what the acceptance rule of [DQM25] Eq. (72)
samples by multiplying the Metropolis ratio by |Aut(G')| / |Aut(G)|.

WHY IT MATTERS HERE. A point is distinguishable only to the extent that its relationships
distinguish it, so A measures how much of an arrangement is genuinely interchangeable rather
than distinguished by bookkeeping. Regular, repetitive arrangements gain and irregular ones
lose, and the question this raises for the project is whether that changes the ORDER of the
transition, which these sizes are far too small to show.
"""
import csv
import sys
from math import exp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ergodicity_small.csv"

with path.open(newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["cap"] == "none" and r["class_id"] != "-1"]


def ensemble(n, lam, g, interchangeable):
    sel = [r for r in rows if r["N"] == str(n)]
    weights = []
    for r in sel:
        h = 16 * (n - int(r["squares"])) + 4 * lam * int(r["surplus"])
        mult = 1.0 if interchangeable else float(r["labelled_states"])
        weights.append(mult * exp(-h / g))
    z = sum(weights)
    phi = sum(w * int(r["squares"]) for w, r in zip(weights, sel)) / z / n
    return phi, {id(r): w / z for w, r in zip(weights, sel)}


for n in (16, 18):
    sel = [r for r in rows if r["N"] == str(n)]
    densest = max(sel, key=lambda r: int(r["squares"]))
    print("\nN = %d: %d shapes, %d named arrangements. Densest shape: S = %s, %s renamings, "
          "%s pieces, %s of it in 4-cubes."
          % (n, len(sel), sum(int(r["labelled_states"]) for r in sel), densest["squares"],
             densest["symmetries"], densest["pieces"], densest["cubes"]))
    for lam in (0.0, 1.0):
        print("  penalty = %.1f" % lam)
        print("      %-6s %-12s %-14s %-10s %s" % ("temp", "phi named", "phi interch.", "shift",
                                                   "share of the densest shape"))
        for g in (20.0, 8.0, 4.0, 2.0):
            a, wa = ensemble(n, lam, g, False)
            b, wb = ensemble(n, lam, g, True)
            fa, fb = wa[id(densest)], wb[id(densest)]
            print("      %-6.1f %-12.5f %-14.5f %+ -10.5f %.4f -> %.4f  (x%.1f)"
                  % (g, a, b, b - a, fa, fb, fb / fa if fa else float("inf")))

print("\nThese sizes are a single piece and cannot shatter, so this is the mechanism and not its")
print("consequence. For the same comparison at simulable sizes see scripts/measure_symmetry_cost.py,")
print("which measures the renaming count of a shattered state directly.")
