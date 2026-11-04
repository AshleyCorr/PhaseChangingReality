"""Exact averages at N = 16 or 18 from the complete list of states (task T4). Usage:

    python scripts/exact_small_averages.py results/ergodicity_small.csv 18 5,10 0,1,1.5,2,3,5,10,30

Arguments: the class table written by check_ergodicity.py, N, a comma-separated
list of couplings g, and one of values of lambda. No sampling: every labelled
state is in the table with its squares S and surplus X, so with
H = 16 (N - S) + 4 lambda X the average of anything is a finite sum with weights
(labelled states in the class) x exp(-H / g). Printed for each (g, lambda):
phi = <S>/N, <X>/N, and the share of the weight on "folded" states, those with
at least one edge carrying more than two squares (X > 0). The last column is the
capped model, which is the same sum restricted to the states valid under the cap.
Prints only; writes nothing.
"""
import csv
import sys
from math import exp


def averages(classes, n, lam, g, capped=False):
    use = [c for c in classes if c["valid_under_cap"] == "True" or not capped]
    weights = [int(c["labelled_states"]) * exp(-(16 * (n - int(c["squares"])) + 4 * lam * int(c["surplus"])) / g)
               for c in use]
    total = sum(weights)
    phi = sum(w * int(c["squares"]) for w, c in zip(weights, use)) / total / n
    surplus = sum(w * int(c["surplus"]) for w, c in zip(weights, use)) / total / n
    folded = sum(w for w, c in zip(weights, use) if int(c["surplus"]) > 0) / total
    return phi, surplus, folded


def main(table, n, couplings, lambdas):
    with open(table, newline="") as fh:
        classes = [r for r in csv.DictReader(fh) if r["N"] == str(n) and r["cap"] == "none" and r["class_id"] != "-1"]
    if not classes:
        raise SystemExit(f"no states for N = {n} without the cap in {table}")
    print(f"N = {n}: {len(classes)} classes, {sum(int(c['labelled_states']) for c in classes):,} labelled states")
    print(f"{'g':>6} {'lambda':>7} | {'phi':>7} {'X/N':>7} {'folded':>7}")
    for g in couplings:
        for lam in lambdas:
            phi, surplus, folded = averages(classes, n, lam, g)
            print(f"{g:6.2f} {lam:7.2f} | {phi:7.4f} {surplus:7.4f} {folded:7.1%}")
        if any(c["valid_under_cap"] == "True" for c in classes):
            phi, _, _ = averages(classes, n, 0.0, g, capped=True)
            print(f"{g:6.2f} {'cap':>7} | {phi:7.4f} {0.0:7.4f} {0.0:7.1%}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [float(x) for x in sys.argv[3].split(",")],
         [float(x) for x in sys.argv[4].split(",")])
