"""T7 observable 1, the waiting-time distribution, from any run that records phi per sweep.

    python scripts/analyse_t7_waiting.py cqg_tube_waiting_lam125

PREREGISTRATION.md T7: the waiting time is the sweep at which phi first leaves the tube's value
by more than three times its resting standard deviation, the latter measured over the first 200
sweeps. A constant-hazard process has exponential waiting times and coefficient of variation
(std / mean) 1. A CV near 1 alone does not establish that model; a smooth deformation has a
characteristic time and CV << 1. The prediction on record is CV in [0.7, 1.3] at every size.

Replicas with no recorded detector crossing are counted separately. The historical CV
diagnostic is computed over recorded crossings and with missing values placed at the run
length. That imputation is not a survival analysis: a missing detector crossing does not
establish that the graph remained an unconverted tube (see the T38 counterexample in O108).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(name, out_dir="results"):
    rows = list(csv.DictReader(open(Path(out_dir) / (name + ".csv"), newline="")))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[int(r["N"])][int(r["replica"])].append((int(r["sweep"]), float(r["phi"])))
    lam = float(rows[0]["lam"])
    phi_tube = 1.25          # 16 x 4 torus: one curled side, a quarter of a square per vertex
    print("T7 observable 1 from %s (lambda = %g).\n" % (name, lam))
    print("%-5s %-7s %-9s %-9s %-9s %-7s %-7s  %s"
          % ("N", "decays", "censored", "mean", "std", "CV", "CV*", "prediction: CV in [0.7, 1.3]"))
    for n in sorted(by):
        waits, censored = [], 0
        for rep, trace in sorted(by[n].items()):
            trace.sort()
            sweeps = np.array([t[0] for t in trace]); phi = np.array([t[1] for t in trace])
            rest = phi[sweeps <= 200]
            thresh = phi_tube - 3.0 * max(rest.std(), 1e-6) - 1e-9
            left = np.flatnonzero(phi < thresh)
            if len(left):
                waits.append(int(sweeps[left[0]]))
            else:
                censored += 1
        w = np.array(waits, float)
        if len(w) < 2:
            print("%-5d %-7d %-9d  too few decays" % (n, len(w), censored))
            continue
        cv = w.std(ddof=1) / w.mean()
        w_star = np.concatenate([w, np.full(censored, float(sweeps.max()))])
        cv_star = w_star.std(ddof=1) / w_star.mean()
        verdict = "in range" if 0.7 <= cv <= 1.3 else "OUTSIDE"
        print("%-5d %-7d %-9d %-9.0f %-9.0f %-7.2f %-7.2f  %s"
              % (n, len(w), censored, w.mean(), w.std(ddof=1), cv, cv_star, verdict))
    print("\n  CV  = over decays that happened.  CV* = with never-decayed replicas placed at the run"
          "\n  length, a lower bound on their waiting time. If the two differ much, censoring matters.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cqg_tube_waiting_lam125")
