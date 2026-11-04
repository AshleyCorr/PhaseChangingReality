"""Read T68 (PREREGISTRATION T68): the untied slab under a feed whose rate varies, or that arrives at many points, at
three curling costs. Usage:

    python scripts/analyse_t68.py                     # every cell, then the word per (lambda, size)
    python scripts/analyse_t68.py --write-prediction  # the held-out ranges from stage 1 alone (rule 14), printed

The words per replica, per cell and per size are T67's, unchanged (`analyse_t67.replica_outcome`, `cell_word`,
`size_verdict`): OPENS, FRONT, BREAKS, SITS. A cell is (lambda, schedule, mode, k, C, rate, N). Per (lambda, N):
OPENS if any cell reads OPENS or FRONT, else BREAKS if every cell reads BREAKS, else MIXED. The drift must be below
1e-6 in every row or nothing is read.
"""
import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_t67 import cell_word, replica_outcome, size_verdict  # noqa: E402

KEY = ["lam", "schedule", "mode", "k", "C", "rate", "N"]


def cells(df):
    out = {}
    for key, g in df.groupby(KEY):
        outcomes = [replica_outcome(r) for _, r in g.groupby("replica")]
        words = [o["word"] for o in outcomes]
        stops = sorted(o["stop_sweep"] for o in outcomes)
        feds = sorted(o["fed"] for o in outcomes)
        out[key] = dict(word=cell_word(words), counts=Counter(words), median_stop=stops[len(stops) // 2],
                        median_fed=feds[len(feds) // 2], outcomes=outcomes)
    return out


def load(results="results", prefix="t68_"):
    paths = sorted(glob.glob(str(ROOT / results / (prefix + "*.csv"))))
    if not paths:
        raise SystemExit("no results/%s*.csv" % prefix)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    bad = df[df["drift"] > 1e-6]
    if len(bad):
        raise SystemExit("energy drift above 1e-6 in %d rows; not read" % len(bad))
    return df


def main(results="results", write_prediction=False):
    df = load(results)
    table = cells(df)
    print("lam | schedule | mode | k | C | rate | N | word | counts | median stop | median fed | first escape (median) | regions at stop (median) | fronts")
    for key in sorted(table):
        c = table[key]
        esc = sorted(o["first_escape"] for o in c["outcomes"] if o["first_escape"] is not None)
        reg = sorted(o["regions"] for o in c["outcomes"])
        fronts = sum(1 for o in c["outcomes"] if o["front"] is not None)
        print("%.2f | %s | %s | %d | %d | %g | %d | %s | %s | %d | %.0f | %s | %d | %d"
              % (*key, c["word"], dict(c["counts"]), c["median_stop"], c["median_fed"],
                 esc[len(esc) // 2] if esc else "none", reg[len(reg) // 2], fronts))
    for lam in sorted(df["lam"].unique()):
        for n in sorted(df["N"].unique()):
            words = [c["word"] for k, c in table.items() if k[0] == lam and k[6] == n]
            if words:
                print("lambda = %.2f, N = %d: %s" % (lam, n, size_verdict(words)))
    if write_prediction:
        print("\nHeld-out numbers (rule 14), from stage 1 alone, per (lambda, schedule, mode, rate): the words at the stage-1 sizes and the range of the median stop and median fed:")
        keys = sorted({(k[0], k[1], k[2], k[5]) for k in table})
        for lam, sched, mode, rate in keys:
            rows = [c for k, c in table.items() if (k[0], k[1], k[2], k[5]) == (lam, sched, mode, rate)]
            print("%.2f %s %s r=%g: words %s; median stop %d to %d; median fed %.0f to %.0f"
                  % (lam, sched, mode, rate, [c["word"] for c in rows], min(c["median_stop"] for c in rows),
                     max(c["median_stop"] for c in rows), min(c["median_fed"] for c in rows), max(c["median_fed"] for c in rows)))


if __name__ == "__main__":
    main(write_prediction="--write-prediction" in sys.argv[1:])
