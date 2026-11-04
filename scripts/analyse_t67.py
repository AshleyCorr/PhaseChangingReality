"""Read T67 (PREREGISTRATION T67, written 2026-10-16 before any registered run). Usage:

    python scripts/analyse_t67.py                    # every results/t67_*.csv
    python scripts/analyse_t67.py --write-prediction # the held-out numbers from stage 1 alone (rule 14), printed

Per replica, from its records in order: the FIRST ESCAPE sweep = the first record with curled < N; the FRONT sweep =
the first record whose largest open region exceeds N / 2 while damaged <= N / 10; the COMPLETION sweep = the first
record with open >= 0.9 N. The replica's word: OPENS if it completed with damaged <= N / 10 at that record; FRONT if a
front sweep exists but no completion; BREAKS if damaged exceeded N / 10 before any front; SITS otherwise (the run
ended with neither). Per cell (mode, C, rate, N): the word held by more than half the replicas, else MIXED; the
median stop sweep; the median energy fed at the stop. Per size: OPENS if any cell reads OPENS or FRONT (the slab can
be opened by a feed at that size), else BREAKS if every cell reads BREAKS, else MIXED. Reported, not scored: the first
escape sweeps, the number of open regions at the stop, the end census, the store mean at the stop.
"""
import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_REPLICAS = 4


def replica_outcome(rows):
    """rows: one replica's records. Returns dict(word, first_escape, front, completion, stop_sweep, fed, regions)."""
    rows = rows.sort_values("sweep")
    n = int(rows["N"].iloc[0])
    esc = rows[rows["curled"] < n]
    front = rows[(rows["largest"] > n / 2) & (rows["damaged"] <= n / 10)]
    comp = rows[rows["open"] >= 0.9 * n]
    broke = rows[rows["damaged"] > n / 10]
    first_escape = int(esc["sweep"].iloc[0]) if len(esc) else None
    front_sweep = int(front["sweep"].iloc[0]) if len(front) else None
    comp_sweep = int(comp["sweep"].iloc[0]) if len(comp) else None
    broke_sweep = int(broke["sweep"].iloc[0]) if len(broke) else None
    if comp_sweep is not None and int(comp["damaged"].iloc[0]) <= n / 10:
        word = "OPENS"
    elif front_sweep is not None and (broke_sweep is None or front_sweep < broke_sweep):
        word = "FRONT"
    elif broke_sweep is not None:
        word = "BREAKS"
    else:
        word = "SITS"
    last = rows.iloc[-1]
    return dict(word=word, first_escape=first_escape, front=front_sweep, completion=comp_sweep,
                stop_sweep=int(last["sweep"]), fed=float(last["fed"]), regions=int(last["regions"]),
                end_open=int(last["open"]), end_damaged=int(last["damaged"]), store_mean=float(last["store_mean"]))


def cell_word(words):
    counts = Counter(words)
    top, votes = counts.most_common(1)[0]
    return top if (votes > len(words) / 2 and len(words) >= MIN_REPLICAS) else "MIXED"


def cells(df):
    """{(mode, C, rate, N): dict(word, counts, median_stop, median_fed, outcomes)}"""
    out = {}
    for key, g in df.groupby(["mode", "C", "rate", "N"]):
        outcomes = [replica_outcome(r) for _, r in g.groupby("replica")]
        words = [o["word"] for o in outcomes]
        stops = sorted(o["stop_sweep"] for o in outcomes)
        feds = sorted(o["fed"] for o in outcomes)
        out[key] = dict(word=cell_word(words), counts=Counter(words), median_stop=stops[len(stops) // 2],
                        median_fed=feds[len(feds) // 2], outcomes=outcomes)
    return out


def size_verdict(cell_words):
    if any(w in ("OPENS", "FRONT") for w in cell_words):
        return "OPENS"
    if cell_words and all(w == "BREAKS" for w in cell_words):
        return "BREAKS"
    return "MIXED"


def load(results="results"):
    paths = sorted(glob.glob(str(ROOT / results / "t67_*.csv")))
    if not paths:
        raise SystemExit("no results/t67_*.csv")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    bad = df[df["drift"] > 1e-6]
    if len(bad):
        raise SystemExit("energy drift above 1e-6 in %d rows; not read" % len(bad))
    return df


def main(results="results", write_prediction=False):
    df = load(results)
    table = cells(df)
    sizes = sorted(df["N"].unique())
    print("cell | word | counts | median stop sweep | median fed at stop | first escape (median) | regions at stop (median)")
    for key in sorted(table):
        c = table[key]
        esc = sorted(o["first_escape"] for o in c["outcomes"] if o["first_escape"] is not None)
        reg = sorted(o["regions"] for o in c["outcomes"])
        print("%s C=%d r=%g N=%d | %s | %s | %d | %.0f | %s | %d"
              % (key[0], key[1], key[2], key[3], c["word"], dict(c["counts"]), c["median_stop"], c["median_fed"],
                 esc[len(esc) // 2] if esc else "none", reg[len(reg) // 2]))
    for n in sizes:
        words = [c["word"] for k, c in table.items() if k[3] == n]
        print("N = %d: %s" % (n, size_verdict(words)))
    if write_prediction:
        print("\nHeld-out numbers (rule 14), from stage 1 alone, per (mode, C, rate): the word at both sizes, and the range of the median stop sweep and median fed at stop across the stage-1 sizes:")
        keys = sorted({(k[0], 1 if k[0] == "spot" else k[1] // k[3], k[2]) for k in table})
        for mode, cper, rate in keys:
            rows = [c for k, c in table.items() if k[0] == mode and k[2] == rate and (mode == "spot" or k[1] // k[3] == cper)]
            print("%s C=%s r=%g: words %s; median stop %d to %d; median fed %.0f to %.0f"
                  % (mode, ("N" if mode == "spot" else "%dN" % cper), rate, [c["word"] for c in rows],
                     min(c["median_stop"] for c in rows), max(c["median_stop"] for c in rows),
                     min(c["median_fed"] for c in rows), max(c["median_fed"] for c in rows)))


if __name__ == "__main__":
    main(write_prediction="--write-prediction" in sys.argv[1:])
