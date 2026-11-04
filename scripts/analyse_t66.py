"""Read T66 (PREREGISTRATION T66, written 2026-10-16 before any registered run). Usage:

    python scripts/analyse_t66.py                    # every results/t66_*.csv
    python scripts/analyse_t66.py --write-prediction # the held-out numbers from stage 1 alone (rule 14), printed

Per replica: peak = the largest damaged count over the records; the scar level = ceil(N / 100) damaged points; the
healing time = sweeps from the peak record to the first record at or below the scar level; a replica that never gets
there is NOT HEALED and its time is censored at (last sweep - peak sweep). Per cell (D, N, rho): healed share, and the
median healing time with censored times counted as larger than any uncensored (if more than half are censored the
median is "beyond the run"). Per (N, rho): FASTER IN 3D if the three-direction cell's healed share is at least the
two-direction cell's and its median time is smaller (a median beyond the run counts as larger than any number);
SLOWER IN 3D if both the other way; else MIXED. Per size: FASTER IN 3D if both rho read FASTER IN 3D, SLOWER IN 3D if
both read SLOWER, else MIXED. Reported, not scored: the matched-by-breaks comparison (2D at rho = 1 against 3D at
rho = 2), the end damage, the store mean at the end.
"""
import glob
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEYOND = "beyond the run"


def replica_stats(rows, n):
    """rows: one replica's records sorted by sweep. Returns (peak, peak_sweep, time, healed)."""
    rows = rows.sort_values("sweep")
    dam = rows["damaged"].to_numpy()
    sweeps = rows["sweep"].to_numpy()
    ip = int(dam.argmax())
    scar = math.ceil(n / 100)
    after = dam[ip:]
    hit = [i for i, v in enumerate(after) if v <= scar]
    if hit:
        return int(dam[ip]), int(sweeps[ip]), int(sweeps[ip + hit[0]] - sweeps[ip]), True
    return int(dam[ip]), int(sweeps[ip]), int(sweeps[-1] - sweeps[ip]), False


def cell_stats(df):
    """df: one (D, N, rho) cell. Returns dict(healed_share, median_time (number or BEYOND), n, peak_median, ...)."""
    n = int(df["N"].iloc[0])
    stats = [replica_stats(g, n) for _, g in df.groupby("replica")]
    reps = len(stats)
    healed = [s for s in stats if s[3]]
    share = len(healed) / reps
    if len(healed) * 2 <= reps:
        median = BEYOND
    else:
        times = sorted(s[2] for s in healed) + [float("inf")] * (reps - len(healed))
        median = times[(reps - 1) // 2] if reps % 2 else (times[reps // 2 - 1] + times[reps // 2]) / 2
        if median == float("inf"):
            median = BEYOND
    end = df[df["sweep"] == df["sweep"].max()]
    return dict(n=n, replicas=reps, healed_share=share, median_time=median,
                peak_median=float(pd.Series([s[0] for s in stats]).median()),
                end_damaged_median=float(end["damaged"].median()), end_store_mean=float(end["store_mean"].mean()))


def compare(two, three):
    """FASTER IN 3D / SLOWER IN 3D / MIXED from two cell_stats dicts."""
    def less(a, b):      # a < b with BEYOND larger than any number
        if a == BEYOND:
            return False
        if b == BEYOND:
            return True
        return a < b
    if three["healed_share"] >= two["healed_share"] and less(three["median_time"], two["median_time"]):
        return "FASTER IN 3D"
    if three["healed_share"] <= two["healed_share"] and less(two["median_time"], three["median_time"]):
        return "SLOWER IN 3D"
    return "MIXED"


def size_verdict(words):
    if all(w == "FASTER IN 3D" for w in words):
        return "FASTER IN 3D"
    if all(w == "SLOWER IN 3D" for w in words):
        return "SLOWER IN 3D"
    return "MIXED"


def load(paths):
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main(argv):
    paths = sorted(glob.glob(str(ROOT / "results" / "t66_*.csv")))
    if not paths:
        print("no results/t66_*.csv"); return
    df = load(paths)
    cells = {}
    for (dim, n, rho), g in df.groupby(["D", "N", "rho"]):
        cells[(int(dim), int(n), float(rho))] = cell_stats(g)
        c = cells[(int(dim), int(n), float(rho))]
        print("D=%d N=%d rho=%g: healed %d/%d, median time %s, peak median %.0f, end damaged median %.0f, store mean %.3f"
              % (dim, n, rho, round(c["healed_share"] * c["replicas"]), c["replicas"], c["median_time"], c["peak_median"],
                 c["end_damaged_median"], c["end_store_mean"]))
    for n in sorted({k[1] for k in cells}):
        words = []
        for rho in sorted({k[2] for k in cells if k[1] == n}):
            if (2, n, rho) in cells and (3, n, rho) in cells:
                w = compare(cells[(2, n, rho)], cells[(3, n, rho)])
                words.append(w)
                print("N=%d rho=%g: %s" % (n, rho, w))
        if words:
            print("VERDICT N=%d: %s" % (n, size_verdict(words)))
        if (2, n, 1.0) in cells and (3, n, 2.0) in cells:
            print("  matched by breaks (2D rho=1 against 3D rho=2), reported not scored: %s"
                  % compare(cells[(2, n, 1.0)], cells[(3, n, 2.0)]))
    if "--write-prediction" in argv:
        small = [k for k in cells if k[1] in (216, 512) and k[0] == 3]
        for rho in sorted({k[2] for k in small}):
            ts = [cells[k]["median_time"] for k in small if k[2] == rho and cells[k]["median_time"] != BEYOND]
            print("held-out numbers rho=%g: 3D median times at the smaller sizes %s" % (rho, ts))


if __name__ == "__main__":
    main(sys.argv[1:])
