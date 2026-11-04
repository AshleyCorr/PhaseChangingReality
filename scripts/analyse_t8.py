"""The pre-registered T8 reading: the lambda map along the tube. Usage:

    python scripts/analyse_t8.py

Implements PREREGISTRATION.md section T8 (written 2026-09-23, before the runs) on the committed files
results/t8_lam<tag>.csv and their _adj directories. T7's rules are reused exactly: gate 2 (at least 30
decays reaching 75 % conversion), gate 3 as amended (the release matches the sheet, the four-point ledge
at 24 lambda - 16, or the resting state read from its saved wiring, via analyse_t7_states.describe and
.gate), and predictions (a) to (c). T8's own definitions: metastable when f_200 < 0.25 in more than half
the decays; the waiting-time law holds when the mean wait is within 25 % of tau(lambda); the scrap is
reported and not scored. Pure functions of parsed rows, tested in tests/test_t8.py.

Lambda = 1.25 is not rerun (T8 says so); its T7 verdict is quoted beside the map, not recomputed.
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t7_states                                          # noqa: E402

G = 1.5
MIN_DECAYS = 30
ENERGY_TOL = 0.01
CV_RANGE = (0.7, 1.3)
TWO_STATE_FRAC = 0.80
FRONT_FRAC = 0.70
META_F200 = 0.25
LAW_TOL = 0.25


def tau(lam, g=G):
    """PREREGISTRATION T8, exact move counts: 3 A-moves and 2 B-moves offered per sweep."""
    return 1.0 / (3.0 * math.exp(-(32.0 - 16.0 * lam) / g) + 2.0 * math.exp(-(64.0 - 40.0 * lam) / g))


def metastable(rows):
    f = [float(r["f_200"]) for r in rows if r.get("f_200", "") != ""]
    return bool(f) and sum(1 for x in f if x < META_F200) > len(f) / 2


def classify_release(row, n, lam, read=None):
    """'sheet', 'ledge', 'read' (a resting state read from its wiring that matches), or 'neither'."""
    rel = float(row["released"])
    expect = 4.0 * (lam - 1.0)
    ring = (24.0 * lam - 16.0) / n
    if abs(rel - expect) <= ENERGY_TOL * expect:
        return "sheet"
    if abs(rel - (expect - ring)) <= ENERGY_TOL * expect:
        return "ledge"
    if read is not None and read(row):
        return "read"
    return "neither"


def read_from_wiring(adj_dir, lam):
    """A function row -> True when the saved final graph's energy matches the recorded release."""
    def check(row):
        f = Path(adj_dir) / ("N%s_rep%s.npz" % (row["N"], row["replica"]))
        if not f.exists():
            return False
        z = np.load(f)
        state = analyse_t7_states.describe(z["adj"], lam)
        _, ok = analyse_t7_states.gate(float(row["released"]), float(z["h0"]), state["h"], int(row["N"]), lam)
        return bool(ok)
    return check


def cell(rows, n, lam, read=None):
    """Everything T8 reports for one (lambda, N)."""
    out = dict(N=n, lam=lam, decays=len(rows), metastable=metastable(rows))
    f = [float(r["f_200"]) for r in rows if r.get("f_200", "") != ""]
    out["f200_median"] = float(np.median(f)) if f else math.nan
    kinds = [classify_release(r, n, lam, read) for r in rows]
    out.update({k: kinds.count(k) for k in ("sheet", "ledge", "read", "neither")})
    out["release_share"] = float(np.mean([float(r["released"]) for r in rows]) / (4.0 * (lam - 1.0)))
    if not out["metastable"]:
        out.update(gate2=False, gate3=False, a=False, b=False, c=False, sharp=False,
                   cv=math.nan, wait=math.nan, two=math.nan, largest=math.nan, law=False)
        return out
    reached = [r for r in rows if r["reached"] and float(r["reached"]) >= 0.75]
    out["gate2"] = len(reached) >= MIN_DECAYS
    out["gate3"] = all(classify_release(r, n, lam, read) != "neither" for r in reached)
    waits = np.array([float(r["waiting"]) for r in reached if r["waiting"] != ""])
    out["wait"] = float(waits.mean()) if len(waits) else math.nan
    out["cv"] = float(waits.std(ddof=1) / waits.mean()) if len(waits) > 1 else math.nan
    two = [(int(r["d1_50"]) + int(r["d2_50"])) / n for r in reached if r["d2_50"] != ""]
    largest = [float(r["largest_50"]) for r in reached if r["largest_50"] != ""]
    out["two"] = float(np.mean(two)) if two else math.nan
    out["largest"] = float(np.mean(largest)) if largest else math.nan
    out["a"] = CV_RANGE[0] <= out["cv"] <= CV_RANGE[1]
    out["b"] = out["two"] >= TWO_STATE_FRAC
    out["c"] = out["largest"] >= FRONT_FRAC
    out["sharp"] = out["a"] and out["b"] and out["c"]
    out["law"] = abs(out["wait"] / tau(lam) - 1.0) <= LAW_TOL if not math.isnan(out["wait"]) else False
    return out


def lam_status(cells):
    """Per lambda: 'not metastable', 'sharp', 'not sharp', or 'unread' (no metastable size passes its gates)."""
    meta = [c for c in cells if c["metastable"]]
    if not meta:
        return "not metastable"
    gated = [c for c in meta if c["gate2"] and c["gate3"]]
    if not gated:
        return "unread"
    if all(c["sharp"] for c in gated):
        return "sharp"
    if not any(c["sharp"] for c in gated):
        return "not sharp"
    return "unread"


def verdict(status):
    """status: {lambda: lam_status}. The pre-registered verdicts."""
    lams = sorted(status)
    below_edge = [l for l in lams if status[l] != "not metastable"]
    if not below_edge:
        return "INCONCLUSIVE (no metastable lambda)"
    edge = max(below_edge)
    inside = [l for l in lams if l <= edge]
    if any(status[l] == "not metastable" for l in inside):
        return "INCONCLUSIVE (metastability is not a single window)"
    if all(status[l] == "sharp" for l in inside):
        return "SHARP DOWN TO 1"
    for k in range(1, len(inside)):
        low, high = inside[:k], inside[k:]
        if all(status[l] == "not sharp" for l in low) and all(status[l] == "sharp" for l in high):
            return "A FLOOR (lambda* = %.2f)" % high[0]
    return "INCONCLUSIVE"


def main(out_dir="results"):
    status = {}
    for f in sorted(glob.glob(str(Path(out_dir) / "t8_lam*.csv"))):
        by_n = defaultdict(list)
        for r in csv.DictReader(open(f, newline="")):
            by_n[int(r["N"])].append(r)
        lam = float(next(iter(by_n.values()))[0]["lam"])
        read = read_from_wiring(Path(f).with_suffix("").as_posix() + "_adj", lam)
        cells = [cell(by_n[n], n, lam, read) for n in sorted(by_n)]
        status[lam] = lam_status(cells)
        print("lambda = %.2f  (tau predicted %.0f sweeps; x N for interchangeable points)  -> %s"
              % (lam, tau(lam), status[lam]))
        for c in cells:
            if not c["metastable"]:
                print("   N=%-4d NOT METASTABLE (median f_200 %.2f); ends: sheet %d, ledge %d, read %d, neither %d; "
                      "release %.2f of 4(lam-1)" % (c["N"], c["f200_median"], c["sheet"], c["ledge"], c["read"],
                                                    c["neither"], c["release_share"]))
                continue
            print("   N=%-4d gate2 %s gate3 %s | wait %.0f (law %s; interchangeable %.0f) CV %.2f | "
                  "d in {1,2} %.3f | largest %.2f | a b c %s %s %s | ends: sheet %d, ledge %d, read %d, neither %d"
                  % (c["N"], "pass" if c["gate2"] else "FAIL", "pass" if c["gate3"] else "FAIL", c["wait"],
                     "holds" if c["law"] else "fails", c["wait"] * c["N"], c["cv"], c["two"], c["largest"],
                     *("Y" if c[k] else "n" for k in "abc"), c["sheet"], c["ledge"], c["read"], c["neither"]))
    print("\nlambda = 1.25 (T7, not rerun): TWO-STATE CHANGE at N = 64, 96, 192; N = 144 set aside by gate 3.")
    print("PRE-REGISTERED VERDICT:", verdict(status))


if __name__ == "__main__":
    main(*sys.argv[1:])
