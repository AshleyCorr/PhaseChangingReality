"""The pre-registered T60 reading: the lambda map a fourth time, with the wait timed with no detector and a size
held out. Usage:

    python scripts/analyse_t60.py [results_dir]                     # the report
    python scripts/analyse_t60.py [results_dir] --write-prediction  # after stage 1: fix the numbers for N = 288

Implements PREREGISTRATION.md section T60 (written 2026-10-10, before the runs) on results/t60_lam<tag>_n<N>.csv
and their _adj directories. Everything is T24's (scripts/analyse_t24.py: (b) two orders side by side, (c) one
front, gate 2, gate 3', the exact resting states, the window and edge verdicts) except the memoryless check, which
becomes (a''): the wait is the first look, from sweep 0, at which a decay no longer reads as the perfect tube
(`first_left`, recorded with no detector, no 200-sweep watch and no threshold), taken over every decay of the cell.
Its coefficient of variation must lie inside T23's central 99.9 % band for that many memoryless waits.

Two stages. Stage 1 is the grid of the three earlier maps (N = 64 to 192) and gives the window and edge verdicts.
From stage 1 alone, three numbers per window lambda are written for N = 288 by a recipe fixed in the registration
(`predict`), saved to configs/t60_heldout_prediction.json before stage 2 runs, and the size claim is scored at
N = 288 against that file (CLAUDE.md rule 14). Pure functions of parsed rows, tested in tests/test_t60.py.
"""
import csv
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t8                                                 # noqa: E402
import analyse_t23                                                # noqa: E402
import analyse_t24                                                # noqa: E402

STAGE1_SIZES = (64, 96, 144, 192)
HELD_OUT = 288
CAP = {64: 100000, 96: 100000, 144: 100000, 192: 100000, 288: 200000}     # n_sweeps of each size's configs
RATIO_RANGE = 0.32                    # the mean wait over tau at N = 288: within this share of the predicted value
TWO_RANGE = 0.03                      # (b) at N = 288: within this of the fitted line
FRONT_RANGE = 0.10                    # (c) at N = 288: within this of the fitted line
PREDICTION = Path(__file__).resolve().parents[1] / "configs" / "t60_heldout_prediction.json"


def first_exits(rows, cap):
    """The wait of every decay with no detector: the first look at which it no longer read as the perfect tube.
    A decay that never left enters at the cap (a lower bound) and is counted in the second value returned."""
    never = sum(1 for r in rows if r["first_left"] == "")
    return np.array([float(r["first_left"]) if r["first_left"] != "" else float(cap) for r in rows]), never


def cell(rows, n, lam, states, cap=None):
    """T24's cell with (a') replaced by (a'')."""
    out = analyse_t24.cell(rows, n, lam, states)
    if not out["metastable"]:
        return out
    waits, never = first_exits(rows, CAP[n] if cap is None else cap)
    out.update(n_read=len(waits), never_left=never)
    if len(waits) < 2 or waits.mean() <= 0:
        out.update(cv=math.nan, band=(math.nan, math.nan), a=False, sharp=False, wait=math.nan, ratio=math.nan)
        return out
    out["wait"] = float(waits.mean())
    out["ratio"] = out["wait"] / analyse_t8.tau(lam)
    out["cv"] = float(waits.std(ddof=1) / waits.mean())
    out["band"] = analyse_t23.cv_band(len(waits))
    out["a"] = out["band"][0] <= out["cv"] <= out["band"][1]
    out["sharp"] = out["a"] and out["b"] and out["c"]
    out["law"] = abs(out["ratio"] - 1.0) <= analyse_t8.LAW_TOL
    return out


def line_at(sizes, values, at):
    """The least-squares straight line through (size, value), read at `at`."""
    x, y = np.asarray(sizes, dtype=float), np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * at)


def predict(cells):
    """The registered recipe. `cells` are the four stage-1 cells of one lambda (all metastable). Returns, for
    N = 288: the predicted mean wait over tau with its range, and (b) and (c) with theirs."""
    cells = sorted(cells, key=lambda c: c["N"])
    sizes = [c["N"] for c in cells]
    assert tuple(sizes) == STAGE1_SIZES, sizes
    ratio = float(np.mean([c["ratio"] for c in cells]))
    two = min(1.0, max(0.0, line_at(sizes, [c["two"] for c in cells], HELD_OUT)))
    front = min(1.0, max(0.0, line_at(sizes, [c["largest"] for c in cells], HELD_OUT)))
    return dict(ratio=[ratio, ratio * (1.0 - RATIO_RANGE), ratio * (1.0 + RATIO_RANGE)],
                two=[two, max(0.0, two - TWO_RANGE), min(1.0, two + TWO_RANGE)],
                largest=[front, max(0.0, front - FRONT_RANGE), min(1.0, front + FRONT_RANGE)],
                sharp=bool(two >= analyse_t8.TWO_STATE_FRAC and front >= analyse_t8.FRONT_FRAC))


def misses(c, p):
    """What of the held-out cell `c` falls outside the prediction `p`: a list of strings, empty if nothing does."""
    if not c["metastable"]:
        return ["not metastable"]
    out = []
    if not c["gate2"]:
        out.append("gate 2")
    if not c["gate3"]:
        out.append("gate 3'")
    if not c["a"]:
        out.append("memoryless: CV %.3f outside %.3f to %.3f" % (c["cv"], c["band"][0], c["band"][1]))
    for key, label in (("ratio", "mean wait over tau"), ("two", "two orders"), ("largest", "one front")):
        if not (p[key][1] <= c[key] <= p[key][2]):
            out.append("%s %.3f outside %.3f to %.3f" % (label, c[key], p[key][1], p[key][2]))
    return out


def held_out_verdict(cells, predictions):
    """cells, predictions: {lambda: ...} over the window. Returns (first line, second line) of the verdict."""
    missing = [l for l in analyse_t23.WINDOW if l not in cells or l not in predictions]
    if missing:
        return ("NOT READ (no data or no prediction at %s)" % ", ".join("%.2f" % l for l in missing), "")
    wrong = {l: misses(cells[l], predictions[l]) for l in analyse_t23.WINDOW}
    wrong = {l: m for l, m in wrong.items() if m}
    first = ("AS PREDICTED AT THE HELD-OUT SIZE" if not wrong else
             "NOT AS PREDICTED AT " + "; ".join("%.2f (%s)" % (l, ", ".join(m)) for l, m in sorted(wrong.items())))
    gated = [l for l in analyse_t23.WINDOW if cells[l]["metastable"] and cells[l]["gate2"] and cells[l]["gate3"]]
    sharp = [l for l in gated if cells[l]["sharp"]]
    blunt = [l for l in analyse_t23.WINDOW if l not in sharp]
    second = "sharp at N = %d at %s" % (HELD_OUT, ", ".join("%.2f" % l for l in sharp) if sharp else "no lambda")
    if blunt:
        second += "; not sharp or not read at " + ", ".join("%.2f" % l for l in blunt)
    return first, second


def load(out_dir="results"):
    data = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(out_dir) / "t60_lam*_n*.csv"))):
        m = re.search(r"t60_lam(\d+)_n(\d+)\.csv$", f)
        lam, n = int(m.group(1)) / 100.0, int(m.group(2))
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        data[lam][n] = (rows, Path(f).with_suffix("").as_posix() + "_adj")
    return data


def read_cells(data):
    cells, pooled = defaultdict(dict), defaultdict(list)
    for lam in sorted(data):
        for n in sorted(data[lam]):
            rows, adj_dir = data[lam][n]
            states = analyse_t24.read_final_states(rows, adj_dir, lam)
            cells[lam][n] = cell(rows, n, lam, states)
            if n in STAGE1_SIZES:
                pooled[lam] += [(r, r["replica"] in states and states[r["replica"]]["kind"] == "FLAT") for r in rows]
    return cells, pooled


def show(c):
    if not c["metastable"]:
        print("lambda=%.2f N=%-4d NOT METASTABLE (median f_200 %.2f)" % (c["lam"], c["N"], c["f200_median"]))
        return
    print("lambda=%.2f N=%-4d gate2 %s gate3' %s | %d waits, CV %.3f in [%.3f, %.3f] %s | d in {1,2} %.3f | "
          "largest %.2f | sharp %s | mean wait %.0f = %.3f of tau (never left %d) | ends flat %d ledge %d other %d | "
          "release exact %.4f"
          % (c["lam"], c["N"], "pass" if c["gate2"] else "FAIL", "pass" if c["gate3"] else "FAIL", c["n_read"],
             c["cv"], c["band"][0], c["band"][1], "Y" if c["a"] else "n", c["two"], c["largest"],
             "Y" if c["sharp"] else "n", c["wait"], c["ratio"], c["never_left"], c["flat"], c["ledge"], c["other"],
             c["release_exact"]))


def stage1_predictions(cells):
    """{lambda: predict(...)} for every window lambda whose four stage-1 cells are present and metastable."""
    out = {}
    for lam in analyse_t23.WINDOW:
        four = [cells[lam][n] for n in STAGE1_SIZES if n in cells.get(lam, {})]
        if len(four) == len(STAGE1_SIZES) and all(c["metastable"] for c in four):
            out[lam] = predict(four)
    return out


def main(out_dir="results", flag=""):
    cells, pooled = read_cells(load(out_dir))
    print("T60: the lambda map a fourth time. STAGE 1 (N = 64 to 192), the wait timed with no detector\n")
    status = {}
    for lam in sorted(cells):
        stage1 = [cells[lam][n] for n in STAGE1_SIZES if n in cells[lam]]
        for c in stage1:
            show(c)
        if len(stage1) == len(STAGE1_SIZES):
            status[lam] = analyse_t8.lam_status(stage1)
    done = sum(len([n for n in cells[lam] if n in STAGE1_SIZES]) for lam in cells)
    if done < 28:
        print("\nSTAGE 1 is not complete (%d of 28 cells). No verdict is read." % done)
        return
    window = {l: s for l, s in status.items() if l in analyse_t23.WINDOW}
    print("\nWINDOW (1.05 to 1.30, N = 64 to 192):", analyse_t23.window_verdict(window))
    e = analyse_t23.edge_report(pooled[analyse_t23.BELOW_EDGE], pooled[analyse_t23.EDGE])
    print("EDGE (1.35 against 1.30): ending flat %.2f -> %.2f (E1 %s); converted region in more than one piece at "
          "25%% %.2f -> %.2f (E2 %s): %s"
          % (*e["flat"], "yes" if e["e1"] else "no", *e["multi"], "yes" if e["e2"] else "no", e["verdict"]))

    recipe = stage1_predictions(cells)
    print("\nTHE RECIPE'S NUMBERS FOR N = %d, from stage 1 alone (value, low, high):" % HELD_OUT)
    for lam in sorted(recipe):
        p = recipe[lam]
        print("  lambda=%.2f mean wait over tau %.3f (%.3f to %.3f) | two orders %.3f (%.3f to %.3f) | one front %.3f "
              "(%.3f to %.3f) | predicts %s"
              % (lam, *p["ratio"], *p["two"], *p["largest"], "sharp" if p["sharp"] else "NOT sharp"))
    if flag == "--write-prediction":
        if PREDICTION.exists():
            raise SystemExit("refusing to overwrite %s" % PREDICTION)
        if len(recipe) < len(analyse_t23.WINDOW):
            raise SystemExit("stage 1 does not give a prediction at every window lambda; nothing written")
        PREDICTION.write_text(json.dumps({"%.2f" % l: p for l, p in sorted(recipe.items())}, indent=2) + "\n",
                              encoding="utf-8", newline="\n")
        print("\nwrote", PREDICTION.name)
        return

    print("\nSTAGE 2 (N = %d, held out)" % HELD_OUT)
    if not PREDICTION.exists():
        print("  no prediction file yet: stage 2 is not run or read until it is committed.")
        return
    fixed = {float(k): v for k, v in json.loads(PREDICTION.read_text(encoding="utf-8")).items()}
    held = {lam: cells[lam][HELD_OUT] for lam in cells if HELD_OUT in cells[lam]}
    if len(held) < 7:
        print("  %d of 7 cells. Not complete; not read." % len(held))
        return
    for lam in sorted(held):
        show(held[lam])
    first, second = held_out_verdict(held, fixed)
    print("\nHELD-OUT SIZE:", first)
    print("             ", second)


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    paths = [a for a in args if not a.startswith("--")]
    main(paths[0] if paths else "results", flags[0] if flags else "")
