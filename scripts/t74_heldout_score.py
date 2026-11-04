"""Score T74's held-out size L = 16 against the numbers written before it ran (PREREGISTRATION.md T74, Amendment 6;
CLAUDE.md rule 14). Usage:

    PYTHONPATH="src;scripts" python scripts/t74_heldout_score.py [configs/t74_heldout_prediction.json]

Reads the 54 L = 16 files `results/t74_lam{110,125,140}_L16_*.csv` by name (no glob) with `analyse_t74.read_cells`,
the stage-1 reader, unchanged. Verdicts, as Amendment 6 writes them:

* the size verdict (the registered claim, scored alone): HOLDS AT L = 16 if all 18 pooled fed-phase periods (the
  three backgrounds pooled, as stage 1's timing read pools them) lie in their ranges; MISSES AT L = 16 otherwise,
  naming each period outside;
* three secondary reads, each REPEATS STAGE 1 or CHANGES: fed (the 48 listed cells BURPS, and none of the six
  lambda = 1.10, c = 2R meal cells NEVER BURPS); timing (the 12 words as written); unfed (the 33 listed cells give their
  word, and the count of the other 21 that give their most likely word lies in the written range);
* NOT READ for any number whose cell is missing or has other than 16 replicas, with the reason.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyse_t74 as A          # noqa: E402
import make_t74_configs as M     # noqa: E402


def setting(name):
    """(lam, feed, cap) of a cell name: t74_lam110_L16_meals16_c2R_cold -> ('1.10', 'meals16', 'c2R')."""
    parts = name.split("_")
    return "%.2f" % (int(parts[1][3:]) / 100), parts[3], parts[4]


def score(pred, cells):
    """pred: the prediction file's dict; cells: {config name: analyse_t74 cell dict}. Returns a dict of verdicts."""
    n_rep = int(pred["replicas"])
    names = [c["name"] for c in M.held_out_cells()]
    missing = [n for n in names if n not in cells or cells[n]["n"] != n_rep]
    out = dict(not_read=missing)

    pooled = defaultdict(list)
    complete = defaultdict(lambda: True)
    for n in names:
        key = setting(n)
        if n in missing:
            complete[key] = False
        else:
            pooled[key].extend(cells[n]["periods"])
    periods, misses = [], []
    for p in pred["periods"]:
        key = (p["lam"], p["feed"], p["cap"])
        if not complete[key] or not pooled[key]:
            periods.append(dict(p, value=None, inside=None))
            continue
        value = sum(pooled[key]) / len(pooled[key])
        inside = p["low"] <= value <= p["high"]
        periods.append(dict(p, value=value, inside=inside))
        if not inside:
            misses.append(key)
    if any(x["value"] is None for x in periods):
        out["size"] = "NOT READ"
    else:
        out["size"] = "HOLDS AT L = 16" if not misses else "MISSES AT L = 16"
    out["periods"], out["misses"] = periods, misses

    def word_of(n, kind):
        return None if n in missing else cells[n][kind]

    fed_bad = [n for n in pred["fed"]["must_burps"] if word_of(n, "fed_word") != "BURPS"]
    fed_bad += [n for n in pred["fed"]["six"] if word_of(n, "fed_word") in (None, "NEVER BURPS")]
    out["fed"] = ("REPEATS STAGE 1" if not fed_bad else "CHANGES", fed_bad)

    tim = {}
    for lam, feed, cap, word in pred["timing"]:
        group = [n for n in names if setting(n) in ((lam, feed, cap), (lam, "steady", cap))]
        if any(n in missing for n in group):
            tim[(lam, feed, cap)] = (word, None)
            continue
        meals = [p for n in group if setting(n)[1] == feed for p in cells[n]["periods"]]
        steady = [p for n in group if setting(n)[1] == "steady" for p in cells[n]["periods"]]
        m, cv = A.pooled(meals)
        ms, cvs = A.pooled(steady)
        # as analyse_t74.timing: a steady feed with no spread gives an infinite ratio if the meals spread, else none
        ratio = m / ms
        cv_ratio = (cv / cvs) if cvs else (float("inf") if cv else None)
        got = ("DIFFERENT TIMING" if abs(ratio - 1) > A.TIMING_TOL else
               "SPREAD ONLY" if cv_ratio is not None and cv_ratio >= A.SPREAD_FACTOR else "SAME TIMING")
        tim[(lam, feed, cap)] = (word, got, ratio, cv_ratio)
    tim_bad = [k for k, v in tim.items() if v[1] is not None and v[1] != v[0]]
    out["timing"] = ("NOT READ" if any(v[1] is None for v in tim.values()) else
                     "REPEATS STAGE 1" if not tim_bad else "CHANGES", tim_bad, tim)

    rep_bad = [n for n, w in pred["unfed"]["must_repeat"].items() if word_of(n, "unfed_word") != w]
    others = pred["unfed"]["others"]
    n_modal = sum(word_of(n, "unfed_word") == w for n, w in others.items())
    lo, hi = pred["unfed"]["others_range"]
    out["unfed"] = ("REPEATS STAGE 1" if not rep_bad and lo <= n_modal <= hi else "CHANGES", rep_bad, n_modal)
    return out


def main(path=ROOT / "configs" / "t74_heldout_prediction.json"):
    pred = json.loads(Path(path).read_text())
    rows = []
    for c in M.held_out_cells():
        p = ROOT / "results" / (c["name"] + ".csv")
        if p.exists():
            rows.extend(csv.DictReader(open(p, newline="")))
    by_key = A.read_cells(rows) if rows else {}
    cells = {}
    for (lam, L, feed, cap, bg), cell in by_key.items():
        cells["t74_lam%d_L%02d_%s_%s_%s" % (round(lam * 100), L, feed, cap, bg)] = cell
    out = score(pred, cells)
    print("size verdict:", out["size"])
    for p in out["periods"]:
        print("  lam %s %-8s %-6s | fit %.3f (%.3f to %.3f) | L = 16 %s %s"
              % (p["lam"], p["feed"], p["cap"], p["fit"], p["low"], p["high"],
                 "%.3f" % p["value"] if p["value"] is not None else "-",
                 {True: "inside", False: "OUTSIDE", None: "NOT READ"}[p["inside"]]))
    print("fed read:", out["fed"][0], out["fed"][1] or "")
    print("timing read:", out["timing"][0], out["timing"][1] or "")
    print("unfed read:", out["unfed"][0], out["unfed"][1] or "", "| other 21 on their most likely word:",
          out["unfed"][2])
    if out["not_read"]:
        print("NOT READ (missing, or not 16 replicas):", ", ".join(out["not_read"]))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:2])
