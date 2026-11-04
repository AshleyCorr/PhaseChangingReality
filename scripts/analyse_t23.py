"""The pre-registered T23 reading: the lambda map again, with the memoryless check sized to the sample. Usage:

    python scripts/analyse_t23.py

Implements PREREGISTRATION.md section T23 (written 2026-09-24, before the runs) on results/t23_lam<tag>_n<N>.csv
and their _adj directories. Everything is T8's (scripts/analyse_t8.py) except criterion (a), which becomes (a'):
the coefficient of variation of the time after the 200-sweep rest, r = waiting - 200, must lie inside the central
99.9 % band of the CV of n independent exponential draws, n the number of decays read, computed from 200,000
simulated samples with seed 20262399. Two verdicts: the window (lambda = 1.05 to 1.30) and the edge (1.35 against
1.30). Pure functions of parsed rows, tested in tests/test_t23.py.
"""
import csv
import glob
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t8                                                 # noqa: E402

REST = 200
BAND_DRAWS = 200_000
BAND_SEED = 20262399
BAND_P = 0.999
WINDOW = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30)
EDGE, BELOW_EDGE = 1.35, 1.30


@lru_cache(maxsize=None)
def cv_band(n):
    """Central 99.9 % range of the sample CV (ddof 1) of n independent exponential draws."""
    rng = np.random.default_rng(BAND_SEED)
    x = rng.exponential(1.0, size=(BAND_DRAWS, n))
    cv = x.std(axis=1, ddof=1) / x.mean(axis=1)
    tail = (1.0 - BAND_P) / 2.0
    lo, hi = np.quantile(cv, [tail, 1.0 - tail])
    return float(lo), float(hi)


def cell(rows, n, lam, read=None):
    """T8's cell, with (a) replaced by (a')."""
    out = analyse_t8.cell(rows, n, lam, read)
    if not out["metastable"]:
        return out
    reached = [r for r in rows if r["reached"] and float(r["reached"]) >= 0.75 and r["waiting"] != ""]
    resid = np.array([float(r["waiting"]) - REST for r in reached])
    out["n_read"] = len(resid)
    if len(resid) < 2 or resid.mean() <= 0:
        out.update(cv=math.nan, band=(math.nan, math.nan), a=False, sharp=False)
        return out
    out["cv"] = float(resid.std(ddof=1) / resid.mean())
    out["band"] = cv_band(len(resid))
    out["a"] = out["band"][0] <= out["cv"] <= out["band"][1]
    out["sharp"] = out["a"] and out["b"] and out["c"]
    return out


def window_verdict(status):
    """status: {lambda: analyse_t8.lam_status} for the window lambdas."""
    missing = [l for l in WINDOW if l not in status]
    if missing:
        return "INCONCLUSIVE (no data at %s)" % ", ".join("%.2f" % l for l in missing)
    values = [status[l] for l in WINDOW]
    if all(v == "sharp" for v in values):
        return "SHARP ACROSS THE WINDOW"
    if all(v in ("sharp", "not sharp") for v in values):
        return "NOT SHARP AT " + ", ".join("%.2f" % l for l in WINDOW if status[l] == "not sharp")
    return "INCONCLUSIVE (" + ", ".join("%.2f %s" % (l, status[l]) for l in WINDOW if status[l] not in ("sharp", "not sharp")) + ")"


def share_diff(k1, n1, k2, n2):
    """(p2 - p1, standard error of the difference) for two proportions."""
    p1, p2 = k1 / n1, k2 / n2
    return p2 - p1, math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def edge_report(below, edge):
    """below, edge: lists of (row, is_flat) pooled over sizes at 1.30 and 1.35. Returns the report dict."""
    flat_b = sum(1 for _, f in below if f); flat_e = sum(1 for _, f in edge if f)
    d1, se1 = share_diff(flat_b, len(below), flat_e, len(edge))
    multi = lambda rs: [r for r, _ in rs if r.get("pieces_25", "") != ""]
    mb, me = multi(below), multi(edge)
    k_b = sum(1 for r in mb if int(r["pieces_25"]) > 1); k_e = sum(1 for r in me if int(r["pieces_25"]) > 1)
    d2, se2 = share_diff(k_b, len(mb), k_e, len(me)) if mb and me else (math.nan, math.nan)
    e1 = d1 < -2 * se1
    e2 = (not math.isnan(d2)) and d2 > 2 * se2
    verdict = "BREAK-UP BEGINS AT THE EDGE" if e1 and e2 else ("NO BREAK-UP" if not (e1 or e2) else "PARTIAL")
    return dict(flat=(flat_b / len(below), flat_e / len(edge)), d1=d1, se1=se1, e1=e1,
                multi=(k_b / len(mb) if mb else math.nan, k_e / len(me) if me else math.nan), d2=d2, se2=se2, e2=e2,
                verdict=verdict)


def load(out_dir="results"):
    """{lambda: {N: (rows, adj_dir)}} from the committed T23 files."""
    data = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(out_dir) / "t23_lam*_n*.csv"))):
        m = re.search(r"t23_lam(\d+)_n(\d+)\.csv$", f)
        lam, n = int(m.group(1)) / 100.0, int(m.group(2))
        rows = list(csv.DictReader(open(f, newline="")))
        data[lam][n] = (rows, Path(f).with_suffix("").as_posix() + "_adj")
    return data


def main(out_dir="results"):
    data = load(out_dir)
    status, pooled = {}, {}
    for lam in sorted(data):
        cells = []
        pooled[lam] = []
        for n in sorted(data[lam]):
            rows, adj_dir = data[lam][n]
            read = analyse_t8.read_from_wiring(adj_dir, lam)
            c = cell(rows, n, lam, read)
            cells.append(c)
            pooled[lam] += [(r, analyse_t8.classify_release(r, n, lam, read) == "sheet") for r in rows]
            if not c["metastable"]:
                print("lambda=%.2f N=%-4d NOT METASTABLE (median f_200 %.2f)" % (lam, n, c["f200_median"]))
                continue
            print("lambda=%.2f N=%-4d gate2 %s gate3 %s | read %d, CV of wait-200 %.3f in [%.3f, %.3f] %s | "
                  "d in {1,2} %.3f | largest %.2f | sharp %s | wait %.0f vs tau %.0f"
                  % (lam, n, "pass" if c["gate2"] else "FAIL", "pass" if c["gate3"] else "FAIL", c.get("n_read", 0),
                     c["cv"], *c.get("band", (math.nan, math.nan)), "Y" if c["a"] else "n", c["two"], c["largest"],
                     "Y" if c["sharp"] else "n", c["wait"], analyse_t8.tau(lam)))
        status[lam] = analyse_t8.lam_status(cells)
    print("\nWINDOW (1.05 to 1.30):", window_verdict({l: s for l, s in status.items() if l in WINDOW}))
    if EDGE in pooled and BELOW_EDGE in pooled:
        e = edge_report(pooled[BELOW_EDGE], pooled[EDGE])
        print("EDGE (1.35 against 1.30): ending flat %.2f -> %.2f (E1 %s); converted region in more than one piece "
              "at 25%% %.2f -> %.2f (E2 %s): %s"
              % (*e["flat"], "yes" if e["e1"] else "no", *e["multi"], "yes" if e["e2"] else "no", e["verdict"]))
    else:
        print("EDGE: no data at 1.30 and 1.35")


if __name__ == "__main__":
    main(*sys.argv[1:])
