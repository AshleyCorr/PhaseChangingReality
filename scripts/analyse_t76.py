"""T76's verdicts, read from the runner's rows (scripts/run_t76.py). Usage, naming every run read:

    python scripts/analyse_t76.py [--window W] results/t76_l16_g3p2_q16.csv results/t76_l24_g3p2_q16.csv ...

Cells are (g, L, q), q the chance a clear offer is taken (Amendment 2; rows written before q was recorded read as
q = 1). --window sets the settling window for rho_eq (100 as registered; 1,000 under Amendment 4).

Implements the verdict words of PREREGISTRATION.md T76 and its translation to w. Per replica, from its readings:

  N0        the number of points when growth starts (the last settling reading);
  N_stop    the number at the last reading (the stop);
  rho_eq    the mean of rho = D / N over the settling readings of the last `window` (100) fair sweeps;
  rho_stop  rho at the last reading;
  doubled   N_stop >= 2 N0;
  second half (by points added): the growth readings with N - N0 >= (N_stop - N0) / 2;
  constant  doubled, the second half holds at least MIN_HALF (2) readings, and every rho of it lies within 10 % of
            the mean of those rhos (|rho - m| <= 0.1 m; literally, so a second half with rho = 0 throughout counts as
            constant: T76 avoids such couplings). MIN_HALF is ours, a reader's choice to be confirmed by amendment:
            one reading cannot "stay" anywhere, and on a clean sheet one growth step can multiply N by up to 9 (all L
            boundaries of an axis take a pair each, so L triples), reaching the stop within the first growth fair
            sweep and leaving a single reading in the second half;
  falling   doubled and rho_stop < 0.9 rho_eq;
  rising    doubled and rho_stop > 1.1 rho_eq;
  stalls    not doubled (N grew by less than a factor of two before the stop);
  p         the least-squares slope of ln rho against ln N over the second half's readings with rho > 0 (NaN when they
            span fewer than two values of N).

Per (g, L), 16 replicas: the first word whose count reaches 12, in the registered order CONSTANT, FALLING, RISING,
STALLS; else MIXED. STALLS excludes the other three (they need N to double), and FALLING excludes RISING, but CONSTANT
can be met together with FALLING or RISING (a second half that is steady at a level away from rho_eq); T76 does not
say which word wins then, so the registered order is used and every other word that also reached 12 is printed beside
it (ours, a reader's choice, to be confirmed by amendment before any cell is read).

The translation to w (T76, stated before the run): rho ~ A^p with A = N, the toy's volume, gives w = -1 - p. The
cell's p is the median of its replicas' p (their spread printed beside it); a pooled fit over all the cell's
second-half readings is printed as well. The toy is two-dimensional, so the size of a departure from w = -1 carries
the dimension caveat; a constant density is w = -1 in any dimension.

Nothing random; pure functions tested in tests/test_analyse_t76.py.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

QUORUM, WINDOW, MIN_HALF = 12, 100, 2
STEADY, LOW, HIGH = 0.10, 0.90, 1.10
WORDS = ("CONSTANT", "FALLING", "RISING", "STALLS")
NUMERIC = ("g", "lam", "q", "fair_sweep", "N", "Lx", "Ly", "S", "X", "D", "rho", "curled_share", "rows_added", "attempts")


def rho_eq(records, n_settle=None, window=WINDOW):
    """The mean rho over the settling readings of the last `window` fair sweeps (fair_sweep > n_settle - window).
    n_settle defaults to the last settling reading's fair sweep."""
    settle = [r for r in records if r["phase"] == "settle"]
    if not settle:
        raise ValueError("no settling readings")
    if n_settle is None:
        n_settle = max(r["fair_sweep"] for r in settle)
    if window > n_settle:
        raise ValueError("the window is longer than the settling")
    win = [r["rho"] for r in settle if r["fair_sweep"] > n_settle - window]
    return float(np.mean(win))


def power_fit(n, rho):
    """Slope of ln rho on ln n over the points with rho > 0; NaN if they span fewer than two values of n."""
    n = np.asarray(n, dtype=float)
    rho = np.asarray(rho, dtype=float)
    keep = rho > 0
    if np.unique(n[keep]).size < 2:
        return float("nan")
    return float(np.polyfit(np.log(n[keep]), np.log(rho[keep]), 1)[0])


def second_half(records, n0, n_stop):
    grow = sorted((r for r in records if r["phase"] == "grow"), key=lambda r: r["fair_sweep"])
    if n_stop <= n0:
        return []
    return [r for r in grow if r["N"] - n0 >= (n_stop - n0) / 2]


def replica_reading(records, window=WINDOW):
    """The per-replica quantities of the docstring; `records` are one replica's readings as dicts."""
    rec = sorted(records, key=lambda r: r["fair_sweep"])
    before = [r for r in rec if r["phase"] in ("start", "settle")]
    n0 = before[-1]["N"]
    n_stop = rec[-1]["N"]
    eq = rho_eq(rec, None, window)
    half = second_half(rec, n0, n_stop)
    doubled = n_stop >= 2 * n0
    rhos = np.array([r["rho"] for r in half], dtype=float)
    m = float(rhos.mean()) if rhos.size else float("nan")
    steady = rhos.size >= MIN_HALF and bool(np.all(np.abs(rhos - m) <= STEADY * m))
    stop = rec[-1]["rho"]
    return dict(N0=n0, N_stop=n_stop, doubled=doubled, rho_eq=eq, rho_stop=stop, half_mean=m, n_half=len(half),
                constant=doubled and steady, falling=doubled and stop < LOW * eq, rising=doubled and stop > HIGH * eq,
                stalls=not doubled, p=power_fit([r["N"] for r in half], rhos),
                fair_sweeps=rec[-1]["fair_sweep"])


def cell_verdict(readings, quorum=QUORUM):
    """(word, counts per word, other words that also reached the quorum)."""
    counts = {w: sum(1 for r in readings if r[w.lower()]) for w in WORDS}
    met = [w for w in WORDS if counts[w] >= quorum]
    word = met[0] if met else "MIXED"
    return word, counts, met[1:]


def w_of(readings, pooled_records=None):
    """(median p, its quartiles, w = -1 - median p, pooled p) over the replicas with a finite p."""
    ps = np.array([r["p"] for r in readings], dtype=float)
    ps = ps[np.isfinite(ps)]
    if not ps.size:
        return dict(p=float("nan"), q1=float("nan"), q3=float("nan"), w=float("nan"), n=0, pooled_p=float("nan"))
    p = float(np.median(ps))
    pooled = float("nan")
    if pooled_records is not None:
        pooled = power_fit([r["N"] for r in pooled_records], [r["rho"] for r in pooled_records])
    return dict(p=p, q1=float(np.percentile(ps, 25)), q3=float(np.percentile(ps, 75)), w=-1.0 - p, n=int(ps.size),
                pooled_p=pooled)


def read_rows(paths):
    """{(g, L, q): {replica: [readings]}} from the runner's CSVs (q = 1 where a row has none: the registered rate)."""
    cells = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                r = dict(row)
                for k in NUMERIC:
                    if k in r and r[k] != "":
                        r[k] = float(r[k]) if k in ("g", "lam", "q", "rho", "curled_share") else int(r[k])
                cells[(r["g"], int(r["L"]), float(r.get("q") or 1.0))][int(r["replica"])].append(r)
    return cells


def analyse(cells, window=WINDOW):
    """[(g, L, q, n replicas, word, counts, also, w dict, median rho_stop / rho_eq, median N_stop / N0)] per cell."""
    out = []
    for (g, size, q), reps in sorted(cells.items()):
        readings = [replica_reading(recs, window) for _, recs in sorted(reps.items())]
        pooled = []
        for (_, recs), rd in zip(sorted(reps.items()), readings):
            pooled += second_half(recs, rd["N0"], rd["N_stop"])
        word, counts, also = cell_verdict(readings)
        ratio = float(np.median([r["rho_stop"] / r["rho_eq"] if r["rho_eq"] > 0 else np.nan for r in readings]))
        growth = float(np.median([r["N_stop"] / r["N0"] for r in readings]))
        w = w_of(readings, pooled)
        w["thin"] = sum(1 for r in readings if r["n_half"] < MIN_HALF)
        out.append((g, size, q, len(readings), word, counts, also, w, ratio, growth))
    return out


def main(paths, window=WINDOW):
    rows = analyse(read_rows(paths), window)
    print("T76 (PREREGISTRATION.md): per (g, L), quorum %d; w = -1 - p for rho ~ N^p over the second half of growth "
          "(two-dimensional toy: the size of a departure carries the dimension caveat)." % QUORUM)
    for g, size, q, n, word, counts, also, w, ratio, growth in rows:
        flag = "" if n == 16 else "  [%d replicas, 16 registered]" % n
        if w.get("thin"):
            flag += "  [%d replicas with fewer than %d second-half readings]" % (w["thin"], MIN_HALF)
        extra = (" (also %s)" % ", ".join(also)) if also else ""
        print("g = %-6g L = %-3d q = 1/%-5d %-8s%s  counts %s  N_stop/N0 %.2f  rho_stop/rho_eq %.3f  p %.3f [%.3f, %.3f] "
              "(%d fits; pooled %.3f)  w %.3f%s"
              % (g, size, round(1 / q), word, extra, " ".join("%s %d" % (k[:4], v) for k, v in counts.items()), growth, ratio,
                 w["p"], w["q1"], w["q3"], w["n"], w["pooled_p"], w["w"], flag))
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    win = WINDOW
    if args[:1] == ["--window"]:
        win, args = int(args[1]), args[2:]
    if not args:
        sys.exit("name every T76 result read: python scripts/analyse_t76.py [--window W] results/t76_....csv ...")
    main(args, win)
