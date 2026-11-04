"""The pre-registered T30 and T32 readings: the six-link curled torus, sealed. Usage:

    python scripts/analyse_t30.py

Implements PREREGISTRATION.md sections T30 (does one push open both curled directions; the room) and T32 (is the
activation fixed with size), written 2026-09-25 before the runs, on results/t30_*.csv and results/t32_*.csv.

T30, per replica (a 4 x 4 x L torus, two directions curled, H0/N = 8(lambda - 1)):
  the middle rung is reached at the first block with H/N <= 4(lambda - 1)(1 + TOL) and at least half the vertices
  at d >= 2; the flat state is reached at the first block with H/N <= 4(lambda - 1) TOL and at least CLEAN of the
  vertices at d = 3. A replica RESTS on the middle rung if H/N stays within +-TOL of 4(lambda - 1) for at least
  REST consecutive sweeps. Outcomes, from the final block: FLAT (reached flat), MIDDLE (reached the middle rung,
  not flat), STUCK (neither, majority of vertices still at d = 1 and fewer than MELT_FRAC melted), MELTED
  (at least MELT_FRAC of vertices at d > 3), OTHER.
  Per cell (lambda, N, C): the majority outcome. Per (lambda, N): C*, the smallest C whose majority is FLAT.
  Mechanism, over every replica anywhere that reached the flat state: CASCADE if it did not rest on the middle
  rung on the way; STEPWISE if it did.
  Verdicts: ALL AT ONCE if some cell has a FLAT majority and at least half of the flat-reaching replicas
  cascaded; ONE AT A TIME if some cell has a FLAT majority and fewer than half cascaded; FIRST ONLY if no cell has
  a FLAT majority but some cell has a MIDDLE majority; NEVER OPENS if no cell has a FLAT or MIDDLE majority (the
  spark bought nothing, or everything melted).

T32, per (N, spark): the share of replicas that ever left the start (S or X changed). The threshold E* at each N
is the smallest spark at which every replica left. FIXED WALL if E* is the same at every N; GROWS if E* rises
with N; FALLS if it falls; NOT READ if some N has no spark at which every replica left. SHARP is reported beside
it: at every N, nothing below E* ever left.
"""
import csv
import glob
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOL, CLEAN, REST, MELT_FRAC = 0.10, 0.90, 5000, 0.25


def by_replica(rows):
    out = defaultdict(list)
    for r in rows:
        out[(float(r["lam"]), int(r["N"]), int(r["C"]), float(r["spark"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def rung_times(blocks, lam, n):
    """(sweep the middle rung was first reached, sweep the flat state was first reached, rested on the middle rung)"""
    one = 4.0 * (lam - 1.0)
    t_mid = t_flat = None
    rest_start = None
    rested = False
    for b in blocks:
        h = float(b["h_per_vertex"])
        d3 = int(b["d3"])
        d_ge2 = sum(int(b["d%d" % k]) for k in range(2, 8))
        sw = int(b["sweep"])
        if t_mid is None and h <= one * (1 + TOL) and d_ge2 >= n / 2:
            t_mid = sw
        if t_flat is None and h <= one * TOL and d3 >= CLEAN * n:
            t_flat = sw
        on_rung = abs(h - one) <= TOL * one
        if on_rung:
            if rest_start is None:
                rest_start = sw
            elif sw - rest_start >= REST:
                rested = True
        else:
            rest_start = None
    return t_mid, t_flat, rested


def outcome(blocks, lam, n):
    t_mid, t_flat, rested = rung_times(blocks, lam, n)
    last = blocks[-1]
    melted = int(last["melted"])
    if melted >= MELT_FRAC * n:
        return "MELTED", rested
    if t_flat is not None and int(last["d3"]) >= CLEAN * n:
        return "FLAT", rested
    if t_mid is not None:
        return "MIDDLE", rested
    if int(last["d1"]) > n / 2:
        return "STUCK", rested
    return "OTHER", rested


def read_t30(rows):
    reps = by_replica(rows)
    cells = defaultdict(list)
    cascaded, stepwise = 0, 0
    for (lam, n, c, spark, rep), blocks in reps.items():
        o, rested = outcome(blocks, lam, n)
        cells[(lam, n, c)].append(o)
        if o == "FLAT":
            if rested:
                stepwise += 1
            else:
                cascaded += 1
    majority = {}
    for k, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        majority[k] = top if votes > len(outs) / 2 else "MIXED"
    c_star = {}
    for (lam, n, c), m in majority.items():
        if m == "FLAT":
            c_star[(lam, n)] = min(c, c_star.get((lam, n), c))
    return dict(cells={k: dict(Counter(v)) for k, v in cells.items()}, majority=majority, c_star=c_star,
                cascaded=cascaded, stepwise=stepwise)


def verdict_t30(summary):
    m = summary["majority"].values()
    flat_reaching = summary["cascaded"] + summary["stepwise"]
    if any(v == "FLAT" for v in m):
        return "ALL AT ONCE" if summary["cascaded"] >= flat_reaching / 2 else "ONE AT A TIME"
    if any(v == "MIDDLE" for v in m):
        return "FIRST ONLY"
    return "NEVER OPENS"


def read_t32(rows):
    """{N: {spark: share that left}}"""
    reps = by_replica(rows)
    left = defaultdict(lambda: defaultdict(list))
    for (lam, n, c, spark, rep), blocks in reps.items():
        left[n][spark].append(blocks[-1]["left"] == "True")
    return {n: {e: sum(v) / len(v) for e, v in d.items()} for n, d in left.items()}


def verdict_t32(shares):
    e_star = {}
    sharp = True
    for n, d in shares.items():
        full = [e for e, p in d.items() if p >= 1.0]
        if not full:
            return "NOT READ", e_star, sharp
        e_star[n] = min(full)
        if any(p > 0 for e, p in d.items() if e < e_star[n]):
            sharp = False
    ns = sorted(e_star)
    vals = [e_star[n] for n in ns]
    if all(v == vals[0] for v in vals):
        return "FIXED WALL", e_star, sharp
    if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
        return "GROWS", e_star, sharp
    if all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        return "FALLS", e_star, sharp
    return "NOT MONOTONE", e_star, sharp


def is_gas(row):
    """A gas of separate 6-cubes: the runner labels its dims 'k x(a b c)'."""
    return "x(" in row["dims"]


def main(out_dir="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "t30_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="")))
    print("== T30")
    if rows:
        # The pre-registration reads the gas "by the same rules" and reports its verdict "separately as T30-gas".
        # Until 2026-09-25 this script pooled the gas with the tori, which let the gas cell decide the torus verdict
        # (found by a referee reading; ASSUMPTIONS O64). The two are now read apart, as registered.
        for label, subset in (("T30 (tori)", [r for r in rows if not is_gas(r)]), ("T30-gas", [r for r in rows if is_gas(r)])):
            if not subset:
                continue
            s = read_t30(subset)
            for k in sorted(s["cells"]):
                print("  lambda=%.2f N=%-4d C=%-5d %s -> %s" % (*k, s["cells"][k], s["majority"][k]))
            for k, c in sorted(s["c_star"].items()):
                print("  C* (smallest bath with a FLAT majority) at lambda=%.2f N=%d: %d" % (*k, c))
            print("  flat-reaching replicas: cascaded %d, stepwise %d" % (s["cascaded"], s["stepwise"]))
            print("VERDICT %s: %s" % (label, verdict_t30(s)))
    else:
        print("  no results yet")
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "t32_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="")))
    print("\n== T32")
    if rows:
        sh = read_t32(rows)
        for n in sorted(sh):
            print("  N=%-4d " % n + "  ".join("E=%g: %.2f" % (e, sh[n][e]) for e in sorted(sh[n])))
        v, e_star, sharp = verdict_t32(sh)
        print("  E* per N:", {n: e_star[n] for n in sorted(e_star)}, "| sharp:", sharp)
        print("VERDICT T32:", v)
    else:
        print("  no results yet")


if __name__ == "__main__":
    main(*sys.argv[1:])
