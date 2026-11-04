"""The T11 verdict table. Usage:

    python scripts/analyse_t11.py

Reads every results/t11_seam_n*.csv and applies PREREGISTRATION.md section T11 as written:
only replicas that end with exactly one ring enter; at each size, SEAM needs at least 60 % of them
at a distance of at least three quarters of the half-length from where the change started, a
median at least that far, and fewer than 30 % within 1.5 columns of the start; AT THE SEED needs
at least 60 % within 1.5 columns. Gate: at least 15 single-ring replicas per size.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

NEAR_FRACTION = 0.75      # of the half-length: "near the far side"
SEED_COLUMNS = 1.5        # "at the start"
SEAM_MIN, SEED_MAX_FOR_SEAM, ATSEED_MIN = 0.60, 0.30, 0.60
MIN_REPLICAS = 15


def load(out_dir="results"):
    rows = []
    for f in sorted(Path(out_dir).glob("t11_seam_n*.csv")):
        rows.extend(csv.DictReader(open(f, newline="")))
    return rows


def circular_mean(cols, lx):
    ang = 2.0 * np.pi * np.asarray(cols, float) / lx
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m * lx / (2.0 * np.pi)) % lx)


def distance_of(r):
    """The circular distance from the start to the leftover, in columns. The runner filled `dist`
    only when the leftover occupied one column; the leftover in fact spans two or three (it lies
    along the tube, not around it), so the distance is taken here from the recorded columns of
    the d = 1 vertices, as the pre-registration's observable says, via their circular mean."""
    half = float(r["half_length"])
    lx = 2.0 * half
    if r.get("dist", "") not in ("", "nan"):
        return float(r["dist"]), half
    cols = [int(c) for c in str(r.get("ring_cols", "")).split()]
    if not cols or r.get("start", "") in ("", "nan"):
        return float("nan"), half
    d = abs(circular_mean(cols, lx) - float(r["start"])) % lx
    return float(min(d, lx - d)), half


def verdict(rows):
    """Returns (verdict, report lines). Pure function of the parsed rows."""
    by_n = defaultdict(list)
    for r in rows:
        if int(r["rings"]) != 1:
            continue
        d, half = distance_of(r)
        if np.isfinite(d):
            by_n[int(r["N"])].append((d, half))
    lines = ["%-5s %-8s %-10s %-10s %-10s  %s" % ("N", "1-ring", "near far", "at start", "median", "gate")]
    seam, atseed, gated = [], [], []
    for n in sorted(by_n):
        d = np.array([x for x, _ in by_n[n]])
        half = by_n[n][0][1]
        near = float(np.mean(d >= NEAR_FRACTION * half))
        seed = float(np.mean(d <= SEED_COLUMNS))
        med = float(np.median(d))
        ok = len(d) >= MIN_REPLICAS
        gated.append(ok)
        seam.append(ok and near >= SEAM_MIN and med >= NEAR_FRACTION * half and seed < SEED_MAX_FOR_SEAM)
        atseed.append(ok and seed >= ATSEED_MIN)
        lines.append("%-5d %-8d %-10.0f %-10.0f %-10s  %s"
                     % (n, len(d), 100 * near, 100 * seed, "%.1f of %.0f" % (med, half), "ok" if ok else "FAILS (< %d)" % MIN_REPLICAS))
    if not by_n or not all(gated):
        return "INCONCLUSIVE", lines + ["\n  gate 2 not met at every size"]
    if all(seam):
        return "SEAM", lines
    if all(atseed):
        return "AT THE SEED", lines
    return "NEITHER", lines


def main(out_dir="results"):
    rows = load(out_dir)
    if not rows:
        print("no T11 results yet"); return
    print("T11, where the leftover ring sits. PREREGISTRATION.md section T11.\n")
    v, lines = verdict(rows)
    print("\n".join(lines))
    print("\n  PRE-REGISTERED VERDICT, T11:  %s" % v)


if __name__ == "__main__":
    main()
