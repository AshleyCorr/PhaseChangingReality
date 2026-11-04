"""T50: does the tube's opening front slow where energy sits? Read by the rules written before the runs. Usage:

    python scripts/analyse_t50.py

PREREGISTRATION T50 (written 2026-09-27, before any run). Reads `results/t50_*.csv` (scripts/run_front_matter.py).

Per replica: t_R(k) and t_L(k), the first reading at which at least k columns have opened on the right and on the left of
the seed. The band spans distances k_in = centre - width/2 to k_out = centre + width/2 on the right. The crossing times are
T_band = t_R(k_out) - t_R(k_in) and T_mirror = t_L(k_out) - t_L(k_in), the same distances on the left, through bare tube;
the replica's ratio is R = T_band / T_mirror. A replica is valid if both fronts reached k_out. Per config (energy per
point): the number valid and the median R. Per energy e > 0, with Q = median R(e) / median R(0): **SLOWS** if Q >= 1.25,
**SPEEDS** if Q <= 0.8, **NO EFFECT** otherwise, and **NO FRONT** if fewer than half the replicas of that config or of the
control are valid. Reported beside, not scored: the median crossing times themselves, and the band's energy left when
the right front leaves it.
"""
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def first(blocks, key, k):
    return next((int(b["sweep"]) for b in blocks if int(b[key]) >= k), None)


def read_replica(blocks):
    """(R, T_band, T_mirror, band energy when the right front reaches k_out), or None if a front did not get there."""
    c, w = int(blocks[0]["band_centre"]), int(blocks[0]["band_width"])
    k_in, k_out = c - w // 2, c + w // 2
    times = [first(blocks, side, k) for side in ("right", "left") for k in (k_in, k_out)]
    if None in times:
        return None
    t_band, t_mirror = times[1] - times[0], times[3] - times[2]
    left_in_band = next(float(b["band_energy"]) for b in blocks if int(b["sweep"]) == times[1])
    return (t_band / t_mirror if t_mirror > 0 else float("inf")), t_band, t_mirror, left_in_band


def summarize(rows):
    runs = defaultdict(list)
    for r in rows:
        runs[(float(r["energy_per_point"]), int(r["replica"]))].append(r)
    per = defaultdict(list)
    counts = defaultdict(int)
    for (e, rep), blocks in runs.items():
        blocks.sort(key=lambda b: int(b["sweep"]))
        counts[e] += 1
        got = read_replica(blocks)
        if got is not None:
            per[e].append(got)
    table = {}
    for e in sorted(counts):
        vals = per[e]
        table[e] = dict(n=counts[e], valid=len(vals),
                        median_R=statistics.median(v[0] for v in vals) if vals else None,
                        median_band=statistics.median(v[1] for v in vals) if vals else None,
                        median_mirror=statistics.median(v[2] for v in vals) if vals else None,
                        band_energy_left=statistics.median(v[3] for v in vals) if vals else None)
    verdict = {}
    ctrl = table.get(0.0)
    for e, t in table.items():
        if e == 0.0:
            continue
        if ctrl is None or t["valid"] < t["n"] / 2 or ctrl["valid"] < ctrl["n"] / 2:
            verdict[e] = "NO FRONT"
            continue
        q = t["median_R"] / ctrl["median_R"]
        verdict[e] = "SLOWS" if q >= 1.25 else ("SPEEDS" if q <= 0.8 else "NO EFFECT")
        t["Q"] = q
    return dict(table=table, verdict=verdict)


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t50_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T50 results")
        return
    s = summarize(rows)
    for e, t in sorted(s["table"].items()):
        print("energy per point %-5g %s" % (e, t))
    for e, v in sorted(s["verdict"].items()):
        print("VERDICT T50 energy per point %g: %s" % (e, v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
