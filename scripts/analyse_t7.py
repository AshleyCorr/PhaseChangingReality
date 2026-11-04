"""The T7 verdict table. Usage:

    python scripts/analyse_t7.py lam125             (reads results/t7_lam125_n*.csv)
    python scripts/analyse_t7.py lam125 t7b         (the rerun under amendment 1)
    python scripts/analyse_t7.py lam125 t7b t7c     (amendment 3: t7b with its replayed decays swapped in)
    python scripts/analyse_t7.py lam125 t7b t7c t7d (amendment 4 (a): decays whose resting state was
                                                     read from its wiring pass gate 3 when their
                                                     energy matches it; scripts/analyse_t7_states.py)

Applies PREREGISTRATION.md section T7 exactly: gates 2 and 3, the three predictions, and the
three verdicts. With a replay prefix, rows of the replay files replace the base rows with the
same (N, replica); a replay that does not reproduce its original decay up to the old cap
(identical `waiting` and `released_first_window`) is not the same decay, is left out, and is
reported. Prints and writes nothing.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_DECAYS = 30            # gate 2
ENERGY_TOL = 0.01          # gate 3, fractional
CV_RANGE = (0.7, 1.3)      # prediction (a)
TWO_STATE_FRAC = 0.80      # prediction (b): share of vertices at d in {1, 2} at half conversion
FRONT_FRAC = 0.70          # prediction (c): largest sheet piece's share of converted vertices
CONT_CV, CONT_FRAC = 0.4, 0.5


def load(prefix, tag, out_dir="results"):
    """Rows by N from every results/<prefix>_<tag>_n*.csv (N read from the rows, not the name)."""
    by_n = defaultdict(list)
    for f in Path(out_dir).glob("%s_%s_n*.csv" % (prefix, tag)):
        if not re.search(r"_n\d+", f.stem):
            continue
        for r in csv.DictReader(open(f, newline="")):
            by_n[int(r["N"])].append(r)
    return by_n


def same_decay(a, b):
    """A replay reproduces its original up to the old cap: same wait, same first-window release."""
    try:
        return a["waiting"] == b["waiting"] and abs(float(a["released_first_window"]) - float(b["released_first_window"])) < 1e-9
    except (KeyError, ValueError):
        return False


def merge_replays(base, replay):
    """Replay rows replace base rows with the same (N, replica). Returns (merged, replaced, rejected),
    the last two as {N: [replica, ...]}; a rejected replay is one that is not the same decay."""
    merged, replaced, rejected = {}, defaultdict(list), defaultdict(list)
    for n, rows in base.items():
        rep = {int(r["replica"]): r for r in replay.get(n, [])}
        out = []
        for r in rows:
            k = int(r["replica"])
            if k in rep:
                if same_decay(r, rep[k]):
                    out.append(rep[k]); replaced[n].append(k)
                else:
                    out.append(r); rejected[n].append(k)
            else:
                out.append(r)
        merged[n] = out
    return merged, dict(replaced), dict(rejected)


def main(tag, out_dir="results", prefix="t7", replay_prefix=None, identified_prefix=None):
    identified = {}
    if identified_prefix:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import analyse_t7_states                                  # noqa: E402
        identified = analyse_t7_states.check_all(out_dir, identified_prefix, tag)
    by_n = load(prefix, tag, out_dir)
    if not by_n:
        print("no results for", tag); return
    print("T7, %s, %s%s. PREREGISTRATION.md section T7.\n" % (tag, prefix, (" with %s replays" % replay_prefix) if replay_prefix else ""))
    if replay_prefix:
        by_n, replaced, rejected = merge_replays(by_n, load(replay_prefix, tag, out_dir))
        for n in sorted(by_n):
            print("   N=%-4d replayed replicas %s%s" % (n, sorted(replaced.get(n, [])),
                  ("   NOT THE SAME DECAY, left as run: %s" % sorted(rejected[n])) if n in rejected else ""))
        print()
    print("%-5s %-7s %-7s %-6s %-9s %-7s %-14s %-12s %-9s  %s"
          % ("N", "decays", "gate2", "gate3", "wait", "CV", "d in {1,2} @50%", "largest @50%", "pieces", "a b c"))
    per_size = []
    for n in sorted(by_n):
        rows = [r for r in by_n[n] if r["reached"] and float(r["reached"]) >= 0.75]
        if not rows:
            print("%-5d 0" % n); continue
        lam = float(rows[0]["lam"])
        expect = 4.0 * (lam - 1.0)
        ring = (24.0 * lam - 16.0) / n           # the four-point remnant, per point (14/N at 1.25)
        rel = np.array([float(r["released"]) for r in rows])
        g2 = len(rows) >= MIN_DECAYS
        # gate 3 under amendment 2 (enacted 2026-09-22): the energy is checked at whichever state the
        # decay reached -- the full sheet, or the ledge that is the four-point remnant
        at_sheet = np.abs(rel - expect) <= ENERGY_TOL * expect
        at_ring = np.abs(rel - (expect - ring)) <= ENERGY_TOL * expect
        # gate 3 under amendment 4 (a): a decay whose resting state was read from its wiring passes
        # exactly when its recorded release matches the energy of that structure
        read = np.array([(n, int(r["replica"])) in identified and identified[(n, int(r["replica"]))]["ok"]
                         for r in rows], dtype=bool)
        at_ring = at_ring & ~read
        at_sheet = at_sheet & ~read
        g3 = bool(np.all(at_sheet | at_ring | read))
        waits = np.array([float(r["waiting"]) for r in rows if r["waiting"]])
        cv = waits.std(ddof=1) / waits.mean() if len(waits) > 1 else np.nan
        two = np.array([(int(r["d1_50"]) + int(r["d2_50"])) / n for r in rows if r["d2_50"] != ""])
        largest = np.array([float(r["largest_50"]) for r in rows if r["largest_50"] != ""])
        pieces = np.array([int(r["pieces_50"]) for r in rows if r["pieces_50"] != ""])
        a = CV_RANGE[0] <= cv <= CV_RANGE[1]
        b = two.mean() >= TWO_STATE_FRAC
        c = largest.mean() >= FRONT_FRAC
        print("%-5d %-7d %-7s %-6s %-9.0f %-7.2f %-14s %-12s %-9s  %s %s %s"
              % (n, len(rows), "pass" if g2 else "FAIL", "pass" if g3 else "FAIL", waits.mean(), cv,
                 "%.3f +/- %.3f" % (two.mean(), two.std(ddof=1) if len(two) > 1 else 0),
                 "%.2f +/- %.2f" % (largest.mean(), largest.std(ddof=1) if len(largest) > 1 else 0),
                 "%.1f" % pieces.mean(),
                 "Y" if a else "n", "Y" if b else "n", "Y" if c else "n"))
        neither = ~(at_sheet | at_ring | read)
        print("      gate 3: %d at the sheet (%.3f), %d on the four-point ledge (%.3f), %s%d at neither%s"
              % (at_sheet.sum(), expect, at_ring.sum(), expect - ring,
                 ("%d read from their wiring and matching, " % read.sum()) if identified else "",
                 neither.sum(),
                 "" if g3 else " -> released " + ", ".join("%.3f" % v for v in sorted(rel[neither]))))
        if "settle_sweeps" in rows[0]:
            caps = sorted(int(r["replica"]) for r in rows if r["settle_sweeps"] and float(r["settle_sweeps"]) >= 100000)
            unsettled = sorted(int(r["replica"]) for r in rows if r["settle_sweeps"] and float(r["settle_sweeps"]) >= 30000 and int(r["replica"]) not in caps)
            if caps or unsettled:
                print("      settle cap reached: at 30,000 by %s; at 100,000 by %s" % (unsettled or "none", caps or "none"))
        if "ledge_sweeps" in rows[0]:
            led = np.array([float(r["ledge_sweeps"]) for r in rows])
            fw = np.array([float(r["released_first_window"]) for r in rows])
            on = led > 0
            print("      ledge: %d of %d decays paused on the way down (first-window release %.2f of %.2f), "
                  "for %.0f to %.0f sweeps" % (on.sum(), len(rows), fw[on].mean() if on.any() else float('nan'),
                                               expect, led[on].min() if on.any() else 0, led[on].max() if on.any() else 0))
        per_size.append(dict(N=n, g=g2 and g3, a=a, b=b, c=c, cv=cv, two=two.mean()))

    gated = [p for p in per_size if p["g"]]
    print("\nSizes passing gates 2 and 3: %s of %s" % ([p["N"] for p in gated], [p["N"] for p in per_size]))
    if len(gated) < len(per_size):
        print("The verdict is over the sizes run; a size failing a gate is reported and not interpreted.")
    if not gated:
        print("No verdict."); return
    if all(p["a"] and p["b"] and p["c"] for p in gated):
        verdict = "TWO-STATE CHANGE"
    elif all(p["cv"] < CONT_CV and p["two"] < CONT_FRAC for p in gated):
        verdict = "CONTINUOUS DEFORMATION"
    else:
        verdict = "INCONCLUSIVE"
    print("\n  PRE-REGISTERED VERDICT, %s, sizes %s:  %s" % (tag, [p["N"] for p in gated], verdict))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lam125",
         prefix=(sys.argv[2] if len(sys.argv) > 2 else "t7"),
         replay_prefix=(sys.argv[3] if len(sys.argv) > 3 else None),
         identified_prefix=(sys.argv[4] if len(sys.argv) > 4 else None))
