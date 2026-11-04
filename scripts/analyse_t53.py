"""T53: the curled six-link torus (two directions curled, one open) in a warm bath: does it open by itself, and from
one place or several? Read by the rules written before the runs. Usage:

    python scripts/analyse_t53.py [results_dir]

PREREGISTRATION T53 (written 2026-10-05, before any run). Reads `results/t53_*.csv` (scripts/run_curled_bath_d.py).
Every six-link result carries VISION Update 24's caveat: the reproduction gate is open.

Per replica, from its last reading, by T48's rule (`analyse_t48.read_replica`, imported, not rewritten): DAMAGED if at
least a quarter of the points are damaged (more open directions than three); else OPENS if at least half the points
are at d = 3 and the largest connected piece at d = 3 holds at least half of all points; else ADVANCES if the rung
holding the most points is above the starting rung (d = 1); else STAYS. Per cell (L, lambda, g) the majority (more
than half the replicas), else MIXED.

The window, per (L, lambda): **OPENS IN A WINDOW** if at least one g has an OPENS majority; else **ADVANCES ONLY** if
some g has an ADVANCES majority and no g has an OPENS majority; else **DAMAGED** if every g at which anything moved has
a DAMAGED majority; else **STAYS**. How "anything moved" is read here: at that g at least one replica is not STAYS by
the rule above. If nothing moved at any g the verdict is STAYS.

One place or several, for replicas that OPEN or ADVANCE: `open_regions` (the separate connected pieces of at least 16
points at d >= 2) at the first reading at which a tenth of the points are at d >= 2. **SEVERAL** if the median over
such replicas is 2 or more at L = 72 and larger there than at L = 18; **ONE** otherwise. How this is read here: one
verdict, the replicas of every lambda and g at a length taken together; the ordinary median (the mean of the middle
two for an even count); ONE if either length has no such replica. The medians per lambda are printed beside it.

Reported, not scored: per replica the sweep at which half the points first sit at d >= 2 and at d = 3; whether the
opening pauses on the one-curled rung; the final number of connected pieces of the whole graph; the final energy per
point. A pause is defined here, plainly: the rung holding the most points is d = 2 at PAUSE_READINGS (20) or more
readings in a row before d = 3 first holds the most. The words: "pauses" (such a run, then d = 3 takes over),
"straight through" (d = 3 takes over without one), "rests there" (such a run, and d = 3 never takes over), "neither".
"""
import csv
import glob
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t48 import dims_of, hist, read_replica                       # noqa: E402

PAUSE_READINGS = 20      # readings in a row with the most points at d = 2 that count as a pause (ours, fixed here)
LONG, SHORT = 72, 18     # the two lengths the "one place or several" verdict compares
MOVED = ("OPENS", "ADVANCES")


def length_of(row):
    """L: the open side of the 4 x 4 x L torus."""
    return max(dims_of(row))


def replicas(rows):
    """{(L, lambda, g, replica): its readings in order of sweep}."""
    runs = defaultdict(list)
    for r in rows:
        runs[(length_of(r), float(r["lam"]), float(r["g"]), int(r["replica"]))].append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r["sweep"]))
    return runs


def class_of(blocks):
    """DAMAGED, OPENS, ADVANCES or STAYS, from the replica's last reading, by T48's rule."""
    return read_replica(blocks)[0]


def majority_of(classes):
    """The class of more than half the replicas, else MIXED."""
    top, votes = Counter(classes).most_common(1)[0]
    return top if votes > len(classes) / 2 else "MIXED"


def moved_at(cells):
    """The couplings at which anything moved: at least one replica there is not STAYS."""
    return sorted(g for g, classes in cells.items() if any(c != "STAYS" for c in classes))


def window_verdict(cells):
    """The window verdict for one (L, lambda). `cells`: {g: the classes of its replicas}."""
    majorities = {g: majority_of(classes) for g, classes in cells.items()}
    if "OPENS" in majorities.values():
        return "OPENS IN A WINDOW"
    if "ADVANCES" in majorities.values():
        return "ADVANCES ONLY"
    moved = moved_at(cells)
    if moved and all(majorities[g] == "DAMAGED" for g in moved):
        return "DAMAGED"
    return "STAYS"


def first_reading_with(blocks, column, part):
    """The first reading at which at least one part in `part` of the points is counted in `column`, or None."""
    n = int(blocks[0]["N"])
    return next((b for b in blocks if int(b[column]) >= n / part), None)


def regions_at_first_tenth(blocks):
    """`open_regions` at the first reading at which a tenth of the points are at d >= 2, or None if there is none."""
    b = first_reading_with(blocks, "n_open2", 10)
    return None if b is None else int(b["open_regions"])


def several_verdict(regions):
    """SEVERAL or ONE. `regions`: {L: [open_regions at the first tenth, one per replica that OPENS or ADVANCES]}."""
    long_, short = regions.get(LONG, []), regions.get(SHORT, [])
    if not long_ or not short:
        return "ONE"
    at_long, at_short = statistics.median(long_), statistics.median(short)
    return "SEVERAL" if at_long >= 2 and at_long > at_short else "ONE"


def sweep_of(block):
    return None if block is None else int(block["sweep"])


def modal_rung(row):
    """The rung (d from 0 to D) holding the most points; the lower rung on a tie, as in T48's rule."""
    dim = len(dims_of(row))
    rungs, _ = hist(row, dim)
    return max(range(dim + 1), key=lambda k: rungs[k])


def longest_run(values, wanted):
    best = run = 0
    for v in values:
        run = run + 1 if v == wanted else 0
        best = max(best, run)
    return best


def pause_of(blocks, need=PAUSE_READINGS):
    """(word, the longest run of readings with the most points on the one-curled rung before the open rung takes over)."""
    dim = len(dims_of(blocks[0]))
    modal = [modal_rung(b) for b in blocks]
    opened = next((i for i, m in enumerate(modal) if m == dim), None)
    run = longest_run(modal if opened is None else modal[:opened], dim - 1)
    if opened is not None:
        return ("pauses" if run >= need else "straight through"), run
    return ("rests there" if run >= need else "neither"), run


def report_of(blocks):
    """The items reported beside the verdicts, not scored."""
    last = blocks[-1]
    return dict(half_d2=sweep_of(first_reading_with(blocks, "n_open2", 2)),
                half_d3=sweep_of(first_reading_with(blocks, "d3", 2)),
                pause=pause_of(blocks), pieces=int(last["pieces"]), h_per_point=float(last["h"]) / int(last["N"]))


def summarize(rows):
    classes, notes = defaultdict(list), defaultdict(list)
    regions, regions_by_lam = defaultdict(list), defaultdict(list)
    for (length, lam, g, _), blocks in sorted(replicas(rows).items()):
        c = class_of(blocks)
        classes[(length, lam, g)].append(c)
        notes[(length, lam, g)].append(report_of(blocks))
        at_tenth = regions_at_first_tenth(blocks)
        if c in MOVED and at_tenth is not None:
            regions[length].append(at_tenth)
            regions_by_lam[(length, lam)].append(at_tenth)
    window, moved = {}, {}
    for key in sorted({k[:2] for k in classes}):
        cells = {g: v for (length, lam, g), v in classes.items() if (length, lam) == key}
        window[key], moved[key] = window_verdict(cells), moved_at(cells)
    return dict(cells={k: dict(Counter(v)) for k, v in classes.items()},
                majority={k: majority_of(v) for k, v in classes.items()}, window=window, moved=moved,
                regions=dict(regions), regions_by_lam=dict(regions_by_lam), several=several_verdict(regions),
                notes=dict(notes))


def median_line(values):
    return "none" if not values else "median %g over %d replicas %s" % (statistics.median(values), len(values), sorted(values))


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t53_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T53 results")
        return
    s = summarize(rows)
    print("Reported, not scored. A pause: the most points at d = 2 at %d or more readings in a row before d = 3 first "
          "holds the most." % PAUSE_READINGS)
    for k in sorted(s["cells"]):
        notes = s["notes"][k]
        print("L=%-3d lambda=%.2f g=%-4g %-52s -> %s" % (*k, s["cells"][k], s["majority"][k]))
        print("    sweep with half the points at d >= 2: %s" % [m["half_d2"] for m in notes])
        print("    sweep with half the points at d = 3:  %s" % [m["half_d3"] for m in notes])
        print("    pause on the one-curled rung (longest run of readings): %s"
              % ["%s (%d)" % m["pause"] for m in notes])
        print("    final pieces of the whole graph: %s" % [m["pieces"] for m in notes])
        print("    final energy per point: %s" % ["%.3f" % m["h_per_point"] for m in notes])
    for key, v in sorted(s["window"].items()):
        print("VERDICT T53 window L=%d lambda=%.2f: %s (something moved at g = %s)"
              % (key[0], key[1], v, s["moved"][key] or "none"))
    for key in sorted(s["regions_by_lam"]):
        print("    open regions at the first tenth, L=%d lambda=%.2f (beside, not scored): %s"
              % (key[0], key[1], median_line(s["regions_by_lam"][key])))
    for length in (SHORT, LONG):
        print("    open regions at the first tenth, L=%d, every lambda and g: %s"
              % (length, median_line(s["regions"].get(length, []))))
    print("VERDICT T53 one place or several: %s" % s["several"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
