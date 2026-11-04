"""T56: the slab and the rod in a warm bath under the owner's tie, read by the rules written before the runs. Usage:

    python scripts/analyse_t56.py [results_dir]

PREREGISTRATION T56 (written 2026-10-09, before any run). Reads `results/t56_*.csv` (scripts/run_curled_bath_tie_d.py).
Built on T53's analyzer (`analyse_t53`: the replica rule, the majority, the window, the several rule), imported and not
rewritten; what is new is read here:

  the window     per (geometry, L, lambda), over the tied cells: OPENS IN A WINDOW / ADVANCES ONLY / DAMAGED / STAYS;
  one space      per cell with an OPENS majority: ONE SPACE if, in a majority of its OPENS replicas, the whole graph is
                 one piece (`pieces` = 1) and the largest piece at d = 3 holds at least NINE_TENTHS of the points; else
                 OPEN WITH SEAMS;
  the control    the four untied slab cells at lambda = 1.20 against the tied cells with the same (L, g): THE TIE MADE
                 THE DIFFERENCE if the tied cell has an OPENS majority and the untied one does not in at least three
                 pairs; NO DIFFERENCE if the majorities agree in at least three; else MIXED;
  one or several per geometry: the rod by T53's rule (`open_regions` at the first reading with a tenth of the points at
                 d >= 2); the slab one rung up (`open_regions3` at the first reading with a tenth at d = 3); SEVERAL if
                 the median over replicas that OPEN or ADVANCE is 2 or more at the larger stage-1 length and larger
                 there than at the smaller, else ONE;
  held out       per (geometry, lambda): the set of g with an OPENS majority at each size, and the rule's prediction
                 for the held-out size (the g's OPENS at both stage-1 sizes, less any whose mean damaged share at the
                 end at the larger size exceeded ONE_EIGHTH); when the held-out size is present, HOLDS if the predicted
                 and measured sets differ by at most one g, else MISSES.

Reported, not scored: the sweep at which half the points first sit at d = 3, the final (H + T_f) per point, the
pieces of the whole graph, and the damaged share at the end. Every six-link result carries VISION Update 24's caveat.
"""
import csv
import glob
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t48 import dims_of                                               # noqa: E402
from analyse_t53 import (class_of, first_reading_with, majority_of, sweep_of,  # noqa: E402
                         window_verdict)

NINE_TENTHS, ONE_EIGHTH = 0.9, 0.125
STAGE1 = {"slab": (8, 12), "rod": (18, 36)}
HELD_OUT = {"slab": 16, "rod": 72}
CONTROL_LAM = 1.20


def geometry_of(row):
    """slab for 4 x L x L (one side of 4), rod for 4 x 4 x L (two sides of 4)."""
    return "rod" if sum(1 for s in dims_of(row) if s == 4) >= 2 else "slab"


def length_of(row):
    return max(dims_of(row))


def load(results="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(results) / "t56_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return rows


def replicas(rows):
    """{(geometry, L, lambda, g, tie, replica): readings in order of sweep}."""
    runs = defaultdict(list)
    for r in rows:
        runs[(geometry_of(r), length_of(r), float(r["lam"]), float(r["g"]), r["tie"], int(r["replica"]))].append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r["sweep"]))
    return runs


def one_space(blocks_of_opens):
    """ONE SPACE or OPEN WITH SEAMS over the last readings of a cell's OPENS replicas."""
    votes = 0
    for blocks in blocks_of_opens:
        last = blocks[-1]
        n = int(last["N"])
        if int(last["pieces"]) == 1 and int(last["largest_open_d"]) >= NINE_TENTHS * n:
            votes += 1
    return "ONE SPACE" if votes > len(blocks_of_opens) / 2 else "OPEN WITH SEAMS"


def control_verdict(pairs):
    """`pairs`: [(tied majority, untied majority)] for the four control cells."""
    if not pairs:
        return "NO CONTROL"
    made = sum(1 for t, u in pairs if t == "OPENS" and u != "OPENS")
    same = sum(1 for t, u in pairs if t == u)
    if made >= 3:
        return "THE TIE MADE THE DIFFERENCE"
    if same >= 3:
        return "NO DIFFERENCE"
    return "MIXED"


def regions_at_first_tenth(blocks, geometry):
    col_n, col_r = ("n_open3", "open_regions3") if geometry == "slab" else ("n_open2", "open_regions")
    b = first_reading_with(blocks, col_n, 10)
    return None if b is None else int(b[col_r])


def several_verdict(regions, lengths):
    """SEVERAL or ONE for one geometry; `regions`: {L: [regions, one per replica that OPENS or ADVANCES]}."""
    short, long_ = regions.get(lengths[0], []), regions.get(lengths[1], [])
    if not short or not long_:
        return "ONE"
    at_long, at_short = statistics.median(long_), statistics.median(short)
    return "SEVERAL" if at_long >= 2 and at_long > at_short else "ONE"


def damaged_share(blocks):
    last = blocks[-1]
    return int(last["damaged_d"]) / int(last["N"])


def held_out_prediction(opens_by_size, damage_by_size, lengths):
    """The rule: g's with an OPENS majority at both stage-1 sizes, less any whose mean damaged share at the larger
    size exceeded ONE_EIGHTH."""
    both = set(opens_by_size.get(lengths[0], set())) & set(opens_by_size.get(lengths[1], set()))
    return sorted(g for g in both if damage_by_size.get(lengths[1], {}).get(g, 0.0) <= ONE_EIGHTH)


def held_out_score(predicted, measured):
    return "HOLDS" if len(set(predicted) ^ set(measured)) <= 1 else "MISSES"


def summarize(rows):
    runs = replicas(rows)
    cells = defaultdict(dict)            # (geometry, L, lam, tie) -> {g: [classes]}
    blocks_by_cell = defaultdict(lambda: defaultdict(list))
    for (geometry, length, lam, g, tie, rep), blocks in sorted(runs.items()):
        cells[(geometry, length, lam, tie)].setdefault(g, []).append(class_of(blocks))
        blocks_by_cell[(geometry, length, lam, tie)][g].append(blocks)
    out = dict(cells={}, windows={}, one_space={}, control=None, several={}, held_out={}, reported={})
    for key, by_g in cells.items():
        for g, classes in by_g.items():
            out["cells"][key + (g,)] = (dict(Counter(classes)), majority_of(classes))
            if majority_of(classes) == "OPENS":
                opens = [b for b, c in zip(blocks_by_cell[key][g], classes) if c == "OPENS"]
                out["one_space"][key + (g,)] = one_space(opens)
            out["reported"][key + (g,)] = dict(
                half_d3=[sweep_of(first_reading_with(b, "d3", 2)) for b in blocks_by_cell[key][g]],
                energy=[round((float(b[-1]["h"]) + float(b[-1]["t_tie"])) / int(b[-1]["N"]), 3)
                        for b in blocks_by_cell[key][g]],
                pieces=[int(b[-1]["pieces"]) for b in blocks_by_cell[key][g]],
                damaged=[round(damaged_share(b), 3) for b in blocks_by_cell[key][g]])
        if key[3] == "all_at_the_last":
            out["windows"][key[:3]] = window_verdict(by_g)
    pairs = []
    for (geometry, length, lam, tie), by_g in cells.items():
        if geometry == "slab" and lam == CONTROL_LAM and tie == "none":
            for g, classes in by_g.items():
                tied = cells.get((geometry, length, lam, "all_at_the_last"), {}).get(g)
                if tied is not None:
                    pairs.append((majority_of(tied), majority_of(classes)))
    out["control"] = control_verdict(pairs)
    for geometry, lengths in STAGE1.items():
        regions = defaultdict(list)
        for (geo, length, lam, g, tie, rep), blocks in runs.items():
            if geo == geometry and tie == "all_at_the_last" and class_of(blocks) in ("OPENS", "ADVANCES"):
                r = regions_at_first_tenth(blocks, geometry)
                if r is not None:
                    regions[length].append(r)
        out["several"][geometry] = (several_verdict(regions, lengths), dict(regions))
        lams = sorted({k[2] for k in cells if k[0] == geometry})
        for lam in lams:
            opens_by_size, damage_by_size = {}, {}
            for (geo, length, lam2, tie), by_g in cells.items():
                if geo == geometry and lam2 == lam and tie == "all_at_the_last":
                    opens_by_size[length] = {g for g, c in by_g.items() if majority_of(c) == "OPENS"}
                    damage_by_size[length] = {g: statistics.mean(damaged_share(b) for b in blocks_by_cell[(geo, length, lam2, tie)][g])
                                              for g in by_g}
            predicted = held_out_prediction(opens_by_size, damage_by_size, lengths)
            measured = opens_by_size.get(HELD_OUT[geometry])
            out["held_out"][(geometry, lam)] = (opens_by_size, predicted,
                                                None if measured is None else (sorted(measured), held_out_score(predicted, measured)))
    return out


def main(results="results"):
    rows = load(results)
    if not rows:
        print("no t56_*.csv in %s" % results)
        return 1
    s = summarize(rows)
    print("T56, read by the rules of PREREGISTRATION T56 (written 2026-10-09, before any run). Six links: Gate C open.")
    for key, (counts, maj) in sorted(s["cells"].items()):
        geometry, length, lam, tie, g = key
        extra = ("  " + s["one_space"][key]) if key in s["one_space"] else ""
        rep = s["reported"][key]
        print("%-4s L=%-3d lambda=%.2f g=%.1f %-15s %-45s -> %-9s%s" % (geometry, length, lam, g, tie, counts, maj, extra))
        print("    half at d=3: %s | (H+T)/N: %s | pieces: %s | damaged: %s"
              % (rep["half_d3"], rep["energy"], rep["pieces"], rep["damaged"]))
    for (geometry, length, lam), v in sorted(s["windows"].items()):
        print("VERDICT T56 window %s L=%d lambda=%.2f: %s" % (geometry, length, lam, v))
    print("VERDICT T56 control (slab, lambda = %.2f): %s" % (CONTROL_LAM, s["control"]))
    for geometry, (v, regions) in s["several"].items():
        print("VERDICT T56 one place or several, %s: %s  %s" % (geometry, v, regions))
    for (geometry, lam), (opens, predicted, measured) in sorted(s["held_out"].items()):
        print("HELD OUT %s lambda=%.2f: OPENS at %s | rule predicts %s at L=%d | measured %s"
              % (geometry, lam, {k: sorted(v) for k, v in opens.items()}, predicted, HELD_OUT[geometry],
                 "not run" if measured is None else "%s -> %s" % measured))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
