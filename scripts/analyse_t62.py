"""The registered T62 reading: with a price on loops of six, does the tube six around come out wider? Usage:

    python scripts/analyse_t62.py [results_dir]                     # the report
    python scripts/analyse_t62.py [results_dir] --write-prediction  # after stage 1: the numbers for N = 384

Implements PREREGISTRATION.md section T62 on results/t62_mu<tag>_g<tag>_n<N>.csv (scripts/run_sixloop_widen.py).

Per replica, from its end state (the words and thresholds are T61's): `sees3` is the share of points with exactly
25 points within three steps, as an endless flat sheet has.
    WIDE       sees3 >= 0.5
    TUBE       sees3 <  0.05
    PATCHWORK  between
Per cell (mu, g, N): the outcome held by more than half its replicas, else MIXED. Gates per cell: every end graph
valid, and the chain's running counts equal to a full recount in every replica.
The control: at mu = 0 the rule is the toy's own, under which six around is stuck. A cell with mu > 0 is read only
if the control cell at its own g and N passes its gates and at least nine tenths of its replicas end TUBE.
Otherwise the bath alone is moving the tube at that g, and the cell is NOT READ.
Per (mu, N), over the cells that are read: WIDENS if some cell has a WIDE majority; else PATCHWORK if some cell
has a PATCHWORK majority; else STAYS SIX AROUND if every cell has a TUBE majority; else MIXED.

The held-out size (rule 14), N = 384: for every (mu, g) the shares of WIDE and of TUBE replicas are predicted as
the mean of the shares at N = 96 and 192, each with a range of 0.25 either side, written to
configs/t62_heldout_prediction.json before N = 384 runs. AS PREDICTED at a mu if every one of its cells at N = 384
is read and has both shares inside their ranges.

No word is given on incomplete or failed data. The design is fixed here: five prices, two couplings, 24 replicas
numbered 0 to 23 in every cell, every number the reading uses present and finite.

Reported beside the words and scored nowhere, from the saved wiring (scripts/read_flat_kind.py): how many end
states are in more than one piece, and how many are CLEAN AND WIDE: exactly flat, no six-loop excess left, one
piece, two-sided, and eight or more around.
"""
import csv
import glob
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGE1 = (96, 192)
HELD_OUT = 384
WIDE_AT, TUBE_BELOW, RANGE = 0.5, 0.05, 0.25
MUS = (0.0, 0.10, 0.20, 0.25, 0.30)
GS = (1.5, 2.0)
REPLICAS = 24
CONTROL_TUBE = 0.9
NUMBERS = ("N", "mu", "g", "sees3", "e6_start", "e6_end", "h_end", "phi_min", "left_sweep")
PREDICTION = Path(__file__).resolve().parents[1] / "configs" / "t62_heldout_prediction.json"


def outcome(row):
    s = float(row["sees3"])
    return "WIDE" if s >= WIDE_AT else ("TUBE" if s < TUBE_BELOW else "PATCHWORK")


def readable(rows):
    try:
        return all(math.isfinite(float(r[f])) for r in rows for f in NUMBERS) and all(
            r[f] in ("True", "False") for r in rows for f in ("valid", "books", "flat"))
    except (KeyError, TypeError, ValueError):
        return False


def cell(rows):
    """One cell's reading from its replica rows."""
    outs = Counter(outcome(r) for r in rows)
    n = len(rows)
    top, votes = outs.most_common(1)[0]
    return dict(replicas=n, outcomes=dict(outs), majority=top if votes > n / 2 else "MIXED",
                wide=outs["WIDE"] / n, tube=outs["TUBE"] / n,
                gate=all(r["valid"] == "True" and r["books"] == "True" for r in rows),
                left=sum(int(r["left_sweep"]) >= 0 for r in rows) / n,
                flat=sum(r["flat"] == "True" for r in rows),
                flat_no_excess=sum(r["flat"] == "True" and int(r["e6_end"]) == 0 for r in rows),
                h_end=statistics.median(float(r["h_end"]) for r in rows),
                e6_left=statistics.median(int(r["e6_end"]) / max(int(r["e6_start"]), 1) for r in rows),
                phi_min=min(float(r["phi_min"]) for r in rows))


def find(groups, mu, g, n):
    for key in groups:
        if abs(key[0] - mu) < 1e-9 and abs(key[1] - g) < 1e-9 and key[2] == n:
            return key
    return None


def design_problems(groups, sizes, mus=MUS):
    """What the rows lack against the registered design at these sizes and prices: a list of sentences."""
    problems, wanted = [], set()
    for mu in mus:
        for g in GS:
            for n in sizes:
                where = "mu=%.2f g=%.2f N=%d" % (mu, g, n)
                key = find(groups, mu, g, n)
                if key is None:
                    problems.append("missing cell: " + where)
                    continue
                wanted.add(key)
                rows = groups[key]
                try:
                    ids = sorted(int(r["replica"]) for r in rows)
                except (KeyError, TypeError, ValueError):
                    ids = []
                if ids != list(range(REPLICAS)):
                    problems.append("%s: %d rows, not the replicas 0 to %d once each" % (where, len(rows), REPLICAS - 1))
                if not readable(rows):
                    problems.append("%s: a number the reading uses is missing or not finite" % where)
    for key in sorted(groups):
        if key[2] in sizes and any(abs(key[0] - mu) < 1e-9 for mu in mus) and key not in wanted:
            problems.append("a cell outside the registered design: mu=%.2f g=%.2f N=%d" % key)
    return problems


def strays(groups):
    out = ["a price outside the registered design: %.2f" % mu for mu in sorted({k[0] for k in groups})
           if not any(abs(mu - x) < 1e-9 for x in MUS)]
    out += ["a coupling outside the registered design: %.2f" % g for g in sorted({k[1] for k in groups})
            if not any(abs(g - x) < 1e-9 for x in GS)]
    return out + ["a size outside the registered design: N=%d" % n for n in sorted({k[2] for k in groups})
                  if n not in STAGE1 + (HELD_OUT,)]


def control_fails(groups, cells, g, n):
    """Why the control at (g, N) does not license reading its cells: a list, empty if it does."""
    why = design_problems(groups, (n,), (0.0,))
    why = [w for w in why if "g=%.2f" % g in w]
    key = find(cells, 0.0, g, n)
    if key is None:
        return why or ["the control at g=%.2f N=%d is missing or unreadable" % (g, n)]
    c = cells[key]
    if not c["gate"]:
        why.append("the control at g=%.2f fails its gates" % g)
    if c["tube"] < CONTROL_TUBE:
        why.append("the control at g=%.2f: only %.2f end TUBE" % (g, c["tube"]))
    return why


def read_cells(groups, cells, mu, n):
    """(the cells at (mu, N) that may be read, {g: reason} for those that may not)."""
    ok, blocked = {}, {}
    for g in GS:
        why = [w for w in design_problems(groups, (n,), (mu,)) if "g=%.2f" % g in w]
        key = find(cells, mu, g, n)
        if key is not None and not cells[key]["gate"]:
            why.append("its gates fail")
        if mu > 0.0:
            why += control_fails(groups, cells, g, n)
        if why or key is None:
            blocked[g] = "; ".join(why) or "missing"
        else:
            ok[g] = cells[key]
    return ok, blocked


def verdict(majorities):
    if not majorities:
        return "NOT READ"
    if "WIDE" in majorities:
        return "WIDENS"
    if "PATCHWORK" in majorities:
        return "PATCHWORK"
    if all(m == "TUBE" for m in majorities):
        return "STAYS SIX AROUND"
    return "MIXED"


def predict(cells):
    """{'mu|g': dict(wide=[value, low, high], tube=[...])} for N = 384 from the stage-1 cells."""
    out = {}
    for mu in MUS:
        for g in GS:
            pair = [cells[k] for k in (find(cells, mu, g, n) for n in STAGE1) if k is not None]
            if len(pair) < len(STAGE1):
                continue
            entry = {}
            for key in ("wide", "tube"):
                value = sum(c[key] for c in pair) / len(pair)
                entry[key] = [value, max(0.0, value - RANGE), min(1.0, value + RANGE)]
            out["%.2f|%.2f" % (mu, g)] = entry
    return out


def held_out_misses(c, p):
    out = []
    for key in ("wide", "tube"):
        if not (p[key][1] <= c[key] <= p[key][2]):
            out.append("%s share %.2f outside %.2f to %.2f" % (key, c[key], p[key][1], p[key][2]))
    return out


def load(out_dir="results"):
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(Path(out_dir) / "t62_mu*_g*_n*.csv"))):
        if not re.search(r"t62_mu\d+_g\d+_n\d+\.csv$", f):
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = Path(f).stem
                groups[(float(r["mu"]), float(r["g"]), int(r["N"]))].append(r)
    return groups


def geometry(out_dir, rows):
    """From the saved wiring, scored nowhere: (end states in more than one piece, CLEAN AND WIDE, flat with a
    one-sided piece). None if the graphs are not there."""
    from read_flat_kind import KLEIN, TORUS, describe, pieces      # read on the laptop, not by the runner
    split = clean = one_sided = 0
    for r in rows:
        path = Path(out_dir) / (r["_file"] + "_adj") / ("N%s_rep%s.npz" % (r["N"], r["replica"]))
        if not path.exists():
            return None
        adj = np.load(path)["adj"]
        split += len(pieces(adj)) > 1
        if r["flat"] == "True":
            parts = describe(adj)
            one_sided += any(k == KLEIN for _n, k, _w in parts)
            clean += (int(r["e6_end"]) == 0 and len(parts) == 1 and parts[0][1] == TORUS and parts[0][2] is not None
                      and parts[0][2] >= 8)
    return split, clean, one_sided


def main(out_dir="results", flag=""):
    groups = load(out_dir)
    if not groups:
        print("no t62_mu*_g*_n*.csv in %s" % out_dir)
        return 1
    cells = {k: cell(v) for k, v in groups.items() if readable(v)}
    print("T62: with a price on loops of six, does the tube six around come out wider? (PREREGISTRATION T62)")
    for key in sorted(groups, key=lambda k: (k[2], k[0], k[1])):
        mu, g, n = key
        if key not in cells:
            print("mu=%.2f g=%.2f N=%-3d UNREADABLE: a number is missing or not finite" % key)
            continue
        c = cells[key]
        geo = geometry(out_dir, groups[key])
        print("mu=%.2f g=%.2f N=%-3d gate %s | %s -> %-9s | wide %.2f tube %.2f | left six around %.2f | energy left, "
              "median %.1f | six-loop excess left, median %.2f | fewest squares per point %.2f | exactly flat %d, of "
              "those no excess %d%s"
              % (mu, g, n, "pass" if c["gate"] else "FAIL", c["outcomes"], c["majority"], c["wide"], c["tube"],
                 c["left"], c["h_end"], c["e6_left"], c["phi_min"], c["flat"], c["flat_no_excess"],
                 "" if geo is None else " | from the wiring: in pieces %d, clean and wide %d, one-sided %d" % geo))
    stage1_clean = True
    for n in STAGE1:
        for g in GS:
            why = control_fails(groups, cells, g, n)
            print("CONTROL g=%.2f N=%d: %s" % (g, n, "holds" if not why else "FAILS (%s)" % "; ".join(why)))
        for mu in MUS:
            ok, blocked = read_cells(groups, cells, mu, n)
            stage1_clean = stage1_clean and not blocked
            print("VERDICT mu=%.2f N=%d: %s%s" % (mu, n, verdict([c["majority"] for _g, c in sorted(ok.items())]),
                  "" if not blocked else " | not read: " + "; ".join("g=%.2f (%s)" % kv for kv in sorted(blocked.items()))))
    stray = strays(groups)
    for line in stray:
        print("NOTE:", line)
    if flag == "--write-prediction":
        if PREDICTION.exists():
            raise SystemExit("refusing to overwrite %s" % PREDICTION)
        if design_problems(groups, STAGE1) or stray or not all(c["gate"] for k, c in cells.items() if k[2] in STAGE1):
            raise SystemExit("stage 1 is not complete and clean; nothing written")
        recipe = predict(cells)
        if len(recipe) != len(MUS) * len(GS):
            raise SystemExit("stage 1 does not give all %d numbers; nothing written" % (len(MUS) * len(GS)))
        PREDICTION.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("wrote", PREDICTION.name)
        return 0
    if not any(k[2] == HELD_OUT for k in groups):
        return 0
    if not PREDICTION.exists():
        print("N = %d is present but no prediction file is committed: not read." % HELD_OUT)
        return 1
    fixed = json.loads(PREDICTION.read_text(encoding="utf-8"))
    status = 0
    for mu in MUS:
        ok, blocked = read_cells(groups, cells, mu, HELD_OUT)
        missing = [g for g in GS if "%.2f|%.2f" % (mu, g) not in fixed]
        if blocked or missing:
            print("HELD-OUT mu=%.2f N=%d: NOT READ (%s)" % (mu, HELD_OUT, "; ".join(
                ["g=%.2f (%s)" % kv for kv in sorted(blocked.items())] + ["no number written for g=%.2f" % g for g in missing])))
            status = 1
            continue
        wrong = {g: held_out_misses(c, fixed["%.2f|%.2f" % (mu, g)]) for g, c in ok.items()}
        wrong = {g: m for g, m in wrong.items() if m}
        print("HELD-OUT mu=%.2f N=%d: %s | verdict there: %s"
              % (mu, HELD_OUT, "AS PREDICTED" if not wrong else "NOT AS PREDICTED AT " + "; ".join(
                  "g=%.2f (%s)" % (g, ", ".join(m)) for g, m in sorted(wrong.items())),
                 verdict([c["majority"] for _g, c in sorted(ok.items())])))
    return status


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    paths = [a for a in args if not a.startswith("--")]
    sys.exit(main(paths[0] if paths else "results", flags[0] if flags else ""))
