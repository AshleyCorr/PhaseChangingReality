"""The registered T61 reading: if the burp's heat stays, does the tube come out wider? Usage:

    python scripts/analyse_t61.py [results_dir]                     # the report
    python scripts/analyse_t61.py [results_dir] --write-prediction  # after stage 1: the numbers for N = 384

Implements PREREGISTRATION.md section T61 on results/t61_lam<tag>_n<N>.csv and, for the held-out size, one file
per cell, results/t61_lam<tag>_n384_c<stores>.csv (scripts/run_burp_heat.py).

Per replica, from its end state: `sees3` is the share of points with exactly 25 points within three steps, as an
endless flat sheet has (0 on a flat torus six around, 1 on one eight or more around).
    WIDE       sees3 >= 0.5
    TUBE       sees3 <  0.05
    PATCHWORK  between
A replica that never converted (more than a quarter of its points still at d = 1) is reported and left out.
Per cell (lambda, N, stores per point): the outcome held by more than half of its converted replicas, else MIXED.
Per (lambda, N): WIDENS if some cell has a WIDE majority; else PATCHWORK if some cell has a PATCHWORK majority; else
STAYS A TUBE if every cell has a TUBE majority; else MIXED.
Gates per cell: every replica's energy books balance (drift below 1e-6), every end graph is valid, and at least
nine tenths of the replicas converted.

The held-out size (rule 14), N = 384: for every (lambda, stores per point) the shares of WIDE and of TUBE replicas
are predicted as the mean of the shares at N = 96 and 192, each with a range of 0.25 either side, written to
configs/t61_heldout_prediction.json before N = 384 runs. AS PREDICTED at a lambda if every cell passes its gates and
has both shares inside their ranges; the verdict word at N = 384 is read from N = 384 alone.

Safeguards (amendment of 11 October 2026, after an outside review and before N = 384 was read). No word is given on
incomplete or failed data. The registered design is fixed here: two lambda, five amounts of room, 24 replicas
numbered 0 to 23 in every cell, every number the reading uses present and finite. A (lambda, N) whose cells do not
all meet the design and pass their gates reads NOT READ, with the reason; the prediction for N = 384 is refused
unless all of stage 1 does; and N = 384 is scored at a lambda only when all five of its cells are complete.
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

STAGE1 = (96, 192)
HELD_OUT = 384
WIDE_AT, TUBE_BELOW, RANGE = 0.5, 0.05, 0.25
LAMBDAS = (1.25, 1.40)                              # the registered design
RATIOS = (2.0, 1.0, 0.5, 0.333333, 0.25)            # stores per point, as the configs write them
REPLICAS = 24
NUMBERS = ("N", "d1", "sees3", "drift", "h_end", "near5")
PREDICTION = Path(__file__).resolve().parents[1] / "configs" / "t61_heldout_prediction.json"


def outcome(row):
    """WIDE, PATCHWORK, TUBE, or NOT CONVERTED, for one replica."""
    if int(row["d1"]) > int(row["N"]) / 4:
        return "NOT CONVERTED"
    s = float(row["sees3"])
    return "WIDE" if s >= WIDE_AT else ("TUBE" if s < TUBE_BELOW else "PATCHWORK")


def cell(rows):
    """One cell's reading from its replica rows."""
    outs = Counter(outcome(r) for r in rows)
    converted = [r for r in rows if outcome(r) != "NOT CONVERTED"]
    n = len(converted)
    top, votes = (outs - Counter({"NOT CONVERTED": outs["NOT CONVERTED"]})).most_common(1)[0] if n else ("", 0)
    flat = [r for r in converted if r["flat"] == "True"]
    return dict(replicas=len(rows), converted=n, outcomes=dict(outs),
                majority=top if n and votes > n / 2 else "MIXED",
                wide=(outs["WIDE"] / n if n else float("nan")), tube=(outs["TUBE"] / n if n else float("nan")),
                gate=(all(float(r["drift"]) < 1e-6 for r in rows) and all(r["valid"] == "True" for r in rows)
                      and n >= 0.9 * len(rows)),
                flat=len(flat), widths=dict(Counter(int(r["width"]) for r in flat)),
                left=(statistics.median(float(r["h_end"]) for r in converted) if n else float("nan")),
                near5=(statistics.median(float(r["near5"]) for r in converted) if n else float("nan")))


def readable(rows):
    """True if every number the reading uses is present and finite in every row, and the flags are flags."""
    try:
        return all(math.isfinite(float(r[f])) for r in rows for f in NUMBERS) and all(
            r["valid"] in ("True", "False") and r["flat"] in ("True", "False") for r in rows)
    except (KeyError, TypeError, ValueError):
        return False


def find(groups, lam, n, ratio):
    """The key of the registered cell (lambda, N, stores per point) among the groups, or None."""
    for key in groups:
        if abs(key[0] - lam) < 1e-9 and key[1] == n and abs(key[2] - ratio) < 1e-4:
            return key
    return None


def design_problems(groups, sizes, lambdas=LAMBDAS):
    """What the rows lack against the registered design at these sizes and lambda: a list of sentences, empty if
    nothing. A missing cell, a cell without exactly the replicas 0 to 23, an unreadable number, and a cell that
    is not in the design are each a problem."""
    problems, wanted = [], set()
    for lam in lambdas:
        for n in sizes:
            for ratio in RATIOS:
                where = "lambda=%.2f N=%d stores per point %g" % (lam, n, ratio)
                key = find(groups, lam, n, ratio)
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
        if key[1] in sizes and any(abs(key[0] - lam) < 1e-9 for lam in lambdas) and key not in wanted:
            problems.append("a cell outside the registered design: lambda=%.2f N=%d stores per point %g" % key)
    return problems


def strays(groups):
    """Rows that belong to no registered lambda or size: a list of sentences."""
    out = ["a lambda outside the registered design: %.2f" % lam for lam in sorted({k[0] for k in groups})
           if not any(abs(lam - x) < 1e-9 for x in LAMBDAS)]
    return out + ["a size outside the registered design: N=%d" % n for n in sorted({k[1] for k in groups})
                  if n not in STAGE1 + (HELD_OUT,)]


def blocked(groups, cells, lam, n):
    """Why no word may be given at (lambda, N): its design problems and its cells that fail their gates."""
    why = design_problems(groups, (n,), (lam,))
    why += ["gate fails with %g stores per point" % k[2] for k, c in sorted(cells.items())
            if abs(k[0] - lam) < 1e-9 and k[1] == n and not c["gate"]]
    return why


def verdict(majorities):
    """The verdict word over one (lambda, N)'s cell majorities."""
    if "WIDE" in majorities:
        return "WIDENS"
    if "PATCHWORK" in majorities:
        return "PATCHWORK"
    if majorities and all(m == "TUBE" for m in majorities):
        return "STAYS A TUBE"
    return "MIXED"


def predict(cells):
    """{'lam|ratio': dict(wide=[value, low, high], tube=[...])} for N = 384 from the stage-1 cells."""
    out = {}
    keys = sorted({(lam, ratio) for (lam, n, ratio) in cells if n in STAGE1})
    for lam, ratio in keys:
        pair = [cells[(lam, n, ratio)] for n in STAGE1 if (lam, n, ratio) in cells]
        if len(pair) < len(STAGE1):
            continue
        entry = {}
        for key in ("wide", "tube"):
            value = sum(c[key] for c in pair) / len(pair)
            entry[key] = [value, max(0.0, value - RANGE), min(1.0, value + RANGE)]
        out["%.2f|%g" % (lam, ratio)] = entry
    return out


def held_out_misses(c, p):
    """What of a held-out cell falls outside its prediction: a list of strings, empty if nothing does."""
    out = [] if c["gate"] else ["gate"]
    for key in ("wide", "tube"):
        if not (p[key][1] <= c[key] <= p[key][2]):
            out.append("%s share %.2f outside %.2f to %.2f" % (key, c[key], p[key][1], p[key][2]))
    return out


def load(out_dir="results"):
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(Path(out_dir) / "t61_lam*_n*.csv"))):
        if not re.search(r"t61_lam\d+_n\d+(_c\d+)?\.csv$", f):
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                groups[(float(r["lam"]), int(r["N"]), float(r["stores_per_point"]))].append(r)
    return groups


def main(out_dir="results", flag=""):
    groups = load(out_dir)
    if not groups:
        print("no t61_lam*_n*.csv in %s" % out_dir)
        return 1
    cells = {k: cell(v) for k, v in groups.items() if readable(v)}
    print("T61: if the burp's heat stays, does the tube come out wider? (PREREGISTRATION T61)")
    for key in sorted(groups):
        lam, n, ratio = key
        if key not in cells:
            print("lambda=%.2f N=%-3d stores per point %-6g UNREADABLE: a number is missing or not finite" % key)
            continue
        c = cells[key]
        print("lambda=%.2f N=%-3d stores per point %-6g gate %s | %s -> %-9s | wide %.2f tube %.2f | exactly flat %d %s"
              " | energy left, median %.1f | within five steps, median %.1f"
              % (lam, n, ratio, "pass" if c["gate"] else "FAIL", c["outcomes"], c["majority"], c["wide"], c["tube"],
                 c["flat"], c["widths"], c["left"], c["near5"]))
    stage1_clean = True
    for lam in LAMBDAS:
        for n in STAGE1:
            why = blocked(groups, cells, lam, n)
            stage1_clean = stage1_clean and not why
            print("VERDICT lambda=%.2f N=%d: %s" % (lam, n, "NOT READ (%s)" % "; ".join(why) if why else verdict(
                [c["majority"] for k, c in sorted(cells.items()) if abs(k[0] - lam) < 1e-9 and k[1] == n])))
    stray = strays(groups)
    for line in stray:
        print("NOTE:", line)
    if flag == "--write-prediction":
        if PREDICTION.exists():
            raise SystemExit("refusing to overwrite %s" % PREDICTION)
        if not stage1_clean or stray:
            raise SystemExit("stage 1 is not complete and clean; nothing written")
        recipe = predict(cells)
        if len(recipe) != len(LAMBDAS) * len(RATIOS):
            raise SystemExit("stage 1 does not give all %d numbers; nothing written" % (len(LAMBDAS) * len(RATIOS)))
        PREDICTION.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("wrote", PREDICTION.name)
        return 0
    if not any(k[1] == HELD_OUT for k in groups):
        return 0
    if not PREDICTION.exists():
        print("N = %d is present but no prediction file is committed: not read." % HELD_OUT)
        return 1
    fixed = json.loads(PREDICTION.read_text(encoding="utf-8"))
    status = 0
    for lam in LAMBDAS:
        why = design_problems(groups, (HELD_OUT,), (lam,))
        why += ["no number was written beforehand for %g stores per point" % ratio for ratio in RATIOS
                if "%.2f|%g" % (lam, ratio) not in fixed]
        if why:
            print("HELD-OUT lambda=%.2f N=%d: NOT READ (%s)" % (lam, HELD_OUT, "; ".join(why)))
            status = 1
            continue
        mine = {k: c for k, c in cells.items() if abs(k[0] - lam) < 1e-9 and k[1] == HELD_OUT}
        wrong = {k[2]: held_out_misses(c, fixed["%.2f|%g" % (lam, k[2])]) for k, c in mine.items()}
        wrong = {r: m for r, m in wrong.items() if m}
        gates = all(c["gate"] for c in mine.values())
        print("HELD-OUT lambda=%.2f N=%d: %s | verdict there: %s"
              % (lam, HELD_OUT, "AS PREDICTED" if not wrong else "NOT AS PREDICTED AT " + "; ".join(
                  "%g (%s)" % (r, ", ".join(m)) for r, m in sorted(wrong.items())),
                 verdict([c["majority"] for _k, c in sorted(mine.items())]) if gates else "NOT READ (a gate fails)"))
    return status


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    paths = [a for a in args if not a.startswith("--")]
    sys.exit(main(paths[0] if paths else "results", flags[0] if flags else ""))
