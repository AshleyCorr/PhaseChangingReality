"""The registered T63 reading: with a price on loops of six, what does the curled tube itself open into? Usage:

    python scripts/analyse_t63.py [results_dir]                     # the report
    python scripts/analyse_t63.py [results_dir] --write-prediction  # after stage 1: the numbers for N = 384

Implements PREREGISTRATION.md section T63 on results/t63_mu<tag>_n<N>.csv (scripts/run_sixloop_widen.py with
"start": "tube").

Per replica, from its end state. NOT CONVERTED if more than a quarter of a surplus square per point is left (the
curled tube has one per point); such a replica is reported and left out. Otherwise, by `sees3`, the share of
points with exactly 25 points within three steps (the words and thresholds are T61's and T62's):
    WIDE       sees3 >= 0.5
    TUBE       sees3 <  0.05
    PATCHWORK  between
Per cell (mu, N): the outcome held by more than half its converted replicas, else MIXED. Gates per cell: every end
graph valid, the chain's running counts equal to a full recount in every replica, and at least nine tenths
converted.
The control: at mu = 0 the rule is the toy's own, and paper 1's result is that the tube ends six around. A cell
with mu > 0 is read only if the control at its N passes its gates and at least nine tenths of its converted
replicas end TUBE.
Per (mu, N): WIDENS if the cell has a WIDE majority; PATCHWORK if a PATCHWORK majority; ENDS SIX AROUND if a TUBE
majority; else MIXED.

The held-out size (rule 14), N = 384: for every mu the shares of WIDE and of TUBE among converted replicas are
predicted as the mean of the shares at N = 96 and 192, each with a range of 0.25 either side, written to
configs/t63_heldout_prediction.json before N = 384 runs. AS PREDICTED at a mu if its cell at N = 384 is read and
has both shares inside their ranges.

No word is given on incomplete or failed data: five prices, 24 replicas numbered 0 to 23 in every cell, every
number the reading uses present and finite.

Reported beside the words and scored nowhere: when the curl was gone, the energy and six-loop excess left, the
fewest squares per point on the way; and from the saved wiring (scripts/read_flat_kind.py) how many end states are
in more than one piece, hold a one-sided piece, or are CLEAN AND WIDE (exactly flat, no excess left, one piece,
two-sided, eight or more around).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t62 import RANGE, TUBE_BELOW, WIDE_AT, geometry, held_out_misses     # noqa: E402

STAGE1 = (96, 192)
HELD_OUT = 384
MUS = (0.0, 0.025, 0.05, 0.10, 0.20)
REPLICAS = 24
CONVERTED_AT_LEAST = 0.9
CONTROL_TUBE = 0.9
NUMBERS = ("N", "mu", "sees3", "x_end", "e6_start", "e6_end", "h_end", "phi_min", "curl_sweep")
PREDICTION = Path(__file__).resolve().parents[1] / "configs" / "t63_heldout_prediction.json"


def outcome(row):
    if int(row["x_end"]) > int(row["N"]) / 4:
        return "NOT CONVERTED"
    s = float(row["sees3"])
    return "WIDE" if s >= WIDE_AT else ("TUBE" if s < TUBE_BELOW else "PATCHWORK")


def readable(rows):
    try:
        return all(math.isfinite(float(r[f])) for r in rows for f in NUMBERS) and all(
            r[f] in ("True", "False") for r in rows for f in ("valid", "books", "flat"))
    except (KeyError, TypeError, ValueError):
        return False


def cell(rows):
    outs = Counter(outcome(r) for r in rows)
    converted = [r for r in rows if outcome(r) != "NOT CONVERTED"]
    n = len(converted)
    top, votes = (outs - Counter({"NOT CONVERTED": outs["NOT CONVERTED"]})).most_common(1)[0] if n else ("", 0)
    waits = [int(r["curl_sweep"]) for r in converted if int(r["curl_sweep"]) >= 0]
    return dict(replicas=len(rows), converted=n, outcomes=dict(outs),
                majority=top if n and votes > n / 2 else "MIXED",
                wide=(outs["WIDE"] / n if n else float("nan")), tube=(outs["TUBE"] / n if n else float("nan")),
                gate=(all(r["valid"] == "True" and r["books"] == "True" for r in rows)
                      and n >= CONVERTED_AT_LEAST * len(rows)),
                curl_gone=(statistics.median(waits) if waits else float("nan")),
                flat=sum(r["flat"] == "True" for r in converted),
                flat_no_excess=sum(r["flat"] == "True" and int(r["e6_end"]) == 0 for r in converted),
                h_end=(statistics.median(float(r["h_end"]) for r in converted) if n else float("nan")),
                e6_left=(statistics.median(int(r["e6_end"]) / max(int(r["e6_start"]), 1) for r in converted)
                         if n else float("nan")),
                phi_min=min(float(r["phi_min"]) for r in rows))


def find(groups, mu, n):
    for key in groups:
        if abs(key[0] - mu) < 1e-9 and key[1] == n:
            return key
    return None


def design_problems(groups, sizes, mus=MUS):
    problems, wanted = [], set()
    for mu in mus:
        for n in sizes:
            where = "mu=%.3f N=%d" % (mu, n)
            key = find(groups, mu, n)
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
            if any(r.get("start") != "tube" for r in rows):
                problems.append("%s: a replica did not start from the curled tube" % where)
    for key in sorted(groups):
        if key[1] in sizes and any(abs(key[0] - mu) < 1e-9 for mu in mus) and key not in wanted:
            problems.append("a cell outside the registered design: mu=%.3f N=%d" % key)
    return problems


def strays(groups):
    out = ["a price outside the registered design: %.3f" % mu for mu in sorted({k[0] for k in groups})
           if not any(abs(mu - x) < 1e-9 for x in MUS)]
    return out + ["a size outside the registered design: N=%d" % n for n in sorted({k[1] for k in groups})
                  if n not in STAGE1 + (HELD_OUT,)]


def control_fails(groups, cells, n):
    why = design_problems(groups, (n,), (0.0,))
    key = find(cells, 0.0, n)
    if key is None:
        return why or ["the control at N=%d is missing or unreadable" % n]
    c = cells[key]
    if not c["gate"]:
        why.append("the control fails its gates")
    if not (c["tube"] >= CONTROL_TUBE):
        why.append("the control: only %.2f of the converted end six around" % c["tube"])
    return why


def blocked(groups, cells, mu, n):
    """Why no word may be given at (mu, N): a list, empty if it may."""
    why = design_problems(groups, (n,), (mu,))
    key = find(cells, mu, n)
    if key is not None and not cells[key]["gate"]:
        why.append("its gates fail")
    if mu > 0.0:
        why += control_fails(groups, cells, n)
    return why


def verdict(majority):
    return {"WIDE": "WIDENS", "PATCHWORK": "PATCHWORK", "TUBE": "ENDS SIX AROUND"}.get(majority, "MIXED")


def predict(cells):
    out = {}
    for mu in MUS:
        pair = [cells[k] for k in (find(cells, mu, n) for n in STAGE1) if k is not None]
        if len(pair) < len(STAGE1):
            continue
        entry = {}
        for key in ("wide", "tube"):
            value = sum(c[key] for c in pair) / len(pair)
            entry[key] = [value, max(0.0, value - RANGE), min(1.0, value + RANGE)]
        out["%.3f" % mu] = entry
    return out


def load(out_dir="results"):
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(Path(out_dir) / "t63_mu*_n*.csv"))):
        if not re.search(r"t63_mu\d+_n\d+\.csv$", f):
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = Path(f).stem
                groups[(float(r["mu"]), int(r["N"]))].append(r)
    return groups


def main(out_dir="results", flag=""):
    groups = load(out_dir)
    if not groups:
        print("no t63_mu*_n*.csv in %s" % out_dir)
        return 1
    cells = {k: cell(v) for k, v in groups.items() if readable(v)}
    print("T63: with a price on loops of six, what does the curled tube itself open into? (PREREGISTRATION T63)")
    for key in sorted(groups, key=lambda k: (k[1], k[0])):
        mu, n = key
        if key not in cells:
            print("mu=%.3f N=%-3d UNREADABLE: a number is missing or not finite" % key)
            continue
        c = cells[key]
        geo = geometry(out_dir, [r for r in groups[key] if outcome(r) != "NOT CONVERTED"])
        print("mu=%.3f N=%-3d gate %s | %s -> %-9s | wide %.2f tube %.2f | curl gone at sweep, median %.0f | energy left, "
              "median %.1f | six-loop excess left, median %.2f | fewest squares per point %.2f | exactly flat %d, of those "
              "no excess %d%s"
              % (mu, n, "pass" if c["gate"] else "FAIL", c["outcomes"], c["majority"], c["wide"], c["tube"],
                 c["curl_gone"], c["h_end"], c["e6_left"], c["phi_min"], c["flat"], c["flat_no_excess"],
                 "" if geo is None else " | from the wiring: in pieces %d, clean and wide %d, one-sided %d" % geo))
    stage1_clean = True
    for n in STAGE1:
        why = control_fails(groups, cells, n)
        print("CONTROL N=%d: %s" % (n, "holds" if not why else "FAILS (%s)" % "; ".join(why)))
        for mu in MUS:
            why = blocked(groups, cells, mu, n)
            stage1_clean = stage1_clean and not why
            key = find(cells, mu, n)
            print("VERDICT mu=%.3f N=%d: %s" % (mu, n, "NOT READ (%s)" % "; ".join(why) if why
                                                 else verdict(cells[key]["majority"])))
    stray = strays(groups)
    for line in stray:
        print("NOTE:", line)
    if flag == "--write-prediction":
        if PREDICTION.exists():
            raise SystemExit("refusing to overwrite %s" % PREDICTION)
        if design_problems(groups, STAGE1) or stray or not all(c["gate"] for k, c in cells.items() if k[1] in STAGE1):
            raise SystemExit("stage 1 is not complete and clean; nothing written")
        recipe = predict(cells)
        if len(recipe) != len(MUS):
            raise SystemExit("stage 1 does not give all %d numbers; nothing written" % len(MUS))
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
    for mu in MUS:
        why = blocked(groups, cells, mu, HELD_OUT)
        if "%.3f" % mu not in fixed:
            why.append("no number was written beforehand")
        if why:
            print("HELD-OUT mu=%.3f N=%d: NOT READ (%s)" % (mu, HELD_OUT, "; ".join(why)))
            status = 1
            continue
        c = cells[find(cells, mu, HELD_OUT)]
        wrong = held_out_misses(c, fixed["%.3f" % mu])
        print("HELD-OUT mu=%.3f N=%d: %s | verdict there: %s"
              % (mu, HELD_OUT, "AS PREDICTED" if not wrong else "NOT AS PREDICTED (%s)" % ", ".join(wrong),
                 verdict(c["majority"])))
    return status


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    paths = [a for a in args if not a.startswith("--")]
    sys.exit(main(paths[0] if paths else "results", flags[0] if flags else ""))
