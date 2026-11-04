"""The registered T64 reading: T63 again on fresh seeds, with the control's rule rewritten before the run. Usage:

    python scripts/analyse_t64.py [results_dir]                     # the report
    python scripts/analyse_t64.py [results_dir] --write-prediction  # after stage 1: the numbers for N = 384

Implements PREREGISTRATION.md section T64 on results/t64_mu<tag>_n<N>.csv (scripts/run_sixloop_widen.py with
"start": "tube"). Everything is T63's (scripts/analyse_t63.py: the outcome of a replica, the cell, its gates, the
words, the held-out rule) except the control.

The control, as T64 fixes it. At mu = 0 the run is paper 1's own. Paper 1's record at this setting (lambda = 1.25,
g = 1.5; T24 and T60, 120 decays a size) is that 0.80 to 0.91 of the tubes end exactly flat, and that the exactly
flat end states are six around in 98 of 100; the rest come to rest on a state with a leftover. So the control
holds if
    at least 0.6 of its 24 replicas end exactly flat and TUBE (15 or more), and
    at least nine tenths of its replicas that end exactly flat are TUBE.
T63 asked instead for nine tenths of the converted replicas to end TUBE, which paper 1's own maps do not deliver.
"""
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t63 as t63                                                          # noqa: E402
from analyse_t62 import geometry, held_out_misses                                  # noqa: E402

STAGE1, HELD_OUT, MUS = t63.STAGE1, t63.HELD_OUT, t63.MUS
CONTROL_FLAT_TUBE = 0.6          # of all the control's replicas
CONTROL_FLAT_ARE_TUBE = 0.9      # of the control's replicas that end exactly flat
PREDICTION = Path(__file__).resolve().parents[1] / "configs" / "t64_heldout_prediction.json"


def control_counts(rows):
    """(replicas, exactly flat, exactly flat and TUBE) of a control cell."""
    flat = [r for r in rows if r["flat"] == "True" and t63.outcome(r) != "NOT CONVERTED"]
    return len(rows), len(flat), sum(t63.outcome(r) == "TUBE" for r in flat)


def control_fails(groups, cells, n):
    why = t63.design_problems(groups, (n,), (0.0,))
    key = t63.find(cells, 0.0, n)
    if key is None:
        return why or ["the control at N=%d is missing or unreadable" % n]
    if not cells[key]["gate"]:
        why.append("the control fails its gates")
    total, flat, flat_tube = control_counts(groups[key])
    if flat_tube < CONTROL_FLAT_TUBE * total:
        why.append("the control: only %d of %d end exactly flat and six around" % (flat_tube, total))
    if flat_tube < CONTROL_FLAT_ARE_TUBE * flat:
        why.append("the control: only %d of its %d exactly flat end states are six around" % (flat_tube, flat))
    return why


def blocked(groups, cells, mu, n):
    """Why no word may be given at (mu, N): a list, empty if it may."""
    why = t63.design_problems(groups, (n,), (mu,))
    key = t63.find(cells, mu, n)
    if key is not None and not cells[key]["gate"]:
        why.append("its gates fail")
    if mu > 0.0:
        why += control_fails(groups, cells, n)
    return why


def load(out_dir="results"):
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(Path(out_dir) / "t64_mu*_n*.csv"))):
        if not re.search(r"t64_mu\d+_n\d+\.csv$", f):
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = Path(f).stem
                groups[(float(r["mu"]), int(r["N"]))].append(r)
    return groups


def main(out_dir="results", flag=""):
    groups = load(out_dir)
    if not groups:
        print("no t64_mu*_n*.csv in %s" % out_dir)
        return 1
    cells = {k: t63.cell(v) for k, v in groups.items() if t63.readable(v)}
    print("T64: T63 again on fresh seeds: what does the curled tube itself open into? (PREREGISTRATION T64)")
    for key in sorted(groups, key=lambda k: (k[1], k[0])):
        mu, n = key
        if key not in cells:
            print("mu=%.3f N=%-3d UNREADABLE: a number is missing or not finite" % key)
            continue
        c = cells[key]
        geo = geometry(out_dir, [r for r in groups[key] if t63.outcome(r) != "NOT CONVERTED"])
        print("mu=%.3f N=%-3d gate %s | %s -> %-9s | wide %.2f tube %.2f | curl gone at sweep, median %.0f | energy left, "
              "median %.1f | six-loop excess left, median %.2f | fewest squares per point %.2f | exactly flat %d, of those "
              "no excess %d%s"
              % (mu, n, "pass" if c["gate"] else "FAIL", c["outcomes"], c["majority"], c["wide"], c["tube"],
                 c["curl_gone"], c["h_end"], c["e6_left"], c["phi_min"], c["flat"], c["flat_no_excess"],
                 "" if geo is None else " | from the wiring: in pieces %d, clean and wide %d, one-sided %d" % geo))
    for n in STAGE1:
        key = t63.find(cells, 0.0, n)
        why = control_fails(groups, cells, n)
        counts = "" if key is None else " | %d replicas, %d exactly flat, %d exactly flat and six around" % control_counts(
            groups[key])
        print("CONTROL N=%d: %s%s" % (n, "holds" if not why else "FAILS (%s)" % "; ".join(why), counts))
        for mu in MUS:
            why = blocked(groups, cells, mu, n)
            key = t63.find(cells, mu, n)
            print("VERDICT mu=%.3f N=%d: %s" % (mu, n, "NOT READ (%s)" % "; ".join(why) if why
                                                 else t63.verdict(cells[key]["majority"])))
    stray = t63.strays(groups)
    for line in stray:
        print("NOTE:", line)
    if flag == "--write-prediction":
        if PREDICTION.exists():
            raise SystemExit("refusing to overwrite %s" % PREDICTION)
        if (t63.design_problems(groups, STAGE1) or stray
                or not all(c["gate"] for k, c in cells.items() if k[1] in STAGE1)
                or any(control_fails(groups, cells, n) for n in STAGE1)):
            raise SystemExit("stage 1 is not complete and clean, or a control fails; nothing written")
        recipe = t63.predict(cells)
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
        c = cells[t63.find(cells, mu, HELD_OUT)]
        wrong = held_out_misses(c, fixed["%.3f" % mu])
        print("HELD-OUT mu=%.3f N=%d: %s | verdict there: %s"
              % (mu, HELD_OUT, "AS PREDICTED" if not wrong else "NOT AS PREDICTED (%s)" % ", ".join(wrong),
                 t63.verdict(c["majority"])))
    return status


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    paths = [a for a in args if not a.startswith("--")]
    sys.exit(main(paths[0] if paths else "results", flags[0] if flags else ""))
