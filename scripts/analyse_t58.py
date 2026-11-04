"""T58: T38's three long waits (lambda 1.30, N = 64), replayed from their seeds with nothing hidden. Usage:

    python scripts/analyse_t58.py [results_dir]

Reads the replay beside the original, results/t38_lam130_n64_*.csv, and applies PREREGISTRATION.md T58. The replay
has two stages, read apart: results/t58_targets_lam130_n64.csv, the three long waits and six neighbours written
block by block (seconds to run), and results/t58_lam130_n64_*.csv, the whole cell of 4,000 decays with the
detector's numbers in each row (about half an hour on sixteen cores).

  1. the reproduction gate, once per stage: every column the original wrote must come back unchanged for every
     decay, and the traced decays must end on their saved graphs; if not, nothing of that stage is read;
  2. one label per target, from its block-by-block trace over the blocks before the detector fired:
       SECOND CURL    the square count rose above the tube's, or at least 8 points read d = 0 (both directions
                      curled), or a closed piece with three squares on every edge existed   (the owner's prediction)
       LOW THRESHOLD  no second curl, and after sweep 200 the tube lost squares without the detector firing
                      (the assistant's prediction)
       TRUE WAIT      neither, and the tube read as the perfect tube in at least 95 % of the blocks after sweep 200
       OTHER STATE    anything else
  3. the cell-wide numbers: the first exit timed with no detector at all (the first block at which a decay is no
     longer the perfect tube, from sweep 0), against the count; and whether any decay ever had more squares than
     the tube or 8 points at d = 0.
"""
import csv
import glob
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_wait_detector import REST, read_cell, tau_count                     # noqa: E402

LAM, G, N = 1.30, 1.5, 64
S_TUBE, X_TUBE = 80, 64                          # the 16 x 4 torus: 1.25 squares and one surplus square per point
TARGETS = (553, 2748, 3072)
CONTROLS = (551, 552, 2746, 2747, 3070, 3071)
D0_CURL = 8                                      # points at d = 0 that count as a second curl (half a 4-cube)
TUBE_SHARE = 0.95                                # TRUE WAIT needs the perfect tube in this share of blocks
SINGLE_EXIT = 1.25 - 2.0 / N                     # phi after the cheapest exit, which loses two squares


def load(results, pattern):
    rows = {}
    for path in sorted(glob.glob(str(Path(results) / pattern))):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = Path(path).stem
                rows[int(r["replica"])] = r
    return rows


def load_trace(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: int(v) for k, v in line.items()} for line in csv.DictReader(fh)]


def reproduction(old, new):
    """Every (replica, column, old, new) at which the replay differs from the original. Empty means reproduced."""
    bad = []
    for rep in sorted(old):
        if rep not in new:
            bad.append((rep, "missing", "", ""))
            continue
        for key, value in old[rep].items():
            if key.startswith("_"):
                continue
            if not same(value, new[rep].get(key)):
                bad.append((rep, key, value, new[rep].get(key)))
    return bad


def same(a, b):
    """Equal as written; or, for two numbers, equal to one part in 10^9 (an average of floating-point numbers may
    differ in its last digit between machines; a count may not differ at all, and 10^-9 is far below one count)."""
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= 1e-9 * max(1.0, abs(x))


def is_tube(t):
    return t["S"] == S_TUBE and t["X"] == X_TUBE and t["d1"] == N


def energy_above_tube(t, lam=LAM):
    """H of a traced block minus H of the perfect tube, in the model's units."""
    return (16.0 * (N - t["S"]) + 4.0 * lam * t["X"]) - (16.0 * (N - S_TUBE) + 4.0 * lam * X_TUBE)


def classify_target(waiting, trace):
    """The pre-registered label of one traced decay, with the numbers behind it. `waiting` is the recorded
    detection sweep (None if the detector never fired)."""
    before = [t for t in trace if waiting is None or t["sweep"] < waiting]
    stretch = [t for t in before if t["sweep"] > REST]
    second = [t for t in before if t["S"] > S_TUBE or t["d0"] >= D0_CURL or t["baby"] > 0]
    off = [t for t in stretch if not is_tube(t)]
    lost = [t for t in off if t["S"] < S_TUBE]
    exits, was = 0, next((is_tube(t) for t in trace if t["sweep"] == REST), True)
    for t in stretch:
        exits += int(was and not is_tube(t))
        was = is_tube(t)
    share = (len(stretch) - len(off)) / len(stretch) if stretch else 1.0
    if second:
        label = "SECOND CURL"
    elif lost:
        label = "LOW THRESHOLD"
    elif share >= TUBE_SHARE:
        label = "TRUE WAIT"
    else:
        label = "OTHER STATE"
    left = next((t["sweep"] for t in trace if t["sweep"] > REST and not is_tube(t)), None)
    return dict(label=label, blocks=len(stretch), tube_share=share, unseen_exits=exits, off_blocks=len(off),
                lowest_s_unseen=min((t["S"] for t in off), default=None),
                most_pieces=max((t["pieces"] for t in before), default=None),
                highest_s=max(t["S"] for t in before) if before else None,
                most_d0=max(t["d0"] for t in before) if before else None,
                most_energy=max(energy_above_tube(t) for t in before) if before else None,
                second_blocks=len(second), zero_noise_wait=left)


def verdict(labels):
    """The verdict on the three targets from their labels."""
    if sum(1 for v in labels if v == "SECOND CURL") >= 2:
        return "SECOND CURL"
    if all(v == "LOW THRESHOLD" for v in labels):
        return "LOW THRESHOLD"
    if all(v == "TRUE WAIT" for v in labels):
        return "TRUE WAIT"
    return "MIXED"


def population(rows, tau):
    """The cell-wide numbers of step 3. `rows` are the replay's rows. The first exit is timed twice with no
    detector: from sweep 0 over every decay (the scored one; all start as the perfect tube), and from sweep 200 over
    the decays that read as the perfect tube when the watch ended (described)."""
    first = [float(r["first_left"]) for r in rows if r["first_left"] != ""]
    never = len(rows) - len(first)
    perfect = [r for r in rows if r["tube_at_200"] == "1"]
    left = [float(r["first_left_after_200"]) - REST for r in perfect if r["first_left_after_200"] != ""]
    never_left = len(perfect) - len(left)
    low = [r for r in perfect if r["thresh"] != "" and float(r["thresh"]) <= SINGLE_EXIT + 1e-9]
    plain = [r for r in perfect if r["thresh"] != "" and float(r["thresh"]) > SINGLE_EXIT + 1e-9]

    def mean_wait(group):
        w = [float(r["waiting"]) - REST for r in group if r["waiting"] != ""]
        return (statistics.mean(w), len(w)) if w else (float("nan"), 0)

    def scale(values):
        if len(values) < 2:
            return float("nan"), float("nan")
        return statistics.mean(values) / tau, statistics.stdev(values) / math.sqrt(len(values)) / tau

    return dict(
        n=len(rows), never=never, mean_first=statistics.mean(first) if first else float("nan"),
        ratio=scale(first)[0], se_ratio=scale(first)[1],
        beyond_10=sum(1 for v in first if v > 10.0 * tau) + never,
        expected_beyond_10=len(rows) * math.exp(-10.0), longest=max(first) if first else float("nan"),
        perfect=len(perfect), never_left=never_left,
        mean_left=statistics.mean(left) if left else float("nan"),
        ratio_left=scale(left)[0], se_ratio_left=scale(left)[1],
        beyond_10_left=sum(1 for v in left if v > 10.0 * tau) + never_left,
        expected_beyond_10_left=len(perfect) * math.exp(-10.0),
        longest_left=max(left) if left else float("nan"),
        low=len(low), plain=len(plain), wait_low=mean_wait(low), wait_plain=mean_wait(plain),
        above_tube=sum(1 for r in rows if float(r["phi_max"]) > 1.25 + 1e-9),
        d0_curl=sum(1 for r in rows if int(r["d0_max"]) >= D0_CURL),
        d0_any=sum(1 for r in rows if int(r["d0_max"]) > 0),
        d0_most=max(int(r["d0_max"]) for r in rows),
    )


def main(results="results"):
    old = load(results, "t38_lam130_n64_*.csv")
    targets = load(results, "t58_targets_lam130_n64.csv")
    new = load(results, "t58_lam130_n64_*.csv")
    tau = tau_count(LAM, G)
    print("T58: T38's cell at lambda %.2f, N = %d, replayed from its seeds\n" % (LAM, N))
    if not targets:
        print("no t58_targets_lam130_n64.csv under", results)
        return

    cell = read_cell(list(old.values()), LAM)
    cut = 10.0 * cell["tau_perfect"] + REST
    beyond = sorted(int(r["replica"]) for r in old.values()
                    if r["waiting"] and float(r["f_200"]) <= 1e-9 and float(r["waiting"]) > cut)
    print("The original's waits beyond ten times the perfect tubes' own scale (%.0f sweeps): decays %s%s\n"
          % (cut, beyond, "" if beyond == sorted(TARGETS) else "  (NOT the three targets: check)"))

    traced = TARGETS + CONTROLS
    bad = reproduction({rep: old[rep] for rep in traced}, targets)
    for rep in traced:
        if rep not in targets:
            continue
        saved = Path(results) / (old[rep]["_file"] + "_adj") / ("N%d_rep%d.npz" % (N, rep))
        again = Path(results) / (targets[rep]["_file"] + "_trace") / ("N%d_rep%d_final.npz" % (N, rep))
        if not np.array_equal(np.load(saved)["adj"], np.load(again)["adj"]):
            bad.append((rep, "final graph", "saved", "differs"))
    if bad:
        print("1. REPRODUCTION GATE, the traced decays: NOT REPRODUCED (%d differences). Nothing below is read."
              % len(bad))
        for item in bad[:10]:
            print("   rep %s, %s: original %r, replay %r" % item)
        return
    print("1. REPRODUCTION GATE, the traced decays: passed. All %d return every column of the original unchanged and"
          "\n   end on the graph the original saved.\n" % len(traced))

    print("2. THE THREE TARGETS (blocks before the detector fired; the tube has S = %d, X = %d; a knot of 16 points"
          "\n   pinched off it would read S = %d and cost %.1f units)"
          % (S_TUBE, X_TUBE, S_TUBE + 4, energy_above_tube(dict(S=S_TUBE + 4, X=X_TUBE + 16))))
    labels = []
    for rep in traced:
        r = targets[rep]
        trace = load_trace(Path(results) / (r["_file"] + "_trace") / ("N%d_rep%d.csv" % (N, rep)))
        c = classify_target(int(r["waiting"]) if r["waiting"] else None, trace)
        if rep in TARGETS:
            labels.append(c["label"])
        print("   %s rep %-5d recorded wait %6s | threshold phi < %.4f (resting spread %.4f; a single exit reads %.4f)"
              % ("target " if rep in TARGETS else "control", rep, r["waiting"] or "-", float(r["thresh"]),
                 float(r["rest_sd"]), SINGLE_EXIT))
        print("      first left the tube after sweep 200 at sweep %s; unseen exits %d; blocks away from the tube %d of"
              " %d (tube in %.1f %%); fewest squares unseen %s"
              % (c["zero_noise_wait"], c["unseen_exits"], c["off_blocks"], c["blocks"], 100.0 * c["tube_share"],
                 c["lowest_s_unseen"]))
        print("      most squares %s (tube %d); most points at d = 0: %s; most separate pieces: %s; most energy above"
              " the tube: %s; -> %s"
              % (c["highest_s"], S_TUBE, c["most_d0"], c["most_pieces"],
                 "-" if c["most_energy"] is None else "%.1f" % c["most_energy"], c["label"]))
    print("\n   VERDICT ON THE THREE: %s\n" % verdict(labels))

    files = sorted(set(r["_file"] for r in new.values()))
    if len(new) < len(old):
        print("3. THE CELL: the replay of the whole cell is not complete (%d of %d decays, %d files). Not read."
              % (len(new), len(old), len(files)))
        return
    bad = reproduction(old, new)
    if bad:
        print("3. REPRODUCTION GATE, the whole cell: NOT REPRODUCED (%d differences). The cell is not read."
              % len(bad))
        for item in bad[:10]:
            print("   rep %s, %s: original %r, replay %r" % item)
        return
    p = population(list(new.values()), tau)
    print("3. THE CELL. Reproduction gate passed: all %d decays return every column of the original unchanged."
          % len(old))
    print("   The count's mean wait for a first exit: %.1f sweeps. The first exit below is timed with no detector:"
          "\n   the first block at which a decay no longer reads as the perfect tube, from sweep 0, all %d decays."
          % (tau, p["n"]))
    print("   P1  mean first exit: %.1f sweeps = %.3f +- %.3f of the count  [predicted 0.95 to 1.10]"
          % (p["mean_first"], p["ratio"], p["se_ratio"]))
    print("   P2  first exits later than 10 of the count's tau: %d, of which never left: %d (one memoryless population"
          " expects %.2f;\n       longest %.0f)  [predicted at most 1; three or more and O109's reading is abandoned]"
          % (p["beyond_10"], p["never"], p["expected_beyond_10"], p["longest"]))
    print("   P3  decays whose square count ever exceeded the tube's: %d; with %d or more points at d = 0: %d"
          "  [predicted 0 and 0]; with any point at d = 0: %d (at most %d points)"
          % (p["above_tube"], D0_CURL, p["d0_curl"], p["d0_any"], p["d0_most"]))
    print("   Described, not scored:")
    print("   timed from sweep 200 instead, over the %d decays reading as the perfect tube then: mean %.1f sweeps ="
          " %.3f +- %.3f of the count;\n   later than 10 tau: %d (expected %.2f; longest %.0f)"
          % (p["perfect"], p["mean_left"], p["ratio_left"], p["se_ratio_left"], p["beyond_10_left"],
             p["expected_beyond_10_left"], p["longest_left"]))
    print("   of those, the detector's threshold sat at or below a single exit in %d, whose mean recorded wait is %.0f"
          " (n = %d);\n   in the other %d it is %.0f (n = %d)"
          % (p["low"], p["wait_low"][0], p["wait_low"][1], p["plain"], p["wait_plain"][0], p["wait_plain"][1]))


if __name__ == "__main__":
    main(*sys.argv[1:])
