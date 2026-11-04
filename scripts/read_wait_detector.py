"""What the saved decay rows already say about the long waits and the missed detections. Usage:

    python scripts/read_wait_detector.py [results_dir]

EXPLORATORY and retrospective: reads results/ only, runs nothing, changes no verdict (ASSUMPTIONS O109, written
2026-10-10 after the owner's remark that the detector explanation of paper 1's long waits is testable).

The decay runner (`scripts/run_tube_decay.py`) records three clocks per tube:

  waiting   the detector's: the first sweep after 200 at which phi falls below PHI_TUBE - 3 sd, where sd is the
            spread of phi over the tube's own first 200 sweeps (the "watch");
  f_200     how far the tube had already converted when the watch ended (0 = still exactly the tube);
  sweep_25  the first sweep after 200 at which a quarter of the tube had converted (also 50 and 75).

The last two do not depend on the detector's threshold, so they can say whether a long `waiting`, or a missing
one, is a tube that sat as a tube or a tube that had already changed while the detector was still watching.

For each T38 cell this prints: how many tubes had begun to change by sweep 200, against the counted chance that a
first exit has happened by then; where the never-detected tubes were at sweep 200; the pre-registered yardstick
tau_hat beside the scale of the tubes that were still perfect at sweep 200; and the waits beyond 10 tau_hat split
into tubes that had begun by sweep 200 and tubes that had not, with what one memoryless population expects of the
second kind. Then T24's longest waits, read the same way.
"""
import csv
import glob
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REST = 200                      # the detector's watch, in sweeps
G = 1.5                         # every run read here was at this coupling
T38_NAME = re.compile(r"t38_lam(\d+)_n(\d+)_\d+\.csv$")
T24_NAME = re.compile(r"t24_lam(\d+)_n(\d+)\.csv$")


def tau_count(lam, g=G):
    """Paper 1's counted mean wait for the first exit: three exits costing 32 - 16 lam and two costing 64 - 40 lam
    offered per sweep (ASSUMPTIONS Q13; paper 1 Eq. (2))."""
    return 1.0 / (3.0 * math.exp(-(32.0 - 16.0 * lam) / g) + 2.0 * math.exp(-(64.0 - 40.0 * lam) / g))


def poisson_tail(k, mu):
    """P(X >= k) for a Poisson count of mean mu."""
    return 1.0 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(k))


def classify(row):
    """One of: 'never' (the detector never fired), 'begun' (the tube had already left the perfect tube when the
    watch ended, f_200 > 0), 'perfect' (it was still exactly the tube at sweep 200)."""
    if not row["waiting"]:
        return "never"
    return "begun" if float(row["f_200"]) > 1e-9 else "perfect"


def read_cell(rows, lam):
    """The numbers for one cell. `rows` are csv dict rows of one (lambda, N)."""
    kinds = defaultdict(list)
    for r in rows:
        kinds[classify(r)].append(r)
    fired = kinds["begun"] + kinds["perfect"]
    w_all = [float(r["waiting"]) - REST for r in fired]
    tau_hat = statistics.median(w_all) / math.log(2)              # the pre-registered yardstick (T38)
    cut = 10.0 * tau_hat
    w_perfect = [float(r["waiting"]) - REST for r in kinds["perfect"]]
    tau_perfect = statistics.mean(w_perfect)
    begun_all = [r for r in rows if float(r["f_200"]) > 1e-9]     # includes the never-detected
    long_rows = [r for r in fired if float(r["waiting"]) - REST > cut]
    long_perfect = [r for r in long_rows if classify(r) == "perfect"]
    fast = [r for r in fired if float(r["waiting"]) - REST <= 20]
    never = kinds["never"]
    return dict(
        n=len(rows), fired=len(fired), never=len(never),
        begun=len(begun_all), begun_expected=1.0 - math.exp(-REST / tau_count(lam)),
        never_min_f200=min((float(r["f_200"]) for r in never), default=None),
        never_quarter_at_first_check=sum(1 for r in never if r["sweep_25"] and float(r["sweep_25"]) <= REST + 5),
        fast=len(fast), fast_begun=sum(1 for r in fast if float(r["f_200"]) > 1e-9),
        tau_hat=tau_hat, tau_perfect=tau_perfect, tau_perfect_median=statistics.median(w_perfect) / math.log(2),
        tau_count=tau_count(lam), cut=cut,
        long=len(long_rows), long_begun=len(long_rows) - len(long_perfect), long_perfect=len(long_perfect),
        perfect=len(w_perfect),
        expected_long_perfect=len(w_perfect) * math.exp(-cut / tau_perfect),
        expected_preregistered=len(fired) * math.exp(-10.0),
        long_own=sum(1 for v in w_perfect if v > 10.0 * tau_perfect),
        expected_long_own=len(w_perfect) * math.exp(-10.0),
        long_rows=sorted(long_rows, key=lambda r: -float(r["waiting"])),
    )


def load(results, name):
    cells = defaultdict(list)
    for path in sorted(glob.glob(str(Path(results) / (name.pattern.split("_")[0] + "_*.csv")))):
        m = name.search(path.replace("\\", "/"))
        if not m:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            cells[(int(m.group(1)) / 100.0, int(m.group(2)))].extend(csv.DictReader(fh))
    return cells


def mark(v):
    return "-" if v in ("", None) else "%d" % float(v)


def main(results="results"):
    t38 = load(results, T38_NAME)
    if not t38:
        print("no t38_*.csv under", results)
        return
    print("T38, read retrospectively (exploratory; the pre-registered verdict TWO POPULATIONS stands as scored)\n")
    for (lam, n), rows in sorted(t38.items()):
        c = read_cell(rows, lam)
        print("lambda %.2f, N = %d: %d decays" % (lam, n, c["n"]))
        print("  begun by sweep 200 (f_200 > 0): %d (%.1f %%); the count puts a first exit before sweep 200 in %.1f %%"
              % (c["begun"], 100.0 * c["begun"] / c["n"], 100.0 * c["begun_expected"]))
        if c["never"]:
            print("  never detected: %d; the least converted of them at sweep 200: f_200 = %.2f; quarter mark at the"
                  " first check after the watch in %d of %d"
                  % (c["never"], c["never_min_f200"], c["never_quarter_at_first_check"], c["never"]))
        else:
            print("  never detected: 0")
        print("  detected within 20 sweeps of the watch's end: %d (%.1f %%), of which begun by sweep 200: %d (%.0f %%)"
              % (c["fast"], 100.0 * c["fast"] / c["n"], c["fast_begun"], 100.0 * c["fast_begun"] / max(1, c["fast"])))
        print("  yardstick: tau_hat %.0f (pre-registered, all waits); tubes perfect at sweep 200: mean %.0f, median/ln 2"
              " %.0f; the count %.0f" % (c["tau_hat"], c["tau_perfect"], c["tau_perfect_median"], c["tau_count"]))
        print("  waits beyond 10 tau_hat (%.0f sweeps, %.1f of the count's tau): %d = %d begun by sweep 200 + %d perfect"
              " at sweep 200" % (c["cut"], c["cut"] / c["tau_count"], c["long"], c["long_begun"], c["long_perfect"]))
        print("    one memoryless population with the perfect tubes' own mean expects %.2f of the second kind"
              " (chance of %d or more: %.3f); the pre-registered expectation was %.2f"
              % (c["expected_long_perfect"], c["long_perfect"],
                 poisson_tail(c["long_perfect"], c["expected_long_perfect"]), c["expected_preregistered"]))
        print("    measured against the perfect tubes' own scale: %d waits beyond 10 of it, where one population"
              " expects %.2f" % (c["long_own"], c["expected_long_own"]))
        for r in c["long_rows"]:
            print("    rep %-5s waiting %6d | f_200 %.3f | quarter %s half %s three-quarters %s | %s"
                  % (r["replica"], float(r["waiting"]), float(r["f_200"]), mark(r["sweep_25"]), mark(r["sweep_50"]),
                     mark(r["sweep_75"]), "begun by sweep 200" if classify(r) == "begun" else "perfect at sweep 200"))
        print()

    t24 = load(results, T24_NAME)
    print("T24: the longest wait in each cell at lambda = 1.25 and 1.30, and where that tube was at sweep 200")
    for (lam, n), rows in sorted(t24.items()):
        if lam not in (1.25, 1.30):
            continue
        fired = [r for r in rows if r["waiting"]]
        r = max(fired, key=lambda q: float(q["waiting"]))
        print("  lambda %.2f N %3d rep %-4s waiting %6d (%.1f of the count's tau) | f_200 %s | quarter %s | released"
              " %.3f of %.3f" % (lam, n, r["replica"], float(r["waiting"]), (float(r["waiting"]) - REST) / tau_count(lam),
                                 r.get("f_200", "?"), mark(r["sweep_25"]), float(r["released"]), float(r["expected"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
