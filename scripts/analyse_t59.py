"""T59: the long waits at lambda 1.05 read from their seeds, and fresh first exits timed by the move. Usage:

    python scripts/analyse_t59.py [results_dir]

Applies PREREGISTRATION.md T59 to what scripts/run_first_exits.py and scripts/run_tube_decay.py wrote. Four parts,
read apart and in this order; a part whose files are missing or whose gate fails is not read.

  A1  T8's decay 11 at N = 96 (a wait of 78,205 sweeps) and the two before it, replayed block by block. Gate: every
      column of the original row comes back unchanged and the replay ends on the saved graph. Label, by T58's rules
      with this size's numbers: SECOND CURL, LOW THRESHOLD, TRUE WAIT or OTHER STATE.
  A2  T22's seven first exits later than four mean waits and seven controls, replayed with what the chain was
      offered tallied. Gate: every first exit is the attempt T22 saved. Label per run: OFFERED AS COUNTED if in every
      whole stretch of 5,000 sweeps the exits of kind A offered are within 5 % of three a sweep and those of kind B
      within 5 % of two a sweep; STARVED if not.
  B   fresh tubes at lambda 1.05, N = 64 and 96, first exit counted move by move. P1: the mean is the count's within
      the registered band. P2: at most the registered number of first exits later than 8 mean waits. ON THE COUNT
      (both), SLOW TAIL (P2 fails), OFF THE COUNT (P1 fails, P2 holds).
  C   fresh tubes at lambda 1.30, N = 64, the same history on two clocks. P3: the move clock's mean is the count's
      within 5 %. P4: the look-every-five-sweeps clock runs later by 0.035 to 0.107 of the count. P5: more than half
      of that gap comes from tubes whose first exit came back before a look. HIDDEN EXITS (all three), OFF THE COUNT
      (P3 fails), NOT THE LOOKS (P3 holds, P4 fails), THE LOOK'S ROUNDING (P3 and P4 hold, P5 fails).
"""
import csv
import glob
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t58 import D0_CURL, TUBE_SHARE, load_trace, same             # noqa: E402
from read_wait_detector import REST, poisson_tail, tau_count             # noqa: E402

G = 1.5
STRETCH = 5000
OFFER_BAND = 0.05                                 # A2: offers per sweep within this share of 3 (A) and 2 (B)
A_PER_SWEEP, B_PER_SWEEP = 3.0, 2.0               # paper 1, Eq. (2); exact at every member of the torus's family
T8_TARGET, T8_CONTROLS = 11, (9, 10)
T22_TARGETS = {64: (6, 7, 11, 17), 96: (6, 12, 29)}
T22_CONTROLS = {64: (5, 8, 12, 18), 96: (7, 13, 30)}
B_CELLS = {64: dict(tubes=2000, band=0.07, most_beyond=4), 96: dict(tubes=1000, band=0.10, most_beyond=3)}
B_LAM, B_BEYOND = 1.05, 8.0
C_LAM, C_N, C_TUBES, C_BAND = 1.30, 64, 4000, 0.05
C_GAP = (0.035, 0.107)                            # T58's excess, 0.071 +- 0.017, with this run's own error: two sigma
C_HIDDEN_SHARE = 0.5


def cost_a(lam):
    return 32.0 - 16.0 * lam


def cost_b(lam):
    return 64.0 - 40.0 * lam


def rows_of(results, pattern):
    out = []
    for path in sorted(glob.glob(str(Path(results) / pattern))):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = Path(path).stem
                out.append(r)
    return out


# ---- A1: the decay runner's long wait, by T58's rules at any size -------------------------------------------------

def classify_wait(waiting, trace, n):
    """T58's label for one traced decay (scripts/analyse_t58.py, classify_target) with the tube of N = n points:
    S = 5n/4, X = n, every point at d = 1. `waiting` is the recorded detection sweep (None if it never fired)."""
    s_tube, x_tube = 5 * n // 4, n

    def is_tube(t):
        return t["S"] == s_tube and t["X"] == x_tube and t["d1"] == n

    before = [t for t in trace if waiting is None or t["sweep"] < waiting]
    stretch = [t for t in before if t["sweep"] > REST]
    second = [t for t in before if t["S"] > s_tube or t["d0"] >= D0_CURL or t["baby"] > 0]
    off = [t for t in stretch if not is_tube(t)]
    lost = [t for t in off if t["S"] < s_tube]
    share = (len(stretch) - len(off)) / len(stretch) if stretch else 1.0
    if second:
        label = "SECOND CURL"
    elif lost:
        label = "LOW THRESHOLD"
    elif share >= TUBE_SHARE:
        label = "TRUE WAIT"
    else:
        label = "OTHER STATE"
    left = next((t["sweep"] for t in trace if not is_tube(t)), None)
    return dict(label=label, blocks=len(stretch), tube_share=share, off_blocks=len(off), first_left=left,
                highest_s=max((t["S"] for t in before), default=None),
                most_d0=max((t["d0"] for t in before), default=None))


def stage_a1(results):
    new = {int(r["replica"]): r for r in rows_of(results, "t59_t8wait_lam105_n96.csv")}
    if not new:
        print("A1. no t59_t8wait_lam105_n96.csv under %s. Not read.\n" % results)
        return
    old = {int(r["replica"]): r for r in rows_of(results, "t8_lam105.csv") if r["N"] == "96"}
    ids = sorted((T8_TARGET,) + T8_CONTROLS)
    bad = []
    for rep in ids:
        if rep not in new:
            bad.append((rep, "missing", "", ""))
            continue
        bad += [(rep, k, v, new[rep].get(k)) for k, v in old[rep].items()
                if not k.startswith("_") and not same(v, new[rep].get(k))]
        saved = Path(results) / "t8_lam105_adj" / ("N96_rep%d.npz" % rep)
        again = Path(results) / "t59_t8wait_lam105_n96_trace" / ("N96_rep%d_final.npz" % rep)
        if not np.array_equal(np.load(saved)["adj"], np.load(again)["adj"]):
            bad.append((rep, "final graph", "saved", "differs"))
    if bad:
        print("A1. REPRODUCTION GATE: NOT REPRODUCED (%d differences). Not read." % len(bad))
        for item in bad[:10]:
            print("    rep %s, %s: original %r, replay %r" % item)
        print()
        return
    tau = tau_count(B_LAM, G)
    print("A1. T8's long wait at lambda %.2f, N = 96, replayed from its seed. Gate passed: all %d decays return every"
          "\n    column of the original unchanged and end on the saved graph. The count's mean wait is %.0f sweeps."
          % (B_LAM, len(ids), tau))
    for rep in ids:
        r = new[rep]
        trace = load_trace(Path(results) / "t59_t8wait_lam105_n96_trace" / ("N96_rep%d.csv" % rep))
        c = classify_wait(int(r["waiting"]) if r["waiting"] else None, trace, 96)
        print("    %s rep %-2d recorded wait %6s (%.1f mean waits) | first read as other than the tube at sweep %s |"
              " tube in %.1f %% of %d blocks | most squares %s (tube 120), most points at d = 0: %s -> %s"
              % ("target " if rep == T8_TARGET else "control", rep, r["waiting"] or "-",
                 (float(r["waiting"]) / tau) if r["waiting"] else float("nan"), c["first_left"],
                 100.0 * c["tube_share"], c["blocks"], c["highest_s"], c["most_d0"], c["label"]))
    print()


# ---- A2: what T22's long waits were offered ------------------------------------------------------------------------

def offers_label(stretches):
    """OFFERED AS COUNTED or STARVED, from the whole stretches of one wait (a list of dicts with sweeps_in_stretch,
    offered_a, offered_b). A wait shorter than one whole stretch has nothing to score: NO WHOLE STRETCH."""
    whole = [s for s in stretches if int(s["whole_stretch"])]
    if not whole:
        return "NO WHOLE STRETCH"
    for s in whole:
        sweeps = float(s["sweeps_in_stretch"])
        if abs(int(s["offered_a"]) / sweeps - A_PER_SWEEP) > OFFER_BAND * A_PER_SWEEP:
            return "STARVED"
        if abs(int(s["offered_b"]) / sweeps - B_PER_SWEEP) > OFFER_BAND * B_PER_SWEEP:
            return "STARVED"
    return "OFFERED AS COUNTED"


def expected_taken(stretches, lam, g=G):
    """Exits of kinds A and B that independent draws would have taken from the offers made."""
    return (sum(int(s["offered_a"]) for s in stretches) * math.exp(-cost_a(lam) / g)
            + sum(int(s["offered_b"]) for s in stretches) * math.exp(-cost_b(lam) / g))


def stage_a2(results):
    rows = rows_of(results, "t59_offers_lam105.csv")
    if not rows:
        print("A2. no t59_offers_lam105.csv under %s. Not read.\n" % results)
        return
    runs = {}
    for r in rows:
        runs.setdefault((int(r["N"]), int(r["replica"])), []).append(r)
    saved = {(int(r["N"]), int(r["replica"])): r["first_exit_sweeps"]
             for n in (64, 96) for r in rows_of(results, "t22_exits_n%d_lam105.csv" % n)}
    wanted = [(n, rep) for n in (64, 96) for rep in T22_TARGETS[n] + T22_CONTROLS[n]]
    bad = [(key, saved[key], runs[key][0]["first_exit_sweeps"] if key in runs else "missing") for key in wanted
           if key not in runs or not same(saved[key], runs[key][0]["first_exit_sweeps"])]
    if bad:
        print("A2. REPRODUCTION GATE: NOT REPRODUCED (%d of %d first exits differ). Not read." % (len(bad), len(wanted)))
        for item in bad[:10]:
            print("    N, replica %s: T22 saved %s, replay %s" % item)
        print()
        return
    p_a = math.exp(-cost_a(B_LAM) / G)
    print("A2. T22's long first exits at lambda %.2f, replayed from their seeds. Gate passed: all %d first exits are"
          "\n    the attempt T22 saved. Counted: %.0f exits of kind A and %.0f of kind B offered per sweep; a kind-A"
          "\n    offer is taken with chance %.3g. Whole stretches are %d sweeps."
          % (B_LAM, len(wanted), A_PER_SWEEP, B_PER_SWEEP, p_a, STRETCH))
    labels = []
    for n in (64, 96):
        tau = tau_count(B_LAM, G)
        for rep in T22_TARGETS[n] + T22_CONTROLS[n]:
            s = sorted(runs[(n, rep)], key=lambda r: int(r["stretch"]))
            whole = [x for x in s if int(x["whole_stretch"])]
            per_a = [int(x["offered_a"]) / float(x["sweeps_in_stretch"]) for x in whole]
            per_b = [int(x["offered_b"]) / float(x["sweeps_in_stretch"]) for x in whole]
            label = offers_label(s)
            target = rep in T22_TARGETS[n]
            if target:
                labels.append(label)
            taken = expected_taken(s, B_LAM)
            wait = float(s[0]["first_exit_sweeps"])
            print("    %s N=%d rep %-2d first exit at %8.0f sweeps (%.2f mean waits) | A per sweep %s, B per sweep %s"
                  " over %d whole stretches | offers made would take %.2f exits; chance of none before the last: %.2g"
                  " | smallest draw against A: %.3g -> %s"
                  % ("target " if target else "control", n, rep, wait, wait / tau,
                     ("%.2f to %.2f" % (min(per_a), max(per_a))) if per_a else "-",
                     ("%.2f to %.2f" % (min(per_b), max(per_b))) if per_b else "-", len(whole), taken,
                     math.exp(-max(taken - p_a, 0.0)), min(float(x["smallest_draw_a"]) for x in s), label))
    scored = [v for v in labels if v != "NO WHOLE STRETCH"]
    print("    VERDICT ON THE SEVEN: %s\n"
          % ("OFFERED AS COUNTED" if scored and all(v == "OFFERED AS COUNTED" for v in scored)
             else "STARVED (%d of %d)" % (sum(v == "STARVED" for v in scored), len(scored))))


# ---- B and C: fresh tubes on the move clock and the look clock ---------------------------------------------------

def clock_numbers(rows, tau, cap):
    """Means, in units of the count's mean wait `tau`, over one cell's tubes. A tube that never left by the cap
    enters the mean at the cap, which is a lower bound, and counts as beyond every multiple of tau."""
    move = [float(r["first_exit_sweeps"]) if r["first_exit_sweeps"] != "" else float(cap) for r in rows]
    never = sum(1 for r in rows if r["first_exit_sweeps"] == "")
    seen = [r for r in rows if r["seen_every_look"] != "" and r["first_exit_sweeps"] != ""]
    gap = [float(r["seen_every_look"]) - float(r["first_exit_sweeps"]) for r in seen]
    hidden = [g if int(r["exits_before_look"]) > 1 else 0.0 for g, r in zip(gap, seen)]
    gap_1 = [float(r["seen_every_sweep"]) - float(r["first_exit_sweeps"]) for r in seen]
    n = len(move)

    def mean_se(v):
        return (statistics.mean(v) / tau, statistics.stdev(v) / math.sqrt(len(v)) / tau) if len(v) > 1 else (math.nan,) * 2

    return dict(n=n, never=never, move=mean_se(move), look=mean_se([float(r["seen_every_look"]) for r in seen]),
                gap=mean_se(gap), gap_sweep_look=mean_se(gap_1),
                hidden_share=(sum(hidden) / sum(gap)) if gap and sum(gap) > 0 else math.nan,
                tubes_hidden=sum(1 for r in seen if int(r["exits_before_look"]) > 1), seen=len(seen),
                tail={k: (sum(1 for v in move if v > k * tau) , n * math.exp(-k)) for k in (4, 6, 8, 10)},
                longest=max(move) / tau if move else math.nan)


def verdict_b(numbers, band, most_beyond):
    p1 = abs(numbers["move"][0] - 1.0) <= band
    p2 = numbers["tail"][8][0] <= most_beyond
    return p1, p2, ("ON THE COUNT" if p1 and p2 else "SLOW TAIL" if not p2 else "OFF THE COUNT")


def verdict_c(numbers):
    p3 = abs(numbers["move"][0] - 1.0) <= C_BAND
    p4 = C_GAP[0] <= numbers["gap"][0] <= C_GAP[1]
    p5 = numbers["hidden_share"] > C_HIDDEN_SHARE
    if not p3:
        label = "OFF THE COUNT"
    elif not p4:
        label = "NOT THE LOOKS"
    elif not p5:
        label = "THE LOOK'S ROUNDING"
    else:
        label = "HIDDEN EXITS"
    return p3, p4, p5, label


def held(flag):
    return "holds" if flag else "FAILS"


def describe(numbers):
    t = numbers["tail"]
    print("        later than 4, 6, 8, 10 mean waits: %d, %d, %d, %d (one memoryless population expects %.1f, %.1f,"
          " %.2f, %.2f); longest %.1f; never left by the cap: %d"
          % (t[4][0], t[6][0], t[8][0], t[10][0], t[4][1], t[6][1], t[8][1], t[10][1], numbers["longest"],
             numbers["never"]))
    print("        the same tubes on a look every five sweeps: mean %.3f +- %.3f of the count; later than the move"
          " clock by %.4f +- %.4f\n        (a look every sweep: %.4f +- %.4f); tubes whose first exit came back before a"
          " look: %d of %d, carrying %.0f %% of the gap"
          % (numbers["look"] + numbers["gap"] + numbers["gap_sweep_look"]
             + (numbers["tubes_hidden"], numbers["seen"], 100.0 * numbers["hidden_share"])))


def stage_b(results):
    tau = tau_count(B_LAM, G)
    print("B.  Fresh tubes at lambda %.2f, first exit counted move by move. The count's mean wait: %.1f sweeps."
          % (B_LAM, tau))
    for n, cell in B_CELLS.items():
        rows = rows_of(results, "t59_lam105_n%d_*.csv" % n)
        if len(rows) < cell["tubes"]:
            print("    N = %d: %d of %d tubes. Not complete; not read." % (n, len(rows), cell["tubes"]))
            continue
        m = clock_numbers(rows, tau, 200000)
        p1, p2, label = verdict_b(m, cell["band"], cell["most_beyond"])
        print("    N = %d, %d tubes -> %s" % (n, m["n"], label))
        print("        P1 mean first exit %.3f +- %.3f of the count  [registered: within %.2f of 1]  %s"
              % (m["move"] + (cell["band"], held(p1))))
        print("        P2 first exits later than %.0f mean waits: %d  [registered: at most %d; expected %.2f, and a"
              " chance of %.1g of more than %d]  %s"
              % (B_BEYOND, m["tail"][8][0], cell["most_beyond"], m["tail"][8][1],
                 poisson_tail(cell["most_beyond"] + 1, m["tail"][8][1]), cell["most_beyond"], held(p2)))
        describe(m)
    print()


def stage_c(results):
    tau = tau_count(C_LAM, G)
    rows = rows_of(results, "t59_lam130_n64_*.csv")
    print("C.  Fresh tubes at lambda %.2f, N = %d, one history on two clocks. The count's mean wait: %.1f sweeps."
          % (C_LAM, C_N, tau))
    if len(rows) < C_TUBES:
        print("    %d of %d tubes. Not complete; not read.\n" % (len(rows), C_TUBES))
        return
    m = clock_numbers(rows, tau, 50000)
    p3, p4, p5, label = verdict_c(m)
    print("    %d tubes -> %s" % (m["n"], label))
    print("        P3 mean first exit by the move: %.3f +- %.3f of the count  [registered: within %.2f of 1]  %s"
          % (m["move"] + (C_BAND, held(p3))))
    print("        P4 the look every five sweeps runs later by %.4f +- %.4f of the count  [registered: %.3f to %.3f;"
          " T58 found 0.071 +- 0.017 above the count]  %s" % (m["gap"] + C_GAP + (held(p4),)))
    print("        P5 share of that gap from tubes whose first exit came back before a look: %.0f %%  [registered:"
          " more than %.0f %%]  %s" % (100.0 * m["hidden_share"], 100.0 * C_HIDDEN_SHARE, held(p5)))
    describe(m)
    print()


def main(results="results"):
    print("T59: the long waits at lambda 1.05, and first exits timed by the move\n")
    stage_a1(results)
    stage_a2(results)
    stage_b(results)
    stage_c(results)


if __name__ == "__main__":
    main(*sys.argv[1:])
