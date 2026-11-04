"""Two statistical checks for the curled-torus paper (docs/papers/curled_torus/). Usage:

    python scripts/analyse_paper_stats.py

NOT PRE-REGISTERED. Asked for in a review of paper 1 generated with ChatGPT (OpenAI; 2026-09-24), run on data already
committed, with no new simulation; ASSUMPTIONS O42. Prints results; does not write files.
The pooled first-exit bootstrap treats cells independently, but the exploratory
Arrhenius data reuse seeds across couplings. Its p-value is nominal (O108).

1. Are the waiting times exponential? The pre-registered test of "memoryless" was the coefficient
   of variation (PREREGISTRATION T7, T8), a coarse test. Here the whole distribution is compared with
   an exponential. Every time is divided by the mean of its own condition, and the pooled, rescaled
   times are compared with the unit exponential by the Kolmogorov-Smirnov distance D. Because each
   condition's mean comes from the same data, the textbook KS p-value does not apply (the Lilliefors
   problem). The p-value here comes from a parametric bootstrap that repeats the whole procedure:
   same conditions, same counts, exponential times with each condition's measured mean, recorded the
   way the runner records them, rescaled by their own means, 10,000 times. Sets:
   (i)  first exits of the perfect torus, recorded with no watch: results/cqg_tube_arrhenius_lam125.csv
        (to the sweep; twelve conditions, sixteen each) and results/t22_exits_n*_lam*.csv (to the attempt;
        six cells, forty each). T22's times are also compared with Eq. (2)'s mean itself, with nothing
        estimated, by the plain KS test.
   (ii) the pre-registered waiting times of T7 (results/t7_ and t7b_lam125_n*.csv) and of T8
        (results/t8_lam*.csv, stuck cells only, as the paper defines stuck). These are read in blocks of
        five sweeps and only after a 200-sweep watch, so the first possible value is 205. If the wait is
        a true exponential event clock, its residual time conditional on survival to 205 is exponential
        with the same mean. The recorded W is a threshold detector, however, not that event clock;
        subtracting 205 is a diagnostic approximation, not a correction for its full selection rule.
        The test uses W - 205 for every W > 205. A torus that had already begun
        converting by sweep 200 is detected late by construction (its resting spread widens the
        threshold) and piles up just after 205; T8 records how far it had gone at sweep 200 (f_200), so
        T8 is also tested with f_200 = 0 only. T7 predates that column.
   (iii) the far tail at lambda = 1.05: how many of the waits there exceed 75,000 sweeps, against the
        number an exponential with the measured mean gives (Poisson).

2. Is the relic's position along the torus independent of where the change started? T11 applied the
   pre-registered shares (at the far side, at the start). The named test here is a randomisation test
   of the mean distance: each replica's leftover is rotated through every column of the torus with the
   start held fixed, which is what "the leftover lands anywhere, whatever the start" means, and the
   observed mean distance is placed in that null distribution (100,000 draws, two-sided).
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t11 as a11            # noqa: E402

N_BOOT = 10_000
N_PERM = 100_000
WATCH = 205                          # first check after the 200-sweep watch (run_tube_decay.py)
BLOCK = 5                            # sweeps between checks
FAR = 75_000                         # sweeps: "the far tail" at lambda = 1.05


def tau_eq2(lam, g):
    """Eq. (2) of the paper: mean sweeps to the first exit of the perfect 4 x L torus."""
    return 1.0 / (3 * math.exp(-(32 - 16 * lam) / g) + 2 * math.exp(-(64 - 40 * lam) / g))


def ks_to_unit_exponential(x):
    """Kolmogorov-Smirnov distance between the sample x and the exponential of mean 1."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cdf = 1.0 - np.exp(-x)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)))


def pooled_rescaled(groups):
    """Every time divided by the mean of its own group, all groups pooled."""
    return np.concatenate([np.asarray(g, float) / np.mean(g) for g in groups])


def record(sample, how):
    """Exponential draws recorded the way the data were: 'exact', 'sweep' (the first whole sweep
    after the event), or 'block' (the time beyond 205, in whole blocks of five)."""
    if how == "sweep":
        return np.ceil(sample)
    if how == "block":
        return BLOCK * np.ceil(sample / BLOCK)
    return sample


def bootstrap_p(groups, hows, n_boot=N_BOOT, seed=0):
    """(D, p): D of the pooled rescaled data, and the share of simulated data sets whose D is at least
    as large, each simulated with every group's own measured mean, count and way of recording."""
    if isinstance(hows, str):
        hows = [hows] * len(groups)
    d_obs = ks_to_unit_exponential(pooled_rescaled(groups))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(g)) for g in groups]
    sizes = [len(g) for g in groups]
    hits = 0
    for _ in range(n_boot):
        sim = [record(rng.exponential(m, k), how) for m, k, how in zip(means, sizes, hows)]
        hits += ks_to_unit_exponential(pooled_rescaled(sim)) >= d_obs
    return d_obs, (hits + 1) / (n_boot + 1)


def load_first_exits(out_dir="results"):
    """({label: [sweeps]} for the Arrhenius runs, {(N, lam, g): [sweeps]} for T22)."""
    arr = defaultdict(list)
    for r in csv.DictReader(open(Path(out_dir) / "cqg_tube_arrhenius_lam125.csv", newline="")):
        if r["left"] == "True":
            arr["arrhenius N=%s g=%s" % (r["N"], r["g"])].append(float(r["wait"]))
    t22 = defaultdict(list)
    for f in sorted(glob.glob(str(Path(out_dir) / "t22_exits_n*_lam*.csv"))):
        for r in csv.DictReader(open(f, newline="")):
            t22[(int(r["N"]), float(r["lam"]), float(r["g"]))].append(float(r["first_exit_sweeps"]))
    return arr, t22


def stuck(cell):
    """The paper's definition (Sec. V): more than half the decays still at least 75 % torus at sweep 200.
    T7's files predate the f_200 column and are all at lambda = 1.25, where the torus is stuck."""
    f200 = [float(r["f_200"]) for r in cell if r.get("f_200", "") not in ("", None)]
    return (not f200) or sum(1 for x in f200 if x < 0.25) > len(f200) / 2


def load_decays(out_dir="results"):
    """{(set, N, lam): rows} for every stuck cell of T7 (sets t7, t7b) and T8 (set t8)."""
    cells = defaultdict(list)
    for pattern in ("t7_lam125_n*.csv", "t7b_lam125_n*.csv", "t8_lam*.csv"):
        for f in sorted(glob.glob(str(Path(out_dir) / pattern))):
            tag = Path(f).stem.split("_")[0]
            for r in csv.DictReader(open(f, newline="")):
                cells[(tag, int(r["N"]), float(r["lam"]))].append(r)
    return {k: v for k, v in sorted(cells.items()) if stuck(v)}


def beyond_watch(rows, clean_only=False):
    """W - 205 for every recorded W > 205; with clean_only, only tori that had not begun at sweep 200."""
    out = []
    for r in rows:
        if r["waiting"] in ("", None) or float(r["waiting"]) <= WATCH:
            continue
        if clean_only and float(r["f_200"]) != 0.0:
            continue
        out.append(float(r["waiting"]) - WATCH)
    return out


def rotation_test(rows, n_perm=N_PERM, seed=0):
    """(observed mean distance, null mean, two-sided p, count) for the single-leftover replicas given.
    The null rotates each leftover through every column of its torus, the start held fixed."""
    rng = np.random.default_rng(seed)
    obs, null = [], []
    for r in rows:
        d, half = a11.distance_of(r)
        if not np.isfinite(d):
            continue
        lx = int(round(2 * half))
        m = a11.circular_mean([int(c) for c in str(r["ring_cols"]).split()], lx)
        start = float(r["start"])
        dk = []
        for k in range(lx):
            x = abs((m + k) % lx - start) % lx
            dk.append(min(x, lx - x))
        obs.append(d)
        null.append(dk)
    null = np.array(null)
    pick = rng.integers(0, null.shape[1], (len(null), n_perm))
    draws = null[np.arange(len(null))[:, None], pick].mean(axis=0)
    expect = float(null.mean())
    o = float(np.mean(obs))
    p = float(np.mean(np.abs(draws - expect) >= abs(o - expect) - 1e-12))
    return o, expect, p, len(obs)


def survival_sets(out_dir="results"):
    """The two pooled, rescaled samples of the paper's survival panel: every first exit (Arrhenius and
    T22), and T8's waits beyond the watch for tori that had not begun at sweep 200."""
    arr, t22 = load_first_exits(out_dir)
    decays = load_decays(out_dir)
    first = pooled_rescaled(list(arr.values()) + list(t22.values()))
    clean = [beyond_watch(v, True) for k, v in decays.items() if k[0] == "t8"]
    return first, pooled_rescaled([c for c in clean if len(c) >= 5])


def main():
    print("Post-hoc checks for paper 1 (NOT pre-registered; ASSUMPTIONS O42).\n")
    print("1. Waiting times against an exponential, pooled after rescaling by each condition's mean.")
    arr, t22 = load_first_exits()
    groups = list(arr.values()) + list(t22.values())
    hows = ["sweep"] * len(arr) + ["exact"] * len(t22)
    d, p = bootstrap_p(groups, hows)
    per = [bootstrap_p([g], [h], n_boot=2000, seed=i)[1] for i, (g, h) in enumerate(zip(groups, hows))]
    print("  (i) first exits: %d conditions, %d times, D = %.3f, bootstrap p = %.2f; p < 0.05 in %d of %d conditions"
          % (len(groups), sum(map(len, groups)), d, p, sum(q < 0.05 for q in per), len(per)))
    x = np.concatenate([np.array(v) / tau_eq2(lam, g) for (n, lam, g), v in t22.items()])
    ks = stats.kstest(x, "expon")
    print("  (i) T22 against Eq. (2)'s mean, nothing estimated: %d times, D = %.3f, p = %.2f" % (len(x), ks.statistic, ks.pvalue))
    for (n, lam, g), v in sorted(t22.items()):
        k = stats.kstest(np.array(v) / tau_eq2(lam, g), "expon")
        print("        N = %d, lambda = %.2f: mean %.0f against %.0f, p = %.2f" % (n, lam, np.mean(v), tau_eq2(lam, g), k.pvalue))

    decays = load_decays()
    for name, keys, clean in (("(ii) T7 and T8, every wait beyond 205", list(decays), False),
                              ("(ii) T7 alone", [k for k in decays if k[0] != "t8"], False),
                              ("(ii) T8, tori not begun at sweep 200", [k for k in decays if k[0] == "t8"], True)):
        g = [beyond_watch(decays[k], clean) for k in keys]
        g = [x for x in g if len(x) >= 5]
        d, p = bootstrap_p(g, "block")
        print("  %-40s %2d conditions, %3d times: D = %.3f, p = %.2f" % (name, len(g), sum(map(len, g)), d, p))
    t7 = [k for k in decays if k[0] != "t8"]
    scan = []
    for origin in (WATCH, WATCH + 50, WATCH + 100, WATCH + 200):
        g = [[float(r["waiting"]) - origin for r in decays[k] if r["waiting"] and float(r["waiting"]) > origin] for k in t7]
        scan.append("from sweep %d: %d times, D = %.3f, p = %.3f" % ((origin,) + (sum(map(len, g)),) + bootstrap_p(g, "block")))
    print("      T7 with the clock started later (clock-shift sensitivity; detector selection remains):\n        " + "\n        ".join(scan))
    late = sum(1 for k, v in decays.items() if k[0] == "t8" for r in v
               if r["waiting"] and float(r["waiting"]) > WATCH and float(r["f_200"]) > 0)
    at205 = sum(1 for v in decays.values() for r in v if r["waiting"] and float(r["waiting"]) <= WATCH)
    print("      left out: %d waits recorded at 205 (the torus went during the watch); in T8, %d waits beyond 205"
          " from tori already begun at sweep 200" % (at205, late))
    x = pooled_rescaled([x for x in (beyond_watch(v) for v in decays.values()) if len(x) >= 5])
    print("      where every wait beyond 205 departs: share below 0.05 of the mean %.3f (exponential %.3f)"
          % (np.mean(x < 0.05), 1 - math.exp(-0.05)))

    waits = [v for (n, lam, g), v in t22.items() if lam == 1.05]
    waits += [[float(r["waiting"]) for r in v if r["waiting"]] for k, v in decays.items() if k[0] == "t8" and k[2] == 1.05]
    far = sum(1 for w in waits for x in w if x > FAR)
    expect = sum(len(w) * math.exp(-FAR / np.mean(w)) for w in waits)
    print("  (iii) lambda = 1.05: %d of %d waits beyond %d sweeps (%s); an exponential with each cell's mean gives"
          " %.3f, and 3 or more with probability %.1e"
          % (far, sum(map(len, waits)), FAR, sorted(round(x) for w in waits for x in w if x > FAR),
             expect, 1 - stats.poisson.cdf(far - 1, expect)))

    print("\n2. Relic position against where the change started (T11, single-leftover replicas).")
    rows = [r for r in a11.load() if int(r["rings"]) == 1]
    for n in sorted({int(r["N"]) for r in rows}):
        o, e, p, k = rotation_test([r for r in rows if int(r["N"]) == n])
        print("  N = %3d: %2d replicas, mean distance %.2f columns, rotation null %.2f, two-sided p = %.2f"
              % (n, k, o, e, p))


if __name__ == "__main__":
    main()
