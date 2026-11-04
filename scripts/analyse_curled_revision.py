"""Read-only, retrospective checks for the 9 October 2026 paper revision.

Ours, unverified by an independent physicist. No simulation or results/ writes.
Run from the repository root; --neutral exhausts neutral classes at stated sizes.
The default prints replica-level uncertainty, denominators and discrete-bath estimates.
"""
import argparse
import csv
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def binomial_interval(k, n):
    """Two-sided 95% Clopper--Pearson interval, replicas as trials."""
    return (0.0 if k == 0 else beta.ppf(.025, k, n-k+1),
            1.0 if k == n else beta.ppf(.975, k+1, n-k))


def bootstrap_interval(values, seed=20261009):
    """Percentile interval for a mean; resample entire replicas, not vertices."""
    x = np.asarray(values, float)
    if not len(x):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def neutral_closure(n):
    """Exhaustive BFS of the reachable neutral quotient at lambda=5/4.

    Side-preserving isomorphisms map switches bijectively to switches. Enumerating
    every switch from one representative of every discovered class therefore
    certifies closure when the queue empties. No sampled walk or class limit.
    census enumerates all valid switches; we assert that ALL energy-neutral
    switches preserve (S,X), so its neutral list is complete at this lambda.
    """
    from exact_torus_level import A, B, NEUTRAL, census, rate_per_sweep
    from graphity.cqg import NO_CAP, torus
    from graphity.symmetry import canonical_key, count

    adj, part = torus(4, n//4, NO_CAP)
    side = np.flatnonzero(part == 0)
    pending = deque([adj])
    seen = {canonical_key(adj, part)}
    records = []
    while pending:
        current = pending.popleft()
        counts, neutral = census(current, side)
        zero = {key for key in counts if -16*key[0]+5*key[1] == 0}
        assert zero == {NEUTRAL}, zero
        costs = [-16*s+5*x for s, x in counts if (s, x) != NEUTRAL]
        record = dict(symmetries=count(current, part), minimum=min(costs),
                      A=counts[A], B=counts[B], neutral=counts[NEUTRAL],
                      rate=rate_per_sweep(counts, n, 1.25, 1.5),
                      census=tuple(sorted(counts.items())))
        records.append(record)
        for trial, _ in neutral:
            key = canonical_key(trial, part)
            if key not in seen:
                seen.add(key)
                pending.append(trial)
    assert len(records) == len(seen)
    print("NEUTRAL CLOSURE", n, "classes", len(records), "symmetries",
          sorted(r["symmetries"] for r in records), "minimum", min(r["minimum"] for r in records),
          "distinct full censuses", len({r["census"] for r in records}),
          "full exit rate range", (min(r["rate"] for r in records),
                                   max(r["rate"] for r in records)), flush=True)
    return records


def summaries():
    from analyse_t9 import classify

    print("RETROSPECTIVE; intervals 95%; no original verdict rescored.")
    print("T7b: snapshot denominators, mean with replica bootstrap interval, distribution")
    for path in sorted((ROOT / "results").glob("t7b_lam125_n*.csv")):
        rs = rows(path)
        n = int(rs[0]["N"])
        selected = [r for r in rs if r["d2_50"]]
        two = [(int(r["d1_50"])+int(r["d2_50"]))/n for r in selected]
        largest = [float(r["largest_50"]) for r in selected]
        print(path.stem, "total/half/75/detected", len(rs), len(selected),
              sum(float(r["reached"] or 0) >= .75 for r in rs),
              sum(bool(r["waiting"]) for r in rs),
              "signature mean/CI", np.mean(two), bootstrap_interval(two),
              "largest mean/CI", np.mean(largest), bootstrap_interval(largest),
              "largest min/median/max", np.quantile(largest, [0, .5, 1]).tolist(),
              "individual largest<.7", sum(x < .7 for x in largest))

    print("T22: complete-event exponential mean intervals (model conditional); reciprocal-exits bootstrap")
    for path in sorted((ROOT / "results").glob("t22_exits_*.csv")):
        rs = rows(path)
        selected = [r for r in rs if r["went_through"] == "True"]
        waits = np.array([float(r["first_exit_sweeps"]) for r in rs
                          if float(r["first_exit_sweeps"]) > 0])
        exits = np.array([int(r["exits"]) for r in selected])
        count = len(waits)
        interval = 2*waits.sum()/chi2.ppf([.975, .025], 2*count)
        low, high = bootstrap_interval(exits)
        print(path.stem, "total/first/quarter", len(rs),
              len(waits), len(selected),
              "first mean/CI", waits.mean(), interval.tolist(),
              "reciprocal-exits/CI", 1/exits.mean(), (1/high, 1/low))

    print("T8 and T7b: full-window-release indicator, NOT certified topology")
    cells = defaultdict(list)
    paths = list((ROOT / "results").glob("t8_lam*.csv"))
    paths += list((ROOT / "results").glob("t7b_lam125_n*.csv"))
    for path in paths:
        for r in rows(path):
            cells[float(r["lam"])].append(r)
    for lam, rs in sorted(cells.items()):
        expected = 4*(lam-1)
        k = sum(abs(float(r["released"])-expected) <= .01*expected for r in rs)
        print(lam, "total/detected/half/75", len(rs), sum(bool(r["waiting"]) for r in rs),
              sum(bool(r["d2_50"]) for r in rs), sum(float(r["reached"] or 0) >= .75 for r in rs),
              "full-release count/nominal pooled CI", k, binomial_interval(k, len(rs)))
        for n in sorted({int(r["N"]) for r in rs}):
            cell = [r for r in rs if int(r["N"]) == n]
            hits = sum(abs(float(r["released"])-expected) <= .01*expected for r in cell)
            print("  size", n, "full release", hits, len(cell), binomial_interval(hits, len(cell)))

    print("T9: final mean-store energy and hypothetical canonical g(delta=1), NOT calibrated temperatures")
    for path in sorted((ROOT / "results").glob("t9_n*_C*.csv")):
        rs = [r for r in rows(path) if r["final"] == "1"]
        n, c = int(rs[0]["N"]), int(rs[0]["C"])
        u = np.array([float(r["bath_T"]) for r in rs])
        proxy = np.zeros_like(u)
        positive = u > 0
        proxy[positive] = 1/np.log1p(1/u[positive])
        k = sum(classify(r) == "sheet" for r in rs)
        print(n, c, "sheet signature", k, len(rs), binomial_interval(k, len(rs)),
              "mean store", u.mean(), "g proxy range", (proxy.min(), proxy.max()),
              "max drift", max(float(r["drift"]) for r in rs))
    print("T10: cold-box proxy range; no equilibration inferred")
    for path in sorted((ROOT / "results").glob("t10_n*.csv")):
        rs = [r for r in rows(path) if r["final"] == "1"]
        u = np.array([float(r["bath_T"]) for r in rs])
        g = 1/np.log1p(1/u)
        print(path.stem, "mean store range", (u.min(), u.max()), "g proxy range", (g.min(), g.max()))

    print("T38: row completeness and missing detections (excluded by historical analyzer)")
    cells = defaultdict(list)
    for path in sorted((ROOT / "results").glob("t38_*.csv")):
        for r in rows(path):
            cells[(float(r["lam"]), int(r["N"]))].append(r)
    for key, rs in sorted(cells.items()):
        missing = [r for r in rs if not r["waiting"]]
        print(key, "total/detected/missing", len(rs), len(rs)-len(missing), len(missing),
              "missing reaching 75%", sum(float(r["reached"] or 0) >= .75 for r in missing))
    snapshots = list((ROOT / "results").glob("t38_*_waiting/*.npz"))
    print("T38 waiting snapshots available in frozen results/:", len(snapshots))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral", action="store_true")
    args = parser.parse_args()
    if args.neutral:
        for n in (48, 64, 96, 144, 192, 288):
            records = neutral_closure(n)
            # These are the finite-size claims of this paper, not assumptions of BFS.
            assert len(records) == 3
            assert len({r["census"] for r in records}) == 1
            assert all(r["minimum"] == 12 and r["A"] == 3*n and r["B"] == 2*n
                       and r["neutral"] == n//2 for r in records)
    else:
        summaries()
