"""T74: read the runs of `scripts/run_layer_outside.py` by the rules of PREREGISTRATION.md section T74 as amended
(Amendments 2 and 3). Usage:

    python scripts/analyse_t74.py results/t74_<cell>.csv [...]

A named set of files is read (no glob over results/). Nothing here scores the held-out size; that is done against the
dated amendment once it exists.

Per replica: the daughters and the energy of each (the rise in daughter_energy at each logged birth; every birth sweep is
logged, and with one layer proposal a sweep at most one daughter is born per sweep, so each rise is one daughter; a rise
of more than one at once is flagged and its energy split evenly). A daughter born up to the end of digestion
(sweep <= t_unfed) is counted as fed (Amendment 3, item 1); one born after it as unfed. The periods between births are
taken over the fed phase proper (sweep <= n_fed, while the meals arrive), since the timing words compare feeds (ours:
the amendment counts digestion's daughters as fed but does not say they enter the timing). Also read: the share of the
energy fed that has crossed the curl edge into the slab by the end of the fed phase (edge_in / fed at sweep n_fed) and
by the end of digestion; the slab's stored energy (the sum of its stores, ASSUMPTIONS Q24) at the start of the unfed
phase (sweep t_unfed, Amendment 3 item 2) and at the end, its loss, and the mean rate of loss over the unfed phase
against c; whether digestion ended by balance or by its limit; and the common mean of reservoir and slab at the unfed
start against 2a.

Words, as registered:
* fed phase (with digestion), per cell: BURPS if every replica makes at least one fed daughter and every fed daughter
  carries a N_slab within 1e-9 (absolute); NEVER BURPS if no replica makes one; MIXED otherwise;
* unfed phase, per cell (Amendment 2 item 3 as amended by Amendment 3 items 2 and 3), each against the stored energy at
  the unfed start: WAITS if it ends within 5 % of it and no daughter is born in the unfed phase, in at least 15 of 16
  replicas; DISSOLVES if it ends at most half of it, in at least 15 of 16; FED BY ITS SURROUNDINGS if a daughter is born
  in the unfed phase or it ends more than 5 % above it, in at least 15 of 16; if more than one bar is met, MIXED, with
  the bars met noted (Amendment 3, item 3); MIXED otherwise. With other than 16 replicas the bar is ceil(15 n / 16);
* meals against steady, per (lambda, L, c), as registered: the reservoir is isolated while fed, so the background does
  not touch the fed phase and the cells of the three backgrounds are independent replicas of the same fed phase; they
  are pooled (48 replicas a feed). The mean period is pooled over every fed-phase interval of every replica, and its
  coefficient of variation (population standard deviation over the mean, pooled the same way). SAME TIMING if the
  meals' mean is within 10 % of steady's; SPREAD ONLY if in addition the meals' coefficient of variation is at least
  twice steady's; DIFFERENT TIMING if they differ by more than 10 %. No period gives NO PERIOD.

Our prediction for the unfed words (Amendment 3, item 4), scored per cell: cold DISSOLVES; warm FED BY ITS
SURROUNDINGS; equal WAITS where the common mean at the unfed start is below 2a and FED BY ITS SURROUNDINGS where it is
above, read from each replica; a cell whose replicas lie on both sides is reported SPLIT and not scored.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np

EXACT = 1e-9
WAIT_BAND = 0.05
DISSOLVE_SHARE = 0.5
TIMING_TOL = 0.10
SPREAD_FACTOR = 2.0
WAITS, DISSOLVES, FED_BY = "WAITS", "DISSOLVES", "FED BY ITS SURROUNDINGS"


def read_replica(rows):
    rows = sorted(rows, key=lambda r: int(r["sweep"]))
    r0 = rows[0]
    lam, n_slab = float(r0["lam"]), int(r0["N_slab"])
    a = 4.0 * (lam - 1.0)
    a_n = a * n_slab
    n_fed, n_unfed, c, t_unfed = int(r0["n_fed"]), int(r0["n_unfed"]), float(r0["c"]), int(r0["t_unfed"])
    births, flagged = [], 0
    prev_d, prev_e = 0, 0.0
    for r in rows:
        d, e = int(r["daughters"]), float(r["daughter_energy"])
        k = d - prev_d
        if k > 0:
            if k > 1:
                flagged += 1
            births.extend([(int(r["sweep"]), (e - prev_e) / k)] * k)
        prev_d, prev_e = d, e
    fed_b = [b for b in births if b[0] <= t_unfed]
    unfed_b = [b for b in births if b[0] > t_unfed]
    by_sweep = {int(r["sweep"]): r for r in rows}
    for t in (n_fed, t_unfed):
        if t not in by_sweep:
            raise ValueError("the row at sweep %d is missing" % t)
    stop, start = by_sweep[n_fed], by_sweep[t_unfed]
    s0, s_end = float(start["stores"]), float(rows[-1]["stores"])
    lost = s0 - s_end
    fed_total = float(stop["fed"])
    g_common = float(r0["g_common"])
    sweeps = [s for s, _ in births if s <= n_fed]
    waits = abs(s_end - s0) <= WAIT_BAND * s0 and not unfed_b
    dissolves = s_end <= DISSOLVE_SHARE * s0
    fed_by = bool(unfed_b) or s_end > (1.0 + WAIT_BAND) * s0
    return dict(
        a=a, a_n=a_n, c=c, n_fed=n_fed, n_unfed=n_unfed, t_unfed=t_unfed,
        digestion=t_unfed - n_fed, digest_balanced=bool(int(r0["digest_balanced"])),
        g_common=g_common, common_above_2a=g_common > 2.0 * a,
        daughters=len(births), fed_daughters=len(fed_b), digestion_daughters=sum(1 for s, _ in fed_b if s > n_fed),
        unfed_daughters=len(unfed_b), flagged=flagged,
        exact=all(abs(e - a_n) <= EXACT for _, e in fed_b), unfed_exact=all(abs(e - a_n) <= EXACT for _, e in unfed_b),
        energies=[e for _, e in births], periods=list(np.diff(sweeps)) if len(sweeps) >= 2 else [],
        share_entered=(float(stop["edge_in"]) / fed_total) if fed_total > 0 else None,
        share_entered_digested=(float(start["edge_in"]) / fed_total) if fed_total > 0 else None,
        s_unfed_start=s0, s_end=s_end, lost=lost,
        loss_rate_over_c=(lost / n_unfed / c) if (n_unfed and c) else None,
        waits=waits, dissolves=dissolves, fed_by_surroundings=fed_by,
        max_residual=max(abs(float(r["residual"])) for r in rows))


def bar(n):
    return 15 if n == 16 else math.ceil(15 * n / 16)


def unfed_word(n_wait, n_diss, n_fedby, n):
    met = [w for w, k in ((WAITS, n_wait), (DISSOLVES, n_diss), (FED_BY, n_fedby)) if k >= bar(n)]
    if len(met) == 1:
        return met[0], met
    if len(met) > 1:
        return "MIXED (bars met: %s)" % " and ".join(met), met
    return "MIXED", met


def predicted(background, reads):
    """Amendment 3, item 4 (ours)."""
    if background == "cold":
        return DISSOLVES
    if background == "warm":
        return FED_BY
    if background == "equal":
        above = sum(x["common_above_2a"] for x in reads)
        if above == len(reads):
            return FED_BY
        if above == 0:
            return WAITS
        return "SPLIT (%d of %d above 2a)" % (above, len(reads))
    return None


def read_cells(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (float(r["lam"]), int(r["L"]), r["feed_label"], r["cap_label"], r["background"])
        groups[key][int(r["replica"])].append(r)
    cells = {}
    for key, reps in sorted(groups.items()):
        reads = [read_replica(rs) for _, rs in sorted(reps.items())]
        n = len(reads)
        if all(x["fed_daughters"] >= 1 and x["exact"] for x in reads):
            fed_word = "BURPS"
        elif all(x["fed_daughters"] == 0 for x in reads):
            fed_word = "NEVER BURPS"
        else:
            fed_word = "MIXED"
        n_wait = sum(x["waits"] for x in reads)
        n_diss = sum(x["dissolves"] for x in reads)
        n_fedby = sum(x["fed_by_surroundings"] for x in reads)
        word, met = unfed_word(n_wait, n_diss, n_fedby, n)
        pred = predicted(key[4], reads)
        periods = [p for x in reads for p in x["periods"]]
        mean_p, cv = pooled(periods)
        shares = [x["share_entered"] for x in reads if x["share_entered"] is not None]
        rates = [x["loss_rate_over_c"] for x in reads if x["loss_rate_over_c"] is not None]
        cells[key] = dict(
            n=n, fed_word=fed_word, unfed_word=word, bars_met=met, n_wait=n_wait, n_dissolve=n_diss,
            n_fed_by=n_fedby, predicted=pred,
            prediction_holds=(None if pred is None or pred.startswith("SPLIT") else pred == word),
            n_common_above_2a=sum(x["common_above_2a"] for x in reads),
            median_g_common=float(np.median([x["g_common"] for x in reads])),
            n_digest_balanced=sum(x["digest_balanced"] for x in reads),
            median_digestion=float(np.median([x["digestion"] for x in reads])),
            a_n=reads[0]["a_n"], c=reads[0]["c"], two_a=2.0 * reads[0]["a"],
            mean_period=mean_p, cv_period=cv, n_periods=len(periods), periods=periods,
            median_daughters=float(np.median([x["daughters"] for x in reads])),
            median_digestion_daughters=float(np.median([x["digestion_daughters"] for x in reads])),
            median_unfed_daughters=float(np.median([x["unfed_daughters"] for x in reads])),
            unfed_inexact=sum(not x["unfed_exact"] for x in reads), flagged=sum(x["flagged"] for x in reads),
            median_share_entered=float(np.median(shares)) if shares else None,
            median_loss_rate_over_c=float(np.median(rates)) if rates else None,
            max_residual=max(x["max_residual"] for x in reads), replicas=reads)
    return cells


def pooled(periods):
    if not periods or np.mean(periods) <= 0:
        return None, None
    return float(np.mean(periods)), float(np.std(periods) / np.mean(periods))


def timing(cells):
    """{(lam, L, cap, meals label): (word, mean ratio, cv ratio)} for every meals feed with a steady twin, the fed-phase
    intervals pooled over the backgrounds."""
    groups = defaultdict(list)
    for (lam, L, feed, cap, bg), c in cells.items():
        groups[(lam, L, feed, cap)].extend(c["periods"])
    out = {}
    for (lam, L, feed, cap), periods in sorted(groups.items()):
        if feed == "steady" or (lam, L, "steady", cap) not in groups:
            continue
        m, cv = pooled(periods)
        ms, cvs = pooled(groups[(lam, L, "steady", cap)])
        if not m or not ms:
            out[(lam, L, cap, feed)] = ("NO PERIOD", None, None)
            continue
        ratio = m / ms
        cv_ratio = (cv / cvs) if cvs else (math.inf if cv else None)
        if abs(ratio - 1.0) > TIMING_TOL:
            word = "DIFFERENT TIMING"
        elif cv_ratio is not None and cv_ratio >= SPREAD_FACTOR:
            word = "SPREAD ONLY"
        else:
            word = "SAME TIMING"
        out[(lam, L, cap, feed)] = (word, ratio, cv_ratio)
    return out


def main(paths):
    rows = []
    for p in paths:
        rows.extend(csv.DictReader(open(p, newline="")))
    cells = read_cells(rows)
    for (lam, L, feed, cap, bg), c in cells.items():
        print("lam=%.2f L=%2d %-8s %-6s %-5s | fed %-11s unfed %-23s (waits %d, dissolves %d, fed by surroundings %d, "
              "of %d) | predicted %s, holds %s | common mean median %.3f vs 2a %.3f (%d above) | digestion median %s "
              "(%d balanced) | daughters median %s (digestion %s, unfed %s, unfed inexact %d) vs a N_slab %.1f | "
              "period mean %s cv %s (%d) | share entered %s | loss rate / c %s | max residual %.1e%s"
              % (lam, L, feed, cap, bg, c["fed_word"], c["unfed_word"], c["n_wait"], c["n_dissolve"], c["n_fed_by"],
                 c["n"], c["predicted"], c["prediction_holds"], c["median_g_common"], c["two_a"],
                 c["n_common_above_2a"], c["median_digestion"], c["n_digest_balanced"], c["median_daughters"],
                 c["median_digestion_daughters"], c["median_unfed_daughters"], c["unfed_inexact"], c["a_n"],
                 c["mean_period"], c["cv_period"], c["n_periods"], c["median_share_entered"],
                 c["median_loss_rate_over_c"], c["max_residual"],
                 " | %d multi-birth rows flagged" % c["flagged"] if c["flagged"] else ""))
    for (lam, L, cap, feed), (word, ratio, cv_ratio) in sorted(timing(cells).items()):
        print("lam=%.2f L=%2d %-6s %-8s vs steady (backgrounds pooled): %-16s mean ratio %s cv ratio %s"
              % (lam, L, cap, feed, word, ratio, cv_ratio))
    return cells


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("name the result files to read (no glob): python scripts/analyse_t74.py results/t74_<cell>.csv ...")
    main(sys.argv[1:])
