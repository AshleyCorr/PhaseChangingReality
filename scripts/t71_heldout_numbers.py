"""The numbers for T71's held-out size L = 16 (PREREGISTRATION.md T71, Amendment 4), written from L = 8 and 12 alone
(CLAUDE.md rule 14). Reads results/t71_{model,layer,layerref}_* only (already read and scored, O144). Prints the table
the amendment quotes and writes one row per predicted number to configs/t71_heldout_prediction.csv (or the path given):

    PYTHONPATH="src;scripts" python3 scripts/t71_heldout_numbers.py [out.csv]

Columns of the csv: cell (the L = 16 config and the rate), statistic, rule (A, B, C, D, word, or reported), predicted,
low, high, scored (yes/no), units, note. Quantities in "a N_slab" are fractions of the release a N_slab, which at
L = 16 is 409.6, 1,024 and 1,638.4 at lambda = 1.10, 1.25 and 1.40. The arm C control is not in this file: it is left
out of the held-out stage (Amendment 4, section 7, the owner's choice (c)).

Moved here on 2026-10-25 from sandbox/heldout_2026-10-23/t71_heldout_numbers_v2.py (written 2026-10-23); the logic is
unchanged, the csv is added. The scratch first version, t71_heldout_numbers.py in the same folder, is superseded.
What changed from that first version:

  1. The bands are predictive and family-wise. The first version's band (the spread of a 16-median drawn from the
     pooled replicas) left out the spread of the pooled median itself, and its cross-check between L = 8 and 12 put
     6 of 136 statistics outside a nominal 99 % band. Here the band for a new 16-median is the pooled median plus the
     0.5 alpha and 1 - 0.5 alpha quantiles of (median of 16 drawn) - (median of 32 drawn), both from the pooled
     replicas, so it carries both spreads; alpha per number is 0.10 divided by the number of sampled, scored numbers
     that can fail, so that the chance of any chance miss among them, if nothing depends on L, is at most 0.10
     (Bonferroni). The cross-check is rerun with the same construction (16 against 16).
  2. Scored set: the registered per-cell readings only (T71 Definitions). Clean-move totals, replicas with a clean
     move, the f1 open share and refolds, and the control's stop and fed are reported, not scored. A range that spans
     every value the quantity can take is reported as "cannot fail" and not counted.
  3. Rule D (the model arm at lambda = 1.10, r = 1/16): the range also holds a power law in N_slab for the shortfall from
     the value with no clean move (1 for the energy per daughter; arm B's pooled median for the daughters and the slab
     share), and is widened by the family-wise predictive half-widths from the L = 12 replicas alone. The end energy
     there is reported by class in absolute units, not scored.
  4. Rule B at the same family-wise level: the median's band from the exact law, the latest replica's bound set so its
     miss probability is below alpha.
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "t71_heldout_prediction.csv"
COLUMNS = ["cell", "statistic", "rule", "predicted", "low", "high", "scored", "units", "note"]
ARM = {("model", "daughter"): "model", ("layer", "daughter"): "layer", ("layer", "stores"): "layerref"}


def cell_name(c):
    """The L = 16 config of a cell (move_set, release_to, lam, rate), and its rate."""
    ms, rel, lam, rate = c
    return "t71_%s_lam%d_l16 r=1/%d" % (ARM[(ms, rel)], round(lam * 100), round(1 / rate))


STAT_NAME = dict(end_F="end energy above flat", daughters="daughters", per_daughter="energy per daughter",
                 period="period", slab_share="slab share", open_share="open share", refolds="refolds",
                 clean_total="clean moves in all")
STAT_UNITS = dict(daughters="daughters", period="sweeps", slab_share="share of logged sweeps",
                  open_share="share of logged sweeps", refolds="refolds", clean_total="moves in 16 replicas")


def write_prediction(path, rows):
    """Write the predicted numbers, one dict per row with the keys of COLUMNS; numbers rounded to 6 places."""
    def fmt(v):
        if isinstance(v, float):
            return ("%.6f" % v).rstrip("0").rstrip(".")
        return "" if v is None else str(v)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in COLUMNS})
sys.path.insert(0, str(ROOT / "scripts"))
import analyse_t71 as A  # noqa: E402

FILES = ["t71_model_lam110", "t71_model_lam125", "t71_model_lam140",
         "t71_layer_lam110", "t71_layer_lam125", "t71_layer_lam140",
         "t71_layerref_lam110", "t71_layerref_lam125", "t71_layerref_lam140"]
NBOOT, SEED = 200_000, 20261023
FAMILY = 0.10
BOUNDS = dict(slab_share=(0, 1), open_share=(0, 1), per_daughter=(0, 1), end_F=(0, None), daughters=(0, None),
              period=(0, None), refolds=(0, None), clean_total=(0, None))


def t_gate(L, r, gate):
    t = math.floor(gate / (2 * L * L * r))
    while 2 * L * L * r * t + 1e-12 < gate:
        t += 1
    return t


def stat_of(q):
    return np.sum if q == "clean_total" else np.median


def diff_draws(vals, n_new, n_ref, rng, stat):
    """(stat of n_new drawn) - (stat of n_ref drawn), both with replacement from vals; NBOOT draws."""
    v = np.asarray(vals, dtype=float)
    a = stat(rng.choice(v, size=(NBOOT, n_new), replace=True), axis=1)
    b = stat(rng.choice(v, size=(NBOOT, n_ref), replace=True), axis=1)
    if stat is np.sum:          # a total over 16 replicas, predicted from n_ref replicas: scale the reference to 16
        b = b * n_new / n_ref
    return a - b


def band(center, d, alpha):
    return center + float(np.quantile(d, alpha / 2)), center + float(np.quantile(d, 1 - alpha / 2))


def one_sample_band(vals, rng, stat, lo=0.5, hi=99.5, nboot=20000):
    """The first version's band: percentiles of the 16-median drawn from vals alone."""
    v = np.asarray(vals, dtype=float)
    s = stat(rng.choice(v, size=(nboot, 16), replace=True), axis=1)
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def clip(x, q):
    lo, hi = BOUNDS.get(q, (None, None))
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def per_replica(rows):
    rows = sorted(rows, key=lambda x: int(x["sweep"]))
    x = A.read_replica(rows)
    a_n = x["a_n"]
    r9 = lambda v: None if v is None else round(v, 9)   # noqa: E731
    return dict(first_split=x["first_split"], release=r9(x["release"] / a_n if x["release"] is not None else None),
                end_F=r9(x["end_F"] / a_n), end_F_abs=round(x["end_F"], 3), refolds=x["refolds"],
                open_share=x["open_share"], slab_share=x["slab_share"], daughters=x["daughters"],
                per_daughter=r9(x["per_daughter"] / a_n if x["per_daughter"] is not None else None),
                period=x["period"], clean_total=int(rows[-1]["n_clean"]), spent_first=x["spent_first"], word=x["word"],
                end_width=A.width_of(rows[-1]["dims"]))


def load():
    groups = defaultdict(lambda: defaultdict(list))
    for name in FILES:
        for r in csv.DictReader(open(ROOT / "results" / (name + ".csv"), newline="")):
            key = (r["move_set"], r["release_to"], float(r["lam"]), float(r["rate"]), int(r["L"]))
            groups[key][int(r["replica"])].append(r)
    return {k: [per_replica(rs) for _, rs in sorted(v.items())] for k, v in groups.items()}


def quantities(ms, rel):
    qs = ["end_F"] + (["daughters", "per_daughter", "period", "slab_share"] if rel == "daughter"
                      else ["open_share", "refolds"])
    if ms == "model":
        qs.append("clean_total")
    return qs


def is_clean_cell(c):
    ms, rel, lam, rate = c
    return ms == "model" and abs(lam - 1.1) < 1e-9 and abs(rate - 0.0625) < 1e-12


def registered(rel, q):
    """True if q is a registered per-cell reading (T71 Definitions)."""
    if q in ("clean_total", "open_share", "refolds"):
        return False
    return True


def cross_check(reps, cells, alpha, rng, method):
    """Between L = 8 and 12 (the 17 cells with no clean move, and the period of the eighteenth): how often one size's statistic falls outside the band built from the other size alone.
    Returns (misses over every rule-A statistic, checks), (misses over the registered ones, checks)."""
    out_all = n_all = out_reg = n_reg = 0
    for c in cells:
        ms, rel = c[0], c[1]
        r8, r12 = reps[c + (8,)], reps[c + (12,)]
        for q in quantities(ms, rel):
            if q == "per_daughter":
                continue                       # exact (rule C), not a sampled number
            if is_clean_cell(c) and q != "period":
                continue                       # rule D: the size can matter there; only the period takes rule A
            stat = stat_of(q)
            v8 = [x[q] for x in r8 if x[q] is not None]
            v12 = [x[q] for x in r12 if x[q] is not None]
            m8, m12 = float(stat(v8)), float(stat(v12))
            if method == "first":
                b8, b12 = one_sample_band(v8, rng, stat), one_sample_band(v12, rng, stat)
            else:
                b8 = band(m8, diff_draws(v8, 16, 16, rng, stat), alpha)
                b12 = band(m12, diff_draws(v12, 16, 16, rng, stat), alpha)
            miss = int(not (b8[0] - 1e-12 <= m12 <= b8[1] + 1e-12)) + int(not (b12[0] - 1e-12 <= m8 <= b12[1] + 1e-12))
            out_all += miss
            n_all += 2
            if registered(rel, q) and rel == "daughter" or (registered(rel, q) and q == "end_F"):
                out_reg += miss
                n_reg += 2
    return (out_all, n_all), (out_reg, n_reg)


def main(out_path=OUT):
    reps = load()
    pred = []
    cells = sorted({k[:4] for k in reps})

    # the number of sampled, scored numbers: rule A (registered, not exact) + rule D (3) + rule B (median and latest
    # replica, 18 cells each); the "cannot fail" ones are found below and the count is redone until it is stable
    def rule_a_items():
        items = []
        for c in cells:
            ms, rel = c[0], c[1]
            for q in quantities(ms, rel):
                if q == "per_daughter" and not is_clean_cell(c):
                    continue
                if is_clean_cell(c) and q in ("daughters", "per_daughter", "slab_share", "end_F", "clean_total"):
                    continue
                items.append((c, q))
        return items

    items = rule_a_items()
    n_cells = len(cells)
    cannot_fail = set()
    for _ in range(5):
        n_a = sum(1 for (c, q) in items if registered(c[1], q) and (c, q) not in cannot_fail)
        n_tests = n_a + 3 + 2 * n_cells
        alpha = FAMILY / n_tests
        rng = np.random.default_rng(SEED)
        res = {}
        new_cf = set()
        for (c, q) in items:
            ms, rel, lam, rate = c
            stat = stat_of(q)
            v8 = [x[q] for x in reps[c + (8,)] if x[q] is not None]
            v12 = [x[q] for x in reps[c + (12,)] if x[q] is not None]
            pooled = v8 + v12
            m8, m12 = float(stat(v8)), float(stat(v12))
            if q == "clean_total":
                num = 0.0 if sum(pooled) == 0 else float(np.sum(pooled)) / 2
            else:
                num = float(stat(np.asarray(pooled)))
            d = diff_draws(pooled, 16, 32, rng, stat)
            lo, hi = band(num, d, alpha)
            if q != "clean_total":
                lo, hi = min(lo, m8, m12), max(hi, m8, m12)
            lo, hi = clip(lo, q), clip(hi, q)
            if q == "end_F" and set(round(v, 9) for v in pooled) <= {0.0, 1.0} and lo <= 0 and hi >= 1:
                new_cf.add((c, q))
            res[(c, q)] = (m8, m12, num, lo, hi)
        if new_cf == cannot_fail:
            break
        cannot_fail = new_cf

    # the law of the first split's wait at this alpha
    sim = np.random.default_rng(SEED + 1).geometric(0.5, size=(1_000_000, 16)) - 1
    med16 = np.median(sim, axis=1)
    g_num = float(np.median(med16))
    g_lo, g_hi = float(np.quantile(med16, alpha / 2)), float(np.quantile(med16, 1 - alpha / 2))
    p_med_out = float(np.mean((med16 < g_lo) | (med16 > g_hi)))
    k_late = 0
    while 1 - (1 - 2.0 ** -(k_late + 1)) ** 16 > alpha:
        k_late += 1
    p_late = 1 - (1 - 2.0 ** -(k_late + 1)) ** 16
    print("family-wise level %.2f over %d sampled, scored numbers that can fail (%d rule A, 3 rule D, %d rule B): alpha per number %.5f"
          % (FAMILY, n_tests, n_a, 2 * n_cells, alpha))
    print("first-split wait, median of 16: %.1f sweeps, band %.1f to %.1f (P outside, exact law: %.5f); latest replica at most t_gate + %d (P later %.5f); P(no replica at t_gate) = %.1e"
          % (g_num, g_lo, g_hi, p_med_out, k_late, p_late, 2.0 ** -16))

    for c in cells:
        ms, rel, lam, rate = c
        r8, r12 = reps[c + (8,)], reps[c + (12,)]
        tg = t_gate(8, rate, 4.0 * (lam - 1.0) * 256)
        assert tg == t_gate(12, rate, 4.0 * (lam - 1.0) * 576) == t_gate(16, rate, 4.0 * (lam - 1.0) * 1024)
        a16 = 4.0 * (lam - 1.0) * 1024
        words8, words12 = Counter(x["word"] for x in r8), Counter(x["word"] for x in r12)
        print("\n%s / %s, lambda = %.2f, r = 1/%d: words %s, %s; a N_slab at L = 16 = %.1f"
              % (ms, rel, lam, round(1 / rate), dict(words8), dict(words12), a16))
        fs8 = float(np.median([x["first_split"] for x in r8]))
        fs12 = float(np.median([x["first_split"] for x in r12]))
        print("    first split: t_gate %d; medians %.1f, %.1f -> %.1f, band %.1f to %.1f; replicas from %d to at most %d   [rule B]"
              % (tg, fs8, fs12, tg + g_num, tg + g_lo, tg + g_hi, tg, tg + k_late))
        cn = cell_name(c)
        unit_a = "a N_slab (%.1f at L = 16)" % a16
        assert len(words8) == len(words12) == 1 and set(words8) == set(words12)
        (word,) = words8
        pred.append(dict(cell=cn, statistic="majority word", rule="word", predicted=word, scored="yes", units="word",
                         note="16 of 16 expected, as at L = 8 and 12; scored on the majority word"))
        pred.append(dict(cell=cn, statistic="first split, median", rule="B", predicted=tg + g_num, low=tg + g_lo,
                         high=tg + g_hi, scored="yes", units="sweeps", note="t_gate %d plus a coin-flip wait" % tg))
        pred.append(dict(cell=cn, statistic="first split, earliest replica", rule="B", predicted=float(tg),
                         low=float(tg), high=float(tg), scored="yes", units="sweeps",
                         note="no split before t_gate is possible; a miss (no replica at t_gate) has P %.1e" % 2.0 ** -16))
        pred.append(dict(cell=cn, statistic="first split, latest replica", rule="B", predicted=None, low=float(tg),
                         high=float(tg + k_late), scored="yes", units="sweeps", note="an upper bound only"))
        pred.append(dict(cell=cn, statistic="release at the first split, median", rule="C", predicted=1.0, low=1.0,
                         high=1.0, scored="yes", units=unit_a, note="exact"))
        if ms == "model":
            pred.append(dict(cell=cn, statistic="SPENT FIRST count", rule="C", predicted=0.0, low=0.0, high=0.0,
                             scored="yes", units="replicas of 16", note="exact: no clean move can be paid before the first split"))
        for q in quantities(ms, rel):
            if (c, q) in res:
                m8, m12, num, lo, hi = res[(c, q)]
                tag = "rule A"
                if (c, q) in cannot_fail:
                    tag += ", reported, CANNOT FAIL"
                elif not registered(rel, q):
                    tag += ", reported, NOT SCORED"
                extra = ""
                if q == "end_F":
                    extra = "   in energy: %.1f (%.1f to %.1f)" % (num * a16, lo * a16, hi * a16)
                print("    %-12s: %.4f, %.4f -> %.4f (%.4f to %.4f)   [%s]%s" % (q, m8, m12, num, lo, hi, tag, extra))
                scored = "no" if ((c, q) in cannot_fail or not registered(rel, q)) else "yes"
                note = "stage 1: %.4f, %.4f" % (m8, m12)
                if (c, q) in cannot_fail:
                    note += "; reported, cannot fail"
                elif not registered(rel, q):
                    note += "; reported, not scored"
                pred.append(dict(cell=cn, statistic=STAT_NAME[q] + ("" if q == "clean_total" else ", median"),
                                 rule="A", predicted=num, low=lo, high=hi, scored=scored,
                                 units=unit_a if q == "end_F" else STAT_UNITS[q], note=note))
            elif q == "per_daughter" and not is_clean_cell(c):
                print("    per_daughter: 1 exactly (to 1e-9) wherever no clean move fires   [rule C]")
                pred.append(dict(cell=cn, statistic="energy per daughter, median", rule="C", predicted=1.0, low=1.0,
                                 high=1.0, scored="yes", units=unit_a, note="exact, to 1e-9, wherever no clean move fires"))
        if not is_clean_cell(c):
            continue
        # rule D
        base = {}
        for q in ("daughters", "slab_share"):
            vb = [x[q] for k in (8, 12) for x in reps[("layer", "daughter", lam, rate, k)]]
            base[q] = float(np.median(vb))
        base["per_daughter"] = 1.0
        for q in ("per_daughter", "daughters", "slab_share", "clean_total"):
            stat = stat_of(q)
            v8 = [x[q] for x in r8 if x[q] is not None]
            v12 = [x[q] for x in r12 if x[q] is not None]
            m8, m12 = float(stat(v8)), float(stat(v12))
            line_L = 2 * m12 - m8
            line_N = m12 + (m12 - m8) * 448 / 320
            ends = [m12, line_N]
            power = None
            if q in base:
                d8, d12 = base[q] - m8, base[q] - m12
                if d8 > 0 and d12 > 0:
                    p = math.log(d12 / d8) / math.log(576 / 256)
                    power = base[q] - d12 * (1024 / 576) ** p
                    ends.append(power)
            d = diff_draws(v12, 16, 16, rng, stat)
            q_lo, q_hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
            lo, hi = clip(min(ends) + q_lo, q), clip(max(ends) + q_hi, q)
            tag = "rule D" + ("" if registered(rel, q) else ", reported, NOT SCORED")
            print("    %-12s: %.4f, %.4f -> %.4f (%.4f to %.4f); line in N_slab %.4f; power law %s; L = 12 half-widths %.4f, +%.4f   [%s]"
                  % (q, m8, m12, clip(line_L, q), lo, hi, line_N, "%.4f" % power if power is not None else "-",
                     q_lo, q_hi, tag))
            pred.append(dict(cell=cn, statistic=STAT_NAME[q] + ("" if q == "clean_total" else ", median"),
                             rule="D" if registered(rel, q) else "reported", predicted=clip(line_L, q), low=lo, high=hi,
                             scored="yes" if registered(rel, q) else "no",
                             units=unit_a if q == "per_daughter" else STAT_UNITS[q],
                             note="stage 1: %.4f, %.4f; line in L; range from L = 12's value to the farthest of the line in N_slab and the power law, widened by L = 12's predictive half-widths" % (m8, m12)))
            if q == "per_daughter":
                print("                  in energy at L = 16: %.1f (%.1f to %.1f); stage 1: %.1f, %.1f"
                      % (clip(line_L, q) * a16, lo * a16, hi * a16, m8 * 4.0 * (lam - 1.0) * 256, m12 * 4.0 * (lam - 1.0) * 576))
        k8 = sum(1 for x in r8 if x["clean_total"]); k12 = sum(1 for x in r12 if x["clean_total"])
        print("    replicas with a clean move: %d, %d of 16   [reported, NOT SCORED]" % (k8, k12))
        pred.append(dict(cell=cn, statistic="replicas with a clean move", rule="reported", predicted="a majority",
                         scored="no", units="replicas of 16", note="stage 1: %d, %d of 16" % (k8, k12)))
        pred.append(dict(cell=cn, statistic="end energy above flat, median", rule="reported",
                         predicted="open (0, or patches of 128 or more), not the slab (409.6)", scored="no",
                         units="energy", note="reported by class, not scored: its classes change with L"))
        for L, rr in ((8, r8), (12, r12)):
            a_l = 4.0 * (lam - 1.0) * 4 * L * L
            def cls_of(x):
                if x["end_width"] == 4:
                    return "slab" if abs(x["end_F_abs"] - a_l) < 1e-6 else "slab, %g" % x["end_F_abs"]
                return "open, 0" if x["end_F_abs"] == 0 else "open, %g" % x["end_F_abs"]
            cls = Counter(cls_of(x) for x in rr)
            print("    end state at L = %d (absolute units; the slab is %.1f): %s; median %.1f   [reported by class, NOT SCORED]"
                  % (L, a_l, dict(sorted(cls.items())), float(np.median([x["end_F_abs"] for x in rr]))))

    # calibration
    rng_c = np.random.default_rng(SEED + 7)
    print("\ncross-check between L = 8 and 12 (every rule-A statistic, the clean cell's period included, each checked both ways;"
          " which ones miss: sandbox/heldout_2026-10-23/t71_heldout_crosscheck_misses.py):")
    for seed in (1, 2, 3, 20261023):
        (oa, na), (orr, nr) = cross_check(reps, cells, alpha, np.random.default_rng(seed), "first")
        print("  first version's band (one-sample, 99 %%), bootstrap seed %d: %d of %d outside (%d of %d registered)" % (seed, oa, na, orr, nr))
    for lvl in (0.01, alpha):
        (oa, na), (orr, nr) = cross_check(reps, cells, lvl, rng_c, "predictive")
        print("  predictive band, alpha per number %.5f: %d of %d outside (%d of %d registered)" % (lvl, oa, na, orr, nr))

    exp_b = n_cells * (p_med_out + p_late + 2.0 ** -16)
    print("\nexpected chance misses if nothing depends on L: rule A and D at most %.3f (%d numbers x alpha), rule B %.3f; in all at most %.3f; P(AS PREDICTED | nothing depends on L) at least %.3f"
          % ((n_a + 3) * alpha, n_a + 3, exp_b, (n_a + 3) * alpha + exp_b, 1 - ((n_a + 3) * alpha + exp_b)))
    print("cannot fail (reported, not counted): %s" % sorted("%s/%s %.2f 1/%d %s" % (c[0], c[1], c[2], round(1 / c[3]), q) for c, q in cannot_fail))

    # clean moves in the five model cells with none at stage 1: zero in 16 x 256 + 16 x 576 point-replicas
    pr_stage1 = 16 * 256 + 16 * 576
    write_prediction(out_path, pred)
    print("\nwrote %s (%d rows, %d scored)" % (out_path, len(pred), sum(1 for r in pred if r["scored"] == "yes")))
    print("\nclean moves, model cells with none at stage 1: 0 in %d point-replicas; 95 %% upper bound at L = 16 (16 x 1,024): %.2f a cell (rule of three, -ln 0.05 = %.3f)"
          % (pr_stage1, -math.log(0.05) * 16 * 1024 / pr_stage1, -math.log(0.05)))


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
