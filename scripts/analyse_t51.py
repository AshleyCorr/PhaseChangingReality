"""The pre-registered T51 reading: how much scrap freezes in, against how slowly the new space cools. Usage:

    python scripts/analyse_t51.py [results_dir]

Implements PREREGISTRATION.md section T51 (written 2026-10-05, before any run) with its Amendments 1 and 2 (the same
day, before any run) on `<results_dir>/t51_*.csv` (scripts/run_scrap_freeze.py; the default directory is `results`).
Each replica has one row at the end of its opening and one row per block of each cooling; the row flagged `final` is
the end of the hold. Where the saved final graphs (`t51_*_adj/`) are present they are read again, exactly as T37 reads
its end states, and set beside those rows.

Per replica and cooling time: columns and energy left at the end of the opening (c0, h0) and at the end of the hold
(c1, h1). A replica whose final graph has fewer than 90 % of its points at d = 2 is MELTED OR DEFECTED: it is reported
and left out of everything below except the count of such replicas and the frozen share over all opened replicas.

What is scored does not use the count at the end of the opening (Amendment 2). Every cooling time starts from the same
opened sheet, so what two coolings leave can be set against each other directly. C(t) is the columns at the end of the
hold after a cooling of t fair sweeps, summed over the replicas kept in both of the two cells being compared, however
many those turn out to be (the 300,000 cell has only the first 40 of the 80 replicas; Amendment 1).

  P1 (the fair clock)  at L = 256, C(30,000) / C(10,000) within 0.20 of 0.99 and C(100,000) / C(10,000) within 0.20
                       of 0.47, T25's own ratios at 96 points. Not read while a cell is missing or C(10,000) is zero.
  P2 (the size check)  at t_cool = 10,000 the columns left per column of tube at L = 128, 256 and 512 agree pair by
                       pair within two standard errors of their difference (the two errors added in quadrature).
  The verdict          on the decade ratios R(t) = S(10 t) / S(t) at t = 10,000 and 30,000, L = 256. On shared
                       replicas R(t) is C(10 t) / C(t), the count at the end of the opening cancelling. GENTLE if both
                       lie in [0.25, 0.85]; CLIFF if either is below 0.25; FROZEN if both are above 0.85; MIXED
                       otherwise; CLIFF if S(10,000) or S(30,000) is zero, which is C(t) = 0.

Reported, not scored:

  Survival S(t_cool)   sum of c1 over sum of c0, over the replicas kept. S_E is the same ratio for the energy left.
                       They are marked as able to exceed 1: columns can be born after the count at the end of the
                       opening, as pieces still relaxing settle into columns (Amendment 2).
  Frozen share         h1 per point over the release per point, 4(lambda - 1); the mean over replicas.
  Columns left per column of tube   c1 / L; the mean over replicas. A tube 4 x L has L columns. (P2 is scored on it.)
  Also: what becomes of the "other" pieces, the share MELTED OR DEFECTED, the columns per tube at the end of the
  opening (the seed count is not re-measured here), and beside each decade ratio the ratio of the two cells' own S,
  each over all its replicas, with the verdict that would give if it differs.

Standard errors of the ratios (S, S_E and every C(a) / C(b)) come from resampling replicas: BOOT_N resamples drawn
with replacement by a generator seeded afresh with BOOT_SEED on every call. Standard errors of the two means are the
ordinary ones, sd / sqrt(n). Pure functions of parsed rows, tested in tests/test_t51.py.
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t37 import end_state, leftovers                    # noqa: E402

FLAT = 0.90                                  # a final graph below this share of points at d = 2 is MELTED OR DEFECTED
P1_L, P1_BASE, P1_WINDOW = 256, 10000, 0.20
P1_TARGETS = {30000: 0.99, 100000: 0.47}     # T25's (14/18) / (15/19) and (7/19) / (15/19), as Amendment 2 writes them
P2_T, P2_LS = 10000, (128, 256, 512)
VERDICT_L, DECADES = 256, (10000, 30000)
LOW, HIGH = 0.25, 0.85
BOOT_N, BOOT_SEED = 2000, 51
EPS = 1e-12                                  # so that a number exactly on a line counts as inside it


def load(results="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(results) / "t51_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return rows


def openings(rows):
    """({(L, replica): the reading at the end of the opening}, the (L, replica) whose opening rows disagree).

    Every config that names a replica opens it afresh from the same seed. The paired design says those are one and
    the same sheet, so their rows must agree to the last digit; a disagreement is returned, not averaged away.
    """
    out, clash = {}, set()
    for r in rows:
        if r["phase"] != "opening":
            continue
        key = (int(r["L"]), int(r["replica"]))
        rec = dict(N=int(r["N"]), lam=float(r["lam"]), open_sweeps=int(r["open_sweeps"]), columns=int(r["columns"]),
                   others=int(r["others"]), n_d1=int(r["n_d1"]), largest_d1=int(r["largest_d1"]),
                   flat_share=float(r["flat_share"]), h=float(r["h"]), opened=(r["opened"] == "True"))
        if key in out and out[key] != rec:
            clash.add(key)
        out.setdefault(key, rec)
    return out, sorted(clash)


def finals(rows):
    """{(L, t_cool, replica): the row at the end of the hold}."""
    return {(int(r["L"]), int(r["t_cool"]), int(r["replica"])): r
            for r in rows if r["phase"] != "opening" and r["final"] == "True"}


def cells(rows):
    """{(L, t_cool): one record per replica cooled at t_cool, with its opening and its final reading}."""
    first, _ = openings(rows)
    out = defaultdict(list)
    for (length, t_cool, rep), r in sorted(finals(rows).items()):
        o = first[(length, rep)]
        flat = float(r["flat_share"])
        out[(length, t_cool)].append(dict(
            replica=rep, N=o["N"], lam=o["lam"], c0=o["columns"], c1=int(r["columns"]), o0=o["others"],
            o1=int(r["others"]), h0=o["h"], h1=float(r["h"]), flat=flat, melted=flat < FLAT))
    return dict(out)


def mean_se(v):
    """(mean, standard error of the mean); the error is nan for fewer than two values."""
    v = np.asarray(list(v), dtype=float)
    if len(v) == 0:
        return math.nan, math.nan
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else math.nan)


def ratio_se(num, den):
    """(sum(num) / sum(den), its standard error from resampling replicas).

    num and den hold one number per replica. The error is the spread of the ratio over BOOT_N resamples of the
    replicas; a resample whose denominator is zero is left out. The same numbers always give the same error.
    """
    num, den = np.asarray(list(num), dtype=float), np.asarray(list(den), dtype=float)
    if len(num) == 0 or den.sum() == 0:
        return math.nan, math.nan
    ratio = float(num.sum() / den.sum())
    if len(num) < 2:
        return ratio, math.nan
    pick = np.random.default_rng(BOOT_SEED).integers(0, len(num), size=(BOOT_N, len(num)))
    tops, bottoms = num[pick].sum(axis=1), den[pick].sum(axis=1)
    ok = bottoms > 0
    return ratio, (float((tops[ok] / bottoms[ok]).std(ddof=1)) if ok.sum() > 1 else math.nan)


def kept(recs):
    """The replicas that enter the counts: every one that is not MELTED OR DEFECTED."""
    return [r for r in recs if not r["melted"]]


def survival(recs):
    """S over the records given: columns at the end of the hold over columns at the end of the opening, each summed
    over the replicas; nan if no replica had a column at the end of its opening. Reported, not scored."""
    at_opening = sum(r["c0"] for r in recs)
    return sum(r["c1"] for r in recs) / at_opening if at_opening else math.nan


def cell_summary(recs, length):
    """Everything reported for one (L, t_cool) cell. Pairs are (value, standard error)."""
    ok = kept(recs)

    def share(r):
        return r["h1"] / (r["N"] * 4.0 * (r["lam"] - 1.0))
    return dict(n=len(recs), kept=len(ok), melted=len(recs) - len(ok),
                S=ratio_se([r["c1"] for r in ok], [r["c0"] for r in ok]),
                S_E=ratio_se([r["h1"] for r in ok], [r["h0"] for r in ok]),
                frozen=mean_se(share(r) for r in ok), frozen_all=mean_se(share(r) for r in recs),
                per_column=mean_se(r["c1"] / length for r in ok),
                columns0=mean_se(r["c0"] for r in ok), columns1=mean_se(r["c1"] for r in ok),
                others0=mean_se(r["o0"] for r in ok), others1=mean_se(r["o1"] for r in ok),
                others_ratio=ratio_se([r["o1"] for r in ok], [r["o0"] for r in ok]))


def paired(cell_a, cell_b):
    """Two cells cut down to the replicas kept in both, in the same order: the same opened sheets cooled two ways."""
    a = {r["replica"]: r for r in kept(cell_a)}
    b = {r["replica"]: r for r in kept(cell_b)}
    both = sorted(set(a) & set(b))
    return [a[k] for k in both], [b[k] for k in both]


def column_ratio(cell_a, cell_b):
    """C(b) / C(a): the columns at the end of the hold in cell b over those in cell a, each summed over the replicas
    kept in both cells. No count at the end of the opening comes into it.

    Returns None if the cells share no replica. Otherwise dict(ratio, se, n, c_a, c_b): n shared replicas, the two
    sums, and the ratio with its error from resampling the shared replicas; the ratio is None when C(a) is zero.
    """
    in_a, in_b = paired(cell_a, cell_b)
    if not in_a:
        return None
    c_a, c_b = sum(r["c1"] for r in in_a), sum(r["c1"] for r in in_b)
    if c_a == 0:
        return dict(ratio=None, se=math.nan, n=len(in_a), c_a=c_a, c_b=c_b)
    ratio, se = ratio_se([r["c1"] for r in in_b], [r["c1"] for r in in_a])
    return dict(ratio=ratio, se=se, n=len(in_a), c_a=c_a, c_b=c_b)


def score_p1(ratio_by_t):
    """P1 as Amendment 2 restates it, on {t: C(t) / C(10,000)} for t = 30,000 and 100,000 at L = 256.

    A ratio that cannot be read (its cell is missing, or C(10,000) is zero) is absent from the dict or None. Returns
    ({t: (ratio, target, within 0.20)}, holds). The third entry is None for an unread ratio; holds is None, NOT READ,
    while either ratio is unread, and otherwise True only if both ratios lie within their windows.
    """
    lines = {}
    for t, target in P1_TARGETS.items():
        ratio = ratio_by_t.get(t)
        lines[t] = (ratio, target, None if ratio is None else bool(abs(ratio - target) <= P1_WINDOW + EPS))
    marks = [m for _, _, m in lines.values()]
    return lines, (None if None in marks else all(marks))


def score_p2(per_column_by_l):
    """P2 on {L: (mean, standard error)} of the columns left per column of tube at t_cool = 10,000.

    Returns ({(La, Lb): (difference, standard error of the difference, agree)}, holds). A pair agrees when the
    difference is at most two standard errors of the difference; holds is None while a size is missing.
    """
    pairs = {}
    for i, a in enumerate(P2_LS):
        for b in P2_LS[i + 1:]:
            if a not in per_column_by_l or b not in per_column_by_l:
                pairs[(a, b)] = (math.nan, math.nan, None)
                continue
            (ma, sa), (mb, sb) = per_column_by_l[a], per_column_by_l[b]
            diff, se = ma - mb, math.hypot(sa, sb)
            unread = math.isnan(diff) or math.isnan(se)
            pairs[(a, b)] = (diff, se, None if unread else bool(abs(diff) <= 2.0 * se + EPS))
    marks = [m for _, _, m in pairs.values()]
    return pairs, (False if False in marks else None if None in marks else True)


def decade_ratio(s_t, s_10t):
    """S(10 t) / S(t) from two survivals. None when S(t) is zero, which the pre-registration scores as CLIFF."""
    return None if s_t == 0 else s_10t / s_t


def verdict(r_a, r_b):
    """GENTLE, CLIFF, FROZEN or MIXED from R(10,000) and R(30,000); None stands for a zero S(10,000) or S(30,000)."""
    if r_a is None or r_b is None:
        return "CLIFF"
    if r_a < LOW - EPS or r_b < LOW - EPS:
        return "CLIFF"
    if r_a <= HIGH + EPS and r_b <= HIGH + EPS:
        return "GENTLE"
    if r_a > HIGH + EPS and r_b > HIGH + EPS:
        return "FROZEN"
    return "MIXED"


def decade(cell_t, cell_10t):
    """One decade ratio R(t), or None if the two cells share no replica.

    R is C(10 t) / C(t) over the replicas kept in both cells: S(10 t) / S(t) on those replicas, with the count at the
    end of the opening cancelled (Amendment 2). It is None when C(t) is zero, which is S(t) = 0 and is scored as
    CLIFF. `unpaired` is the ratio of the two cells' own S, each over all the replicas it kept; reported only.
    """
    shared = column_ratio(cell_t, cell_10t)
    if shared is None:
        return None
    own_t, own_10t = survival(kept(cell_t)), survival(kept(cell_10t))
    return dict(R=shared["ratio"], se=shared["se"], n=shared["n"], c_t=shared["c_a"], c_10t=shared["c_b"],
                unpaired=(math.nan if math.isnan(own_t) or math.isnan(own_10t) else decade_ratio(own_t, own_10t)))


def summarize(rows):
    """The whole pre-registered reading as one dict: table, P1, P2, the two decade ratios and the verdict."""
    first, clash = openings(rows)
    cs = cells(rows)
    table = {key: cell_summary(recs, key[0]) for key, recs in sorted(cs.items())}
    base = cs.get((P1_L, P1_BASE))
    p1_ratios = {t: (column_ratio(base, cs[(P1_L, t)]) if base is not None and (P1_L, t) in cs else None)
                 for t in P1_TARGETS}
    p1_lines, p1 = score_p1({t: c["ratio"] for t, c in p1_ratios.items() if c is not None})
    p2_pairs, p2 = score_p2({ll: table[(ll, P2_T)]["per_column"] for ll in P2_LS if (ll, P2_T) in table})
    ratios = {t: (decade(cs[(VERDICT_L, t)], cs[(VERDICT_L, 10 * t)])
                  if (VERDICT_L, t) in cs and (VERDICT_L, 10 * t) in cs else None) for t in DECADES}
    read = all(ratios[t] is not None for t in DECADES)
    own = [ratios[t]["unpaired"] for t in DECADES] if read else []
    own_read = read and not any(isinstance(v, float) and math.isnan(v) for v in own)
    return dict(openings=first, clash=clash, table=table, p1_ratios=p1_ratios, p1_lines=p1_lines, p1=p1,
                p2_pairs=p2_pairs, p2=p2, ratios=ratios,
                verdict=(verdict(*[ratios[t]["R"] for t in DECADES]) if read else "NOT READ"),
                verdict_unpaired=(verdict(*own) if own_read else "NOT READ"))


def check_saved(results, rows):
    """Read every saved final graph again, as T37 reads its end states, and set it beside its row.

    Returns (graphs read, the files whose reading differs from the row at the end of their hold).
    """
    last, read, bad = finals(rows), 0, []
    for f in sorted(glob.glob(str(Path(results) / "t51_*_adj" / "*.npz"))):
        z = np.load(f)
        adj = z["adj"]
        r = last.get((adj.shape[0] // 4, int(z["t_cool"]), int(z["replica"])))
        columns, others = leftovers(adj)
        h, flat, _ = end_state(adj, float(z["lam"]))
        read += 1
        if r is None or (columns, others) != (int(r["columns"]), int(r["others"])) \
                or abs(h - float(r["h"])) > 1e-9 or abs(flat - float(r["flat_share"])) > 1e-12:
            bad.append(f)
    return read, bad


def _pm(pair, digits=2):
    return "%.*f +- %.*f" % (digits, pair[0], digits, pair[1])


def _mark(holds):
    return "NOT READ" if holds is None else ("HOLDS" if holds else "FAILS")


def p1_text(s):
    """The lines printed for P1: each ratio with its resampling error and its target, or why it is not read."""
    out = ["P1 (the fair clock), L = %d, what two coolings of the same opened sheets leave: %s"
           % (P1_L, _mark(s["p1"]))]
    for t, (ratio, target, ok) in s["p1_lines"].items():
        c, name = s["p1_ratios"][t], "C(%d) / C(%d)" % (t, P1_BASE)
        if c is None:
            out.append("  %s against T25's %.2f: not read, a cell is missing" % (name, target))
        elif ratio is None:
            out.append("  %s against T25's %.2f: not read. C(%d) is zero over the %d shared replicas: no column "
                       "survived that cooling, so there is nothing to compare with" % (name, target, P1_BASE, c["n"]))
        else:
            out.append("  %s = %.3f +- %.3f against T25's %.2f (columns %d over %d, on %d shared replicas): %s"
                       % (name, ratio, c["se"], target, c["c_b"], c["c_a"], c["n"],
                          ("within %.2f" if ok else "outside %.2f") % P1_WINDOW))
    return out


def main(results="results"):
    rows = load(results)
    if not rows:
        print("no T51 results yet")
        return
    s = summarize(rows)
    print("T51, read by the rules of PREREGISTRATION.md section T51 and its Amendments 1 and 2 (all written "
          "2026-10-05, before any run).")
    print("Errors of ratios (S, S_E, C / C, R): %d resamples of replicas, generator seed %d. Errors of means: "
          "sd / sqrt(n).\n" % (BOOT_N, BOOT_SEED))
    if s["clash"]:
        print("WARNING: the opening rows of these (L, replica) differ between configs, so the design is not paired "
              "there: %s\n" % s["clash"])
    print("The opening (to %.0f %% of points at d = 2 and no piece of points at d = 1 above eight):" % (100 * FLAT))
    for length in sorted(set(k[0] for k in s["openings"])):
        recs = [o for (ll, _), o in sorted(s["openings"].items()) if ll == length]
        done = [o for o in recs if o["opened"]]
        shut = ["replica %d (flat share %.3f, largest piece at d = 1 %d)" % (rep, o["flat_share"], o["largest_d1"])
                for (ll, rep), o in sorted(s["openings"].items()) if ll == length and not o["opened"]]
        line = "L=%-4d opened %d of %d" % (length, len(done), len(recs))
        if done:
            sweeps = [o["open_sweeps"] for o in done]
            line += (" | sweeps to open: mean %.0f, least %d, most %d | columns per tube %s | others per tube %s | "
                     "flat share %.3f | energy left per point %.3f"
                     % (np.mean(sweeps), min(sweeps), max(sweeps), _pm(mean_se(o["columns"] for o in done)),
                        _pm(mean_se(o["others"] for o in done)), np.mean([o["flat_share"] for o in done]),
                        np.mean([o["h"] / o["N"] for o in done])))
        print(line)
        if shut:
            print("       not opened, not followed: " + "; ".join(shut))
    print("(the seed count is not re-measured here; columns per seed need T37's seeds per tube beside these)\n")
    print("The cells (replicas MELTED OR DEFECTED are left out of S, S_E and the means):")
    for (length, t_cool), c in s["table"].items():
        print("L=%-4d t_cool=%-7d replicas %2d, kept %2d, MELTED OR DEFECTED %d (%.2f) | S %s | S_E %s | frozen share "
              "%s (all opened replicas: %.4f) | columns left per column of tube %s | columns per tube %.2f -> %.2f | "
              "others per tube %.2f -> %.2f (ratio %s)"
              % (length, t_cool, c["n"], c["kept"], c["melted"], c["melted"] / c["n"], _pm(c["S"]), _pm(c["S_E"]),
                 _pm(c["frozen"], 4), c["frozen_all"][0], _pm(c["per_column"], 4), c["columns0"][0], c["columns1"][0],
                 c["others0"][0], c["others1"][0], _pm(c["others_ratio"])))
    print("(S and S_E are set against the count and the energy at the end of the opening and can exceed 1: columns "
          "can be born after that count. They are reported, not scored; nothing below uses that count. Amendment 2.)")
    print()
    print("\n".join(p1_text(s)))
    print("P2 (the size check), t_cool = %d: %s" % (P2_T, _mark(s["p2"])))
    for (a, b), (diff, se, ok) in s["p2_pairs"].items():
        print("  L=%d against L=%d: difference %.4f, two standard errors %.4f: %s"
              % (a, b, diff, 2 * se, "not read" if ok is None else "agree" if ok else "differ"))
    print("\nThe decade ratios, L = %d, each over the replicas kept in both of its cells:" % VERDICT_L)
    for t in DECADES:
        r = s["ratios"][t]
        if r is None:
            print("  R(%d) = S(%d) / S(%d): not read" % (t, 10 * t, t))
            continue
        value = "undefined, C(%d) is zero" % t if r["R"] is None else "%.3f +- %.3f" % (r["R"], r["se"])
        own = "undefined" if r["unpaired"] is None else "%.3f" % r["unpaired"]
        print("  R(%d) = S(%d) / S(%d) = C(%d) / C(%d) = %s (columns %d over %d, on %d shared replicas); the ratio of "
              "the two cells' own S, each over its own replicas, is %s"
              % (t, 10 * t, t, 10 * t, t, value, r["c_10t"], r["c_t"], r["n"], own))
    print("  (the %d cell has only the first 40 of the 80 replicas, so R(%d) is taken over those 40 at most)"
          % (10 * DECADES[1], DECADES[1]))
    print("\nVERDICT: %s" % s["verdict"])
    if s["verdict_unpaired"] != s["verdict"]:
        print("  (with each cell's own S instead of the shared replicas it would read %s)" % s["verdict_unpaired"])
    read, bad = check_saved(results, rows)
    if read:
        print("\nSaved final graphs read again as T37 reads its end states: %d, of which %d differ from their rows%s"
              % (read, len(bad), "" if not bad else ": " + ", ".join(bad)))


if __name__ == "__main__":
    main(*sys.argv[1:])
