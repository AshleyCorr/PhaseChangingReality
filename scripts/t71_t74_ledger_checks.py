"""Item 4 of `docs/design/next_experiments_2026-10-26.md`: is the fed-slab burp anything beyond the ledger? Exact
checks on the registered T71 and T74 files (no run; nothing here scores a registered verdict). Usage:

    PYTHONPATH="src;scripts" python scripts/t71_t74_ledger_checks.py [--null] [--json OUT] results/t71_*.csv results/t74_*.csv

The files are named on the command line (never a glob). A file with an `edge_in` column is a T74 file (fed through
the curl edge); one without is a T71 file (fed at every point). Everything printed is arithmetic on the logged rows,
except the bucket null (`--null`), which is a graph-free re-run described below.

Definitions (exact; all per replica, over the fed-phase births: in T74 the daughters born at sweeps <= n_fed, the same
births `scripts/analyse_t74.py` takes the periods from; in T71 every daughter, the run having one phase).

A birth is a rise of `daughters` between consecutive logged rows (every birth sweep is logged, `log_events`); its
energy is the rise of `daughter_energy` (split evenly and flagged if a row adds more than one). The interval before
birth k runs from the row of birth k-1 (or, for the first birth, the start: F0 = a N_slab, stores 0, nothing fed,
nothing across the edge, no daughter, no clean move; F0 is also recovered from the first row's residual and checked) to
the row of birth k, both rows being the slab just after a split.

(a) The ledger identity. Over that interval, with in = the rise of `edge_in` (T74) or of `fed` (T71), dF = the rise of
    F, daughter = the birth's energy, carry = the rise of `stores`:
        in - dF - daughter - carry = 0.
    This is the slab's own ledger (F + stores + daughter - in = F0; in T74, E_out + far = fed - edge_in, so it is the
    file's residual column restricted to the slab). IDENTITY HOLDS for a file if every fed-phase birth meets it to
    1e-9 (absolute), FAILS otherwise. The item wrote the identity without the carry ("energy in since the last birth -
    clean-switch spending - daughter's energy = 0"); dF is the clean-switch spending on an undamaged slab (both rows
    are post-split rows of the same width, where the lattice's F is 0, so dF is what the clean moves and any damage
    they left added), and the carry is what the item's form leaves out: the stores the slab holds at one birth against
    the last. It is reported as the item's remainder, in units of a N_slab, so that a reader can see how far the
    item's form is from exact.

(c) The wiring-free share. A birth is wiring-free if `n_clean` did not rise over its interval (the interval includes
    the birth sweep, since the clean attempts come before the layer proposal in a sweep). Widenings (a split from 6 to
    8, which releases nothing and makes no daughter, O158 (c)) are splits, not graph moves, and do not break a birth's
    wiring-free status; they are counted beside it. Per cell: the pooled share (wiring-free births over all births),
    the least replica share, and the word WIRING-FREE SHARE ABOVE 90 % if the pooled share is at least 0.90, BELOW
    otherwise.

(d) Seed-linked or memoryless, at lambda = 1.10, c = 2R (T74), per (L, feed) with the backgrounds pooled (the reservoir
    is isolated while fed, so the three backgrounds are replicas of one fed phase, as the registration pools them):
    a shortfall is a fed-phase daughter whose energy differs from a N_slab by more than 1e-9 (either way). The test,
    declared before computing: under the memoryless null every birth is a shortfall independently with one
    probability p (the pooled rate), so the replicas' shortfall counts k_i are Binomial(n_i, p); the statistic is the
    dispersion D = sum (k_i - n_i p)^2 / (n_i p (1 - p)); its p-value is the share of 20,000 draws from the null
    (numpy seed 20261026, counts n_i as observed, p the pooled rate) with D at least as large. SEED-LINKED if p < 0.05,
    MEMORYLESS otherwise; NO SHORTFALL if there is none. With a handful of shortfalls among thousands of births the
    test has little power, and that is said beside the word. The same is run on the T71 lambda = 1.10 cells.

(e) The clock count. Clean switches per birth: in T74 the fed-phase clean moves (`n_clean` at the row of sweep n_fed,
    always logged) over the fed-phase births, summed over the replicas of a (lambda, c, feed) setting, the backgrounds
    pooled; in T71 the run's clean moves over its births per (lambda, rate). Against N_slab across the sizes: the
    exponent w in switches per birth = A N^w by Poisson maximum likelihood on the counts (the births are the exposure),
    with a 95 % profile-likelihood range (deviance within 3.841 of its minimum). PER-BIRTH SWITCHES ∝ N if the range
    contains 1 and not 0; FIXED if it contains 0 and not 1; UNDECIDED if it contains both or neither (and NO MOVES if no
    size has a clean move). The counts are small (single digits in T74), and the words carry that.

The residual table (for (b)). Per (lambda, L, c, feed), the backgrounds pooled, the mean fed-phase period in sweeps
(`analyse_t74.read_cells`, unchanged) against the exact a N_slab / min(c, R), and the residual (mean - exact) / exact in
per cent; and the timing words. RESIDUAL PATTERN if, at every lambda and L, the setting with the largest |residual| is
c = 2R with meals of 256; NO PATTERN otherwise.

(b) The bucket null (`--null`). A graph-free re-run of each T74 replica from its own logged seed: the reservoir, the
meals, the relaxation at rho_fed toward g_bg_fed, `edge_flow` with the cap c both ways, the even inflow and the
water-filling outflow (`take_evenly`, taken per layer: within a layer every point holds the same, since nothing in this
null tells them apart), one layer proposal a sweep drawn from the same generator in the same order as
`run_layer_outside` (two draws consumed at the start for the numpy and numba seeds, then an axis, a position and
split-or-contract each sweep), the pool of the layers beside the cut paying a contraction in proportion and the gate
read on the two layers beside a cut, the release leaving into the daughter ledger, the energy being a N_slab at width 4
and 0 at widths 6 and 8 (exact on the lattice; max_width 8), the removed pair's stores going half to each spliced
neighbour, and the wrap case of `contract_layers` copied as the code has it (at x0 = w - 1 the state is renamed by +1
and x0 set to w - 2, so the pair removed is the one at the old w - 3 and w - 2 while the pool that paid is the one
around the old w - 1 and 0; ours, exact from the code, to verify; geometry is unchanged by it, the stores are not).
No wiring, no clean switch. It runs the fed phase (n_fed sweeps). Per replica it is compared birth by birth with the
log (sweep and energy, 1e-9); per setting it gives its own mean period and residual beside the log's. The test
checks it draw for draw against `run_layer_outside` with `move_set="layer"`, which is the registered code with its
wiring moves off.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t74 as A74  # noqa: E402

EXACT = 1e-9
SHARE_BAR = 0.90
NULL_DRAWS = 20000
NULL_SEED = 20261026
CHI2_95 = 3.841458820694124


# ----------------------------------------------------------------------------------------------- reading the files


def read_rows(path):
    rows = list(csv.DictReader(open(path, newline="")))
    kind = "t74" if rows and "edge_in" in rows[0] else "t71"
    return kind, rows


def replica_key(kind, r):
    if kind == "t74":
        return (float(r["lam"]), int(r["L"]), r["feed_label"], r["cap_label"], r["background"]), int(r["replica"])
    return (float(r["lam"]), int(r["L"]), float(r["rate"]), r["move_set"], r.get("release_to", "stores")), int(r["replica"])


def group(kind, rows):
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell, rep = replica_key(kind, r)
        cells[cell][rep].append(r)
    return cells


def start_row(kind, r0):
    """The virtual row before sweep 1, checked against the first row's residual."""
    lam, n_slab = float(r0["lam"]), int(r0["N_slab"])
    f0 = 4.0 * (lam - 1.0) * n_slab
    if kind == "t74":
        back = (float(r0["F"]) + float(r0["stores"]) + float(r0["daughter_energy"]) + float(r0["E_out"])
                + float(r0["far"]) - float(r0["fed"]) + float(r0["leaked"]) - float(r0["residual"]) - float(r0["e_out0"]))
    else:
        back = (float(r0["F"]) + float(r0["stores"]) + float(r0["daughter_energy"]) - float(r0["fed"])
                + float(r0["leaked"]) - float(r0["residual"]))
    if abs(back - f0) > 1e-6:
        raise ValueError("F0 recovered from the first row (%r) is not a N_slab (%r)" % (back, f0))
    return dict(sweep=0, F=f0, stores=0.0, inflow=0.0, daughter_energy=0.0, daughters=0, n_clean=0, n_split=0,
                n_contract=0)


def inflow(kind, r):
    return float(r["edge_in"]) if kind == "t74" else float(r["fed"])


def births_of(kind, rows):
    """Per replica: a list of birth dicts (fed phase only) and the replica's counts."""
    rows = sorted(rows, key=lambda r: int(r["sweep"]))
    r0 = rows[0]
    lam, n_slab = float(r0["lam"]), int(r0["N_slab"])
    a_n = 4.0 * (lam - 1.0) * n_slab
    n_fed = int(r0["n_fed"]) if kind == "t74" else None
    prev = start_row(kind, r0)
    births, flagged = [], 0
    for r in rows:
        d = int(r["daughters"])
        k = d - prev["daughters"]
        cur = dict(sweep=int(r["sweep"]), F=float(r["F"]), stores=float(r["stores"]), inflow=inflow(kind, r),
                   daughter_energy=float(r["daughter_energy"]), daughters=d, n_clean=int(r["n_clean"]),
                   n_split=int(r["n_split"]), n_contract=int(r["n_contract"]))
        if k > 0:
            if k > 1:
                flagged += 1
            e = (cur["daughter_energy"] - prev["daughter_energy"]) / k
            b = dict(sweep=cur["sweep"], energy=e, inflow=cur["inflow"] - prev["inflow"], dF=cur["F"] - prev["F"],
                     carry=cur["stores"] - prev["stores"], clean=cur["n_clean"] - prev["n_clean"],
                     widenings=(cur["n_split"] - prev["n_split"]) - k, period=cur["sweep"] - prev["sweep"],
                     fed=(n_fed is None or cur["sweep"] <= n_fed), k=k)
            b["identity"] = b["inflow"] - b["dF"] - k * e - b["carry"]
            b["remainder"] = b["inflow"] - b["dF"] - k * e           # the item's form, without the carry
            b["short"] = abs(e - a_n) > EXACT
            births.append(b)
            prev = cur                   # the interval before the next birth starts at this birth's row
    fed_births = [b for b in births if b["fed"]]
    if kind == "t74":
        at_fed = [r for r in rows if int(r["sweep"]) == n_fed]
        if not at_fed:
            raise ValueError("the row at sweep n_fed is missing")
        clean_fed = int(at_fed[0]["n_clean"])
    else:
        clean_fed = int(rows[-1]["n_clean"])
    return dict(a_n=a_n, n_slab=n_slab, births=fed_births, all_births=births, flagged=flagged, clean_fed=clean_fed,
                seed=int(r0["seed"]), r0=r0)


# ---------------------------------------------------------------------------------------------------------- (a), (c)


def check_a_c(kind, cells):
    out = {}
    for cell, reps in sorted(cells.items()):
        reads = {rep: births_of(kind, rs) for rep, rs in sorted(reps.items())}
        bs = [b for x in reads.values() for b in x["births"]]
        a_n = next(iter(reads.values()))["a_n"]
        max_ident = max((abs(b["identity"]) for b in bs), default=0.0)
        rem = [b["remainder"] / a_n for b in bs]
        shares = {rep: (sum(1 for b in x["births"] if b["clean"] == 0) / len(x["births"]) if x["births"] else None)
                  for rep, x in reads.items()}
        n_free = sum(1 for b in bs if b["clean"] == 0)
        pooled = n_free / len(bs) if bs else None
        out[cell] = dict(
            n=len(reads), births=len(bs), a_n=a_n, max_identity=max_ident,
            identity_word="IDENTITY HOLDS" if max_ident <= EXACT else "IDENTITY FAILS",
            remainder_max=max((abs(x) for x in rem), default=0.0), remainder_mean=float(np.mean(rem)) if rem else None,
            remainder_sd=float(np.std(rem)) if rem else None,
            wiring_free=n_free, pooled_share=pooled, min_share=min((s for s in shares.values() if s is not None), default=None),
            reps_below=sum(1 for s in shares.values() if s is not None and s < SHARE_BAR),
            share_word=(None if pooled is None else
                        ("WIRING-FREE SHARE ABOVE 90 %" if pooled >= SHARE_BAR else "WIRING-FREE SHARE BELOW 90 %")),
            widenings=sum(b["widenings"] for b in bs), clean_fed=sum(x["clean_fed"] for x in reads.values()),
            short=sum(1 for b in bs if b["short"]), flagged=sum(x["flagged"] for x in reads.values()),
            per_replica={rep: dict(births=len(x["births"]), short=sum(1 for b in x["births"] if b["short"]),
                                   clean_fed=x["clean_fed"], share=shares[rep], seed=x["seed"])
                         for rep, x in reads.items()},
            reads=reads)
    return out


# --------------------------------------------------------------------------------------------------------------- (d)


def dispersion(ks, ns, p):
    ks, ns = np.asarray(ks, float), np.asarray(ns, float)
    v = ns * p * (1.0 - p)
    return float(((ks - ns * p) ** 2 / v).sum())


def check_d_events(ks, draws=NULL_DRAWS, seed=NULL_SEED):
    """The same dispersion test on the fed-phase clean-move counts per replica (equal exposure: every replica has the
    same n_fed sweeps) against a Poisson null with the pooled mean. The shortfall births follow the first clean move
    (every birth after it can be off, since the split's release depends on where the cut falls against the damage), so
    the memoryless question is really about the clean moves, the events, not their consequences."""
    ks = np.asarray(ks, int)
    K = int(ks.sum())
    if K == 0:
        return dict(word="NO MOVES", moves=0, n_replicas=len(ks))
    m = K / len(ks)
    d_obs = float(((ks - m) ** 2 / m).sum())
    rng = np.random.default_rng(seed)
    sims = rng.poisson(m, size=(draws, len(ks)))
    d_sim = ((sims - m) ** 2 / m).sum(axis=1)
    p_value = float((d_sim >= d_obs - 1e-12).mean())
    return dict(word="SEED-LINKED" if p_value < 0.05 else "MEMORYLESS", moves=K, mean=m, D=d_obs, p_value=p_value,
                replicas_with=int((ks > 0).sum()), max_in_one=int(ks.max()), n_replicas=len(ks))


def check_d(ks, ns, draws=NULL_DRAWS, seed=NULL_SEED):
    """The declared test: the dispersion of the per-replica shortfall counts against the binomial null."""
    ks, ns = np.asarray(ks, int), np.asarray(ns, int)
    K, Ntot = int(ks.sum()), int(ns.sum())
    if K == 0:
        return dict(word="NO SHORTFALL", shortfalls=0, births=Ntot, p=None, D=None, p_value=None)
    p = K / Ntot
    d_obs = dispersion(ks, ns, p)
    rng = np.random.default_rng(seed)
    sims = rng.binomial(ns[None, :], p, size=(draws, len(ns)))
    d_sim = ((sims - ns * p) ** 2 / (ns * p * (1 - p))).sum(axis=1)
    p_value = float((d_sim >= d_obs - 1e-12).mean())
    return dict(word="SEED-LINKED" if p_value < 0.05 else "MEMORYLESS", shortfalls=K, births=Ntot, p=p, D=d_obs,
                p_value=p_value, replicas_with=int((ks > 0).sum()), max_in_one=int(ks.max()), n_replicas=len(ks))


# --------------------------------------------------------------------------------------------------------------- (e)


def poisson_exponent(counts, exposures, sizes):
    """Maximum likelihood of counts ~ Poisson(exposure * A * N^w) and the 95 % profile range of w."""
    c = np.asarray(counts, float)
    e = np.asarray(exposures, float)
    x = np.log(np.asarray(sizes, float))
    if c.sum() == 0:
        return None
    x0 = x.mean()

    def dev(w):
        mu_hat = c.sum() / (e * np.exp(w * (x - x0))).sum()
        mu = e * mu_hat * np.exp(w * (x - x0))
        return float(2.0 * (np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0) / mu), 0.0) - (c - mu)).sum())

    grid = np.linspace(-6.0, 8.0, 14001)
    ds = np.array([dev(w) for w in grid])
    i = int(ds.argmin())
    inside = grid[ds <= ds[i] + CHI2_95]
    return dict(w=float(grid[i]), lo=float(inside.min()), hi=float(inside.max()), deviance=float(ds[i]),
                lo_at_edge=bool(inside.min() <= grid[0]), hi_at_edge=bool(inside.max() >= grid[-1]))


def check_e(kind, ac):
    """Clean switches per birth across the sizes of each setting."""
    by_setting = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for cell, c in ac.items():
        if kind == "t74":
            lam, L, feed, cap, bg = cell
            key = (lam, cap, feed)
        else:
            lam, L, rate, move_set, release_to = cell
            key = (lam, rate, move_set, release_to)
        s = by_setting[key][L]
        s[0] += c["clean_fed"]
        s[1] += c["births"]
        s[2] = 4 * L * L
    out = {}
    for key, per_L in sorted(by_setting.items()):
        Ls = sorted(per_L)
        counts = [per_L[L][0] for L in Ls]
        births = [per_L[L][1] for L in Ls]
        ns = [per_L[L][2] for L in Ls]
        rates = [k / b if b else None for k, b in zip(counts, births)]
        fit = poisson_exponent(counts, births, ns) if len(Ls) >= 2 and any(births) else None
        if sum(counts) == 0:
            word = "NO MOVES"
        elif fit is None:
            word = "ONE SIZE"
        else:
            has1, has0 = fit["lo"] <= 1.0 <= fit["hi"], fit["lo"] <= 0.0 <= fit["hi"]
            word = ("PER-BIRTH SWITCHES ∝ N" if has1 and not has0 else "PER-BIRTH SWITCHES FIXED" if has0 and not has1
                    else "UNDECIDED")
        out[key] = dict(sizes=Ls, N=ns, counts=counts, births=births, rates=rates, fit=fit, word=word)
    return out


# ------------------------------------------------------------------------------------------------- the residual table


def t74_cell_summaries(rows):
    """`analyse_t74.read_cells` on one file's rows, keeping only what the residual table needs (the rows of 108 files
    do not fit in memory together)."""
    out = {}
    for key, c in A74.read_cells(rows).items():
        r0 = next(r for r in rows if replica_key("t74", r)[0] == key)
        out[key] = dict(periods=c["periods"], a_n=c["a_n"], c=c["c"], R=float(r0["R"]))
    return out


def residual_table(cells):
    groups = defaultdict(list)
    exact_of = {}
    for (lam, L, feed, cap, bg), c in cells.items():
        groups[(lam, L, cap, feed)].extend(c["periods"])
        exact_of[(lam, L, cap, feed)] = c["a_n"] / min(c["c"], c["R"])
    table = {}
    for key, periods in sorted(groups.items()):
        m, cv = A74.pooled(periods)
        ex = exact_of[key]
        table[key] = dict(mean=m, cv=cv, n=len(periods), exact=ex, residual_pct=(100.0 * (m - ex) / ex) if m else None)
    words = A74.timing(cells)
    pattern = True
    for (lam, L) in sorted({(k[0], k[1]) for k in table}):
        best = max((k for k in table if k[0] == lam and k[1] == L), key=lambda k: abs(table[k]["residual_pct"] or 0.0))
        if not (best[2] == "c2R" and best[3] == "meals256"):
            pattern = False
    return table, words, ("RESIDUAL PATTERN (meals of 256, c = 2R)" if pattern else "NO PATTERN"), cells


# ------------------------------------------------------------------------------------------------ (b) the bucket null


def take_evenly_layers(s, m, x):
    """`take_evenly` on per-layer stores: layer i holds m points of s[i] each. Removes x in place."""
    if x <= 0.0:
        return
    order = sorted(range(len(s)), key=lambda i: s[i])
    left, n_left = x, m * len(s)
    for i in order:
        share = left / n_left
        give = min(share, s[i])
        s[i] -= give
        left -= give * m
        n_left -= m
        if n_left == 0:
            break


def bucket_null(lam, L, n_out, c, n_fed, feed, amount, every, e_out0, rho_fed, g_bg_fed, gate, seed, max_width=8):
    """The graph-free re-run of one T74 replica's fed phase (module docstring (b)). Returns the births [(sweep, e)]
    and the per-sweep (stores_sum, e_out, edge_in) at every sweep."""
    a = 4.0 * (lam - 1.0)
    m = L * L
    n_slab = 4 * m
    a_n = a * n_slab
    rng = np.random.default_rng(seed)
    rng.integers(2 ** 31 - 1)
    rng.integers(2 ** 31 - 1)
    s = [0.0] * 4
    f = a_n
    e_out, edge_in, fed = float(e_out0), 0.0, 0.0
    births, trace = [], []
    for sweep in range(1, n_fed + 1):
        meal = (amount if (feed == "steady" or (sweep - 1) % every == 0) else 0.0)
        e_out += meal
        fed += meal
        if rho_fed > 0:
            e_out -= rho_fed * (e_out - n_out * g_bg_fed)
        w = len(s)
        n_in = w * m
        s_in = sum(s) * m
        x_eq = (e_out * n_in - s_in * n_out) / (n_in + n_out)
        flow = float(np.sign(x_eq) * min(c, abs(x_eq)))
        if flow > 0.0:
            per = flow / n_in
            s = [v + per for v in s]
        elif flow < 0.0:
            take_evenly_layers(s, m, -flow)
        e_out -= flow
        edge_in += flow
        rng.integers(1)
        x0 = int(rng.integers(w))
        contract = bool(rng.integers(2))
        if contract:
            if w >= 6:
                f_new = a_n if w - 2 == 4 else 0.0
                df = f_new - f
                layers = [x0 % w, (x0 + 1) % w, (x0 - 1) % w, (x0 + 2) % w]
                pool = sum(s[i] for i in layers) * m
                ok = True
                if df > 0:
                    if pool + 1e-12 < df:
                        ok = False
                    else:
                        for i in layers:
                            s[i] = s[i] - s[i] * (df / pool)
                elif pool + 1e-12 < gate:
                    ok = False
                if ok:
                    xx = x0 % w
                    if xx == w - 1:
                        s = [s[w - 1]] + s[:w - 1]        # the rename by +1 the code makes before the move
                        xx = w - 2
                    xa, xb, xm, xp = xx, xx + 1, (xx - 1) % w, (xx + 2) % w
                    half = 0.5 * (s[xa] + s[xb])
                    s[xm] += half
                    s[xp] += half
                    s = [v for i, v in enumerate(s) if i not in (xa, xb)]
                    f = f_new
        else:
            if w + 2 <= max_width:
                f_new = 0.0
                df = f_new - f
                layers = [x0 % w, (x0 + 1) % w]
                pool = sum(s[i] for i in layers) * m
                if not (pool + 1e-12 < gate):
                    xx = x0 % w
                    s = s[:xx + 1] + [0.0, 0.0] + s[xx + 1:]
                    if -df > 1e-9:
                        births.append((sweep, -df))
                    f = f_new
        trace.append((sweep, sum(s) * m, e_out, edge_in, len(s)))
    return births, trace


def null_params(r0):
    return dict(lam=float(r0["lam"]), L=int(r0["L"]), n_out=int(r0["n_out"]), c=float(r0["c"]),
                n_fed=int(r0["n_fed"]), feed=r0["feed"], amount=float(r0["meal"]), every=int(r0["every"]),
                e_out0=float(r0["e_out0"]), rho_fed=float(r0["rho_fed"]), g_bg_fed=float(r0["g_bg_fed"]),
                gate=float(r0["gate"]), seed=int(r0["seed"]))


def run_null(ac):
    """Per T74 cell: the null's births against the log's, and its periods."""
    out = {}
    for cell, c in sorted(ac.items()):
        n_fed = None
        match, mismatch, periods = 0, 0, []
        clean_free_match = 0
        for rep, x in c["reads"].items():
            r0 = x["r0"]
            p = null_params(r0)
            n_fed = p["n_fed"]
            births, _ = bucket_null(**p)
            logged = [(b["sweep"], b["energy"]) for b in x["births"]]
            same = (len(births) == len(logged)
                    and all(s1 == s2 and abs(e1 - e2) <= EXACT for (s1, e1), (s2, e2) in zip(births, logged)))
            if same:
                match += 1
                if x["clean_fed"] == 0:
                    clean_free_match += 1
            else:
                mismatch += 1
            sweeps = [s for s, _ in births]
            periods.extend(np.diff(sweeps).tolist() if len(sweeps) >= 2 else [])
        out[cell] = dict(match=match, mismatch=mismatch, periods=periods, clean_free_match=clean_free_match,
                         clean_free=sum(1 for x in c["reads"].values() if x["clean_fed"] == 0))
    return out


def null_table(null_cells, table):
    groups = defaultdict(list)
    for (lam, L, feed, cap, bg), c in null_cells.items():
        groups[(lam, L, cap, feed)].extend(c["periods"])
    out = {}
    for key, periods in sorted(groups.items()):
        m, cv = A74.pooled(periods)
        ex = table[key]["exact"]
        out[key] = dict(mean=m, cv=cv, n=len(periods), residual_pct=(100.0 * (m - ex) / ex) if m else None)
    return out


# ------------------------------------------------------------------------------------------------------------- main


def fmt_cell(kind, cell):
    if kind == "t74":
        lam, L, feed, cap, bg = cell
        return "lam=%.2f L=%2d %-8s %-6s %-5s" % (lam, L, feed, cap, bg)
    lam, L, rate, move_set, release_to = cell
    return "lam=%.2f L=%2d r=%-8g %-8s %-8s" % (lam, L, rate, move_set, release_to)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--null", action="store_true", help="run the graph-free bucket null on the T74 files")
    ap.add_argument("--json", default=None, help="write every number to this JSON file")
    args = ap.parse_args(argv)
    files = {"t71": [], "t74": []}
    report = dict(files={}, e={}, d={}, residual_table=None, null=None)
    t74_cells = {}
    all_ac = {"t71": {}, "t74": {}}
    for p in args.paths:
        kind, rows = read_rows(p)
        files[kind].append(p)
        cells = group(kind, rows)
        ac = check_a_c(kind, cells)
        all_ac[kind].update(ac)
        if kind == "t74":
            t74_cells.update(t74_cell_summaries(rows))
        del rows, cells
        worst = max(c["max_identity"] for c in ac.values())
        word = "IDENTITY HOLDS" if worst <= EXACT else "IDENTITY FAILS"
        report["files"][p] = dict(kind=kind, word=word, max_identity=worst)
        print("%-45s %s | max |identity| %.2e | births %d" % (Path(p).name, word, worst, sum(c["births"] for c in ac.values())))
    print()
    print("(a) and (c), per cell: identity | item's remainder (in - dF - daughter) in units of a N_slab: max |.|, mean, sd"
          " | wiring-free share pooled (least replica; replicas below 90 %) | clean moves in fed phase | short births "
          "| widenings")
    for kind in ("t71", "t74"):
        for cell, c in sorted(all_ac[kind].items()):
            print("%s | %s | rem max %.3g mean %+.3g sd %.3g | %s %.4f (%s; %d below) | clean %d | short %d | widen %d"
                  % (fmt_cell(kind, cell), c["identity_word"], c["remainder_max"], c["remainder_mean"] or 0.0,
                     c["remainder_sd"] or 0.0, c["share_word"], c["pooled_share"] if c["pooled_share"] is not None else -1,
                     "least %.3f" % c["min_share"] if c["min_share"] is not None else "-", c["reps_below"],
                     c["clean_fed"], c["short"], c["widenings"]))
    print()
    print("(d) shortfall counts per replica against the binomial (memoryless) null, lambda = 1.10, c = 2R")
    for kind in ("t74", "t71"):
        pools = defaultdict(lambda: ([], [], []))
        for cell, c in all_ac[kind].items():
            if kind == "t74":
                lam, L, feed, cap, bg = cell
                if not (abs(lam - 1.10) < 1e-9 and cap == "c2R"):
                    continue
                key = ("t74", lam, L, cap, feed)
            else:
                lam, L, rate, move_set, release_to = cell
                if not abs(lam - 1.10) < 1e-9:
                    continue
                key = ("t71", lam, L, rate, move_set)
            for rep, pr in c["per_replica"].items():
                pools[key][0].append(pr["short"])
                pools[key][1].append(pr["births"])
                pools[key][2].append(pr["clean_fed"])
        for key, (ks, ns, kc) in sorted(pools.items()):
            d = check_d(ks, ns)
            de = check_d_events(kc)
            report["d"][str(key)] = dict(shortfalls=d, clean_moves=de)
            print("%s | shortfall births: %s | %d of %d births in %d replicas%s" % (
                " ".join(str(k) for k in key), d["word"], d["shortfalls"], d["births"], len(ks),
                "" if d["p"] is None else (" (in %d replicas, at most %d in one) | p = %.4g | D = %.2f | p-value %.3f"
                                           % (d["replicas_with"], d["max_in_one"], d["p"], d["D"], d["p_value"]))))
            print("%s | clean moves (the events): %s | %d moves%s" % (
                " " * len(" ".join(str(k) for k in key)), de["word"], de["moves"],
                "" if de["moves"] == 0 else (" in %d of %d replicas (at most %d in one) | mean %.3f | D = %.2f | "
                                             "p-value %.3f" % (de["replicas_with"], de["n_replicas"], de["max_in_one"],
                                                               de["mean"], de["D"], de["p_value"]))))
    print()
    print("(e) clean switches per birth against N_slab (Poisson ML exponent w, 95 % profile range)")
    for kind in ("t71", "t74"):
        e = check_e(kind, all_ac[kind])
        report["e"][kind] = {str(k): v for k, v in e.items()}
        for key, v in e.items():
            fit = v["fit"]
            print("%s %s | sizes %s N %s | clean %s | births %s | per birth %s | %s%s" % (
                kind, key, v["sizes"], v["N"], v["counts"], v["births"],
                ["%.2e" % r if r is not None else "-" for r in v["rates"]], v["word"],
                "" if fit is None else " | w = %.2f (%.2f to %.2f)%s" % (
                    fit["w"], fit["lo"], fit["hi"], " [range reaches the grid's edge]" if fit["lo_at_edge"] or fit["hi_at_edge"] else "")))
    if t74_cells:
        print()
        table, words, pattern_word, cells = residual_table(t74_cells)
        report["residual_table"] = {str(k): v for k, v in table.items()}
        report["residual_pattern"] = pattern_word
        print("the residual table (T74, backgrounds pooled): mean fed-phase period against a N_slab / min(c, R)")
        for key, v in table.items():
            print("lam=%.2f L=%2d %-6s %-8s | mean %.3f (cv %.3f, %d) | exact %.2f | residual %+.2f %%"
                  % (key[0], key[1], key[2], key[3], v["mean"], v["cv"], v["n"], v["exact"], v["residual_pct"]))
        print(pattern_word)
        for key, (w, ratio, cvr) in sorted(words.items()):
            print("timing lam=%.2f L=%2d %-6s %-8s: %s" % (key[0], key[1], key[2], key[3], w))
        if args.null:
            print()
            print("(b) the bucket null, same seeds, fed phase: replicas whose births match the log birth for birth")
            nc = run_null(all_ac["t74"])
            tot_match = sum(c["match"] for c in nc.values())
            tot = sum(c["match"] + c["mismatch"] for c in nc.values())
            cf = sum(c["clean_free"] for c in nc.values())
            cfm = sum(c["clean_free_match"] for c in nc.values())
            for cell, c in nc.items():
                if c["mismatch"]:
                    print("%s | match %d mismatch %d (clean-free replicas %d, of which match %d)"
                          % (fmt_cell("t74", cell), c["match"], c["mismatch"], c["clean_free"], c["clean_free_match"]))
            print("all cells: %d of %d replicas match; of the %d replicas with no fed-phase clean move, %d match"
                  % (tot_match, tot, cf, cfm))
            nt = null_table(nc, table)
            report["null"] = dict(match=tot_match, total=tot, clean_free=cf, clean_free_match=cfm,
                                  table={str(k): v for k, v in nt.items()},
                                  cells={str(k): dict(match=v["match"], mismatch=v["mismatch"]) for k, v in nc.items()})
            print("null's residual beside the log's:")
            for key, v in nt.items():
                print("lam=%.2f L=%2d %-6s %-8s | null mean %.3f residual %+.2f %% | log mean %.3f residual %+.2f %%"
                      % (key[0], key[1], key[2], key[3], v["mean"], v["residual_pct"], table[key]["mean"],
                         table[key]["residual_pct"]))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str), newline="\n")
    return report


if __name__ == "__main__":
    main()
