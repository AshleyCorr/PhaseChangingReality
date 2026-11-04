"""The four pre-registered observables, from tempered energy histograms. Usage:

    python scripts/analyse_t6_hist.py t6_lam0_control

Prints to the screen and writes nothing. Implements PREREGISTRATION.md section T6 and makes no
choices of its own.

HOW THE TRANSITION COUPLING IS FOUND. A histogram measured at coupling g gives the histogram at
a nearby g' by multiplying each energy's count by exp(-H(1/g' - 1/g)); this is standard
single-histogram reweighting. Scanning g' and asking where the two humps carry equal weight
locates the transition. Reweighting is only trustworthy while the two distributions overlap, so
the shift is reported and one that is too large is refused rather than quietly used.

DISCRETENESS. [Kelly22] warns that a discrete action spectrum can make one hump look like
several. Two separate things can fake a pair of humps, and each needs its own guard.

  A comb, period two. If the reachable energies sit on a coarser lattice than the recorded bins,
  the histogram alternates up-down-up-down from one bin to the next. Every other bin is then a
  local maximum, so a minimum separation does NOT help: at a period of two bins there are
  spurious peaks at every even spacing, including whatever separation you demanded. The guard is
  to smooth with a binomial [1,2,1]/4 filter, which annihilates a one-bin alternation exactly
  (it has a zero at that frequency) while leaving structure tens of bins wide almost untouched.
  Peaks, valley and depth are all measured on the smoothed curve, which can only shrink a real
  barrier, never invent one.

  A comb of longer period, which is the one this model actually produces. The energy is
  16(N - S) + 4*lambda*X, a sum over two integers with two different quanta, so which energies
  are reachable at all -- and how many ways each can be reached -- depends on arithmetic between
  16 and 4*lambda. When 4*lambda divides 16 the reachable energies form a uniform lattice and
  there is no comb: lambda = 0 (quantum 16) and lambda = 1 (quantum 4) are both of this kind.
  When it does not, the histogram develops teeth. Measured at lambda = 1.25, where the quanta are
  16 and 5, the counts at N = 36 ran 2899, 30, 181, 1024, 2046, 1964, 34 on consecutive levels --
  teeth about every five levels -- and the analysis duly reported two of those teeth as humps 19
  levels apart with a barrier of 1.7, which is arithmetic and not physics. No fixed smoothing
  width can handle this, because the period depends on lambda.

  The guard that does work is coarse-graining. A real barrier is a feature of the energy
  landscape and survives looking at it more coarsely; a comb tooth is a feature of the lattice
  and dies as soon as the bins are wider than the teeth. So the fit must be found BOTH at full
  resolution and again after rebinning the visited range into COARSE_BINS wide bins, with the two
  humps landing in about the same places. Failing the coarse check vetoes the fit. Passing it
  reports the full-resolution numbers, so nothing is lost in accuracy -- the coarse pass is only
  ever a veto.

  Counting noise. A valley one standard error deep is not a valley. The depth is required to
  beat the counting error on the three bins it is built from, by NOISE_SIGMAS.

AN EMPTY BIN IS NOT A DEEP BIN. A bin with no counts means the run never went there, which is
not the same as the run going there rarely, and the difference is not a detail: taking the
logarithm of nothing produces a valley of unlimited depth out of a place we have no information
about. Measured here, that single confusion invented a barrier in six of twenty-five histograms
that were one hump plus counting noise, each time reporting the same depth of about 690, which
is nothing but the logarithm of the floor the code had clamped to. So the analysis is restricted
to the longest unbroken stretch of energies the run actually visited. This matches what a
barrier means: to measure the cost of crossing a valley the run has to have crossed it. A valley
so deep the run never got over it does not show up here as a big number, it shows up as no
round trips, which is a gate in its own right.

HOW FAR A REWEIGHT MAY GO. Reweighting multiplies each bin by exp(-H(1/g' - 1/g)), which over a
wide energy range is a very steep tilt, and it is applied to bins the run barely visited. Push
far enough and a couple of stray counts in the sparse tail are amplified into the tallest
feature on the plot, with a deep clean valley in front of them; measured here, that alone
manufactured a barrier in six of twenty-five noise-only histograms. A limit on the shift in g
does not catch it, because whether a given shift is safe depends on how wide the histogram is.
The honest measure is the effective sample size: how many of the recorded sweeps still carry
weight after the tilt. If reweighting concentrates the answer onto a handful of them, the
answer is about those sweeps and not about the system, so it is refused.

Every guard here errs towards reporting no barrier. That is the safe direction: the claim under
test would be supported by finding a barrier, so the instrument must not be able to manufacture
one.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_SHIFT = 0.25          # refuse a reweight of more than this fraction in 1/g
MIN_LEVELS_APART = 4      # humps closer than this many energy levels are one hump (discreteness)
SMOOTH_PASSES = 2         # binomial passes before peak-finding; kills a one-bin alternation
NOISE_SIGMAS = 3.0        # a valley must be this many counting errors deep to count
MIN_ESS_FRAC = 0.02       # a reweight may not throw away more than this much of the sample
MIN_ESS = 500.0           # ...nor leave fewer than this many sweeps carrying the answer
COARSE_BINS = 12          # a real barrier must survive being looked at this coarsely
COARSE_TOL = 0.35         # ...with the humps landing within this fraction of the range


def reweight(levels, counts, g_from, g_to):
    w = np.log(np.maximum(counts, 1e-300)) - levels * (1.0 / g_to - 1.0 / g_from)
    w -= w.max()
    p = np.exp(w)
    return p / p.sum()


def visited_stretch(levels, counts):
    """The longest unbroken run of energies the run actually reached. See the header."""
    seen = np.asarray(counts) > 0
    best_i, best_n, i = 0, 0, 0
    while i < len(seen):
        if not seen[i]:
            i += 1
            continue
        j = i
        while j < len(seen) and seen[j]:
            j += 1
        if j - i > best_n:
            best_i, best_n = i, j - i
        i = j
    sl = slice(best_i, best_i + best_n)
    return levels[sl], np.asarray(counts, float)[sl]


def rebin(levels, counts, n_bins):
    """Pour the histogram into n_bins uniform bins across its range. Empty bins are dropped."""
    lo, hi = float(levels[0]), float(levels[-1])
    if hi <= lo:
        return levels, counts
    w = (hi - lo) / n_bins
    idx = np.minimum(((levels - lo) / w).astype(int), n_bins - 1)
    c = np.bincount(idx, weights=counts, minlength=n_bins)
    e = lo + (np.arange(n_bins) + 0.5) * w
    keep = c > 0
    return e[keep], c[keep]


def survives_coarsening(levels, counts, g_from, g_to, e_lo, e_hi):
    """Is the same pair of humps still there when the energy is resolved coarsely? See header."""
    cl, cc = rebin(levels, counts, COARSE_BINS)
    if len(cl) < 2 * MIN_LEVELS_APART:
        return True                      # too few bins to coarsen further; nothing to check
    hh = two_humps(reweight(cl, cc, g_from, g_to), cc, passes=0)
    if hh is None:
        return False
    a, b, _, _ = hh
    span = float(levels[-1] - levels[0]) or 1.0
    return (abs(cl[a] - e_lo) / span < COARSE_TOL) and (abs(cl[b] - e_hi) / span < COARSE_TOL)


def effective_sample_size(levels, counts, g_from, g_to):
    """Kish's effective sample size of the tilt: (sum w)^2 / sum w^2, counted over sweeps."""
    lw = -levels * (1.0 / g_to - 1.0 / g_from)
    lw -= lw.max()
    w = np.exp(lw)
    s2 = float((w * w * counts).sum())
    if s2 <= 0.0:
        return 0.0
    return float((w * counts).sum()) ** 2 / s2


def smooth(y, passes=SMOOTH_PASSES):
    """Binomial [1,2,1]/4, edge-padded. One pass exactly annihilates a one-bin alternation."""
    y = np.asarray(y, float)
    for _ in range(passes):
        y = np.convolve(np.r_[y[0], y, y[-1]], [0.25, 0.5, 0.25], mode="valid")
    return y


def two_humps(p, raw, passes=SMOOTH_PASSES):
    """Deepest genuine pair of humps, or None. Returns (low, high, valley, smoothed p).

    `passes` is 0 for the coarse-graining check, where rebinning has already done the smoothing
    and filtering again would wash out the very valley being looked for.
    """
    ps = smooth(p, passes)
    ps = ps / ps.sum()
    cs = smooth(raw, passes)              # counts behind each bin, smoothed the same way
    # An end bin higher than its one neighbour is a maximum too. This matters: at lambda = 0
    # the ground state is the highest reachable S, so the cold hump sits exactly on the edge of
    # the visited range, and an interior-only rule cannot see it (found on the N = 64 walk,
    # which had a valley 4 deep that the analysis reported as 0.04).
    maxima = [i for i in range(len(ps))
              if (i == 0 or ps[i] >= ps[i - 1]) and (i == len(ps) - 1 or ps[i] >= ps[i + 1])]
    best, best_depth = None, -np.inf
    for ai, a in enumerate(maxima):
        for b in maxima[ai + 1:]:
            if b - a < MIN_LEVELS_APART:
                continue
            v = a + int(np.argmin(ps[a:b + 1]))
            if v in (a, b):
                continue
            depth = np.log(min(ps[a], ps[b])) - np.log(max(ps[v], 1e-300))
            err = np.sqrt(sum(1.0 / max(cs[i], 1.0) for i in (a, b, v)))
            if depth <= NOISE_SIGMAS * err:
                continue
            if depth > best_depth:
                best, best_depth = (a, b, v, ps), depth
    return best


def observables(levels, counts, g0, n):
    """Scan for the coupling where the humps balance; return the pre-registered four."""
    levels, counts = visited_stretch(levels, counts)
    if len(levels) < 2 * MIN_LEVELS_APART:
        return None
    best = None
    floor = max(MIN_ESS, MIN_ESS_FRAC * float(np.sum(counts)))
    for g in np.linspace(g0 * (1 - MAX_SHIFT), g0 * (1 + MAX_SHIFT), 241):
        if effective_sample_size(levels, counts, g0, g) < floor:
            continue
        hh = two_humps(reweight(levels, counts, g0, g), counts)
        if hh is None:
            continue
        a, b, v, ps = hh
        if not survives_coarsening(levels, counts, g0, g, levels[a], levels[b]):
            continue
        below = ps[:v + 1].sum()
        imbalance = abs(np.log(max(below, 1e-300) / max(1 - below, 1e-300)))
        if best is None or imbalance < best[0]:
            best = (imbalance, g, a, b, v, ps)
    if best is None:
        return None
    imbalance, g, a, b, v, p = best
    return dict(g_c=g, shift=abs(g - g0) / g0, imbalance=imbalance,
                latent=float(levels[b] - levels[a]) / n,
                barrier=float(np.log(min(p[a], p[b])) - np.log(p[v])),
                levels_apart=int(b - a))


def main(name, out_dir="results"):
    rows = list(csv.DictReader(open(Path(out_dir) / (name + ".csv"), newline="")))
    store = Path(out_dir) / (name + "_hist")
    per_size = defaultdict(list)

    for r in rows:
        n, rep, k, g0 = int(r["N"]), int(r["replica"]), int(r["k"]), float(r["g"])
        f = store / ("N%d_rep%d_k%d.npz" % (n, rep, k))
        if not f.exists():
            continue
        d = np.load(f)
        got = observables(d["levels"].astype(float), d["counts"].astype(float), g0, n)
        if got:
            got.update(N=n, rep=rep, g0=g0, trips=int(r["round_trips"]),
                       binder=float(r["binder"]), c=float(r["c_per_point"]),
                       largest=float(r["largest_frac"]))
            per_size[n].append(got)

    print("T6 by tempering, %s. Criteria from PREREGISTRATION.md section T6.\n" % name)
    print("%-6s %-5s %-9s %-9s %-8s %-9s %-9s %-8s %-7s"
          % ("N", "reps", "g_c", "shift", "levels", "latent", "barrier", "Binder", "trips"))
    summary = []
    for n in sorted(per_size):
        got = per_size[n]
        # for each replica keep its best-balanced coupling, then average over replicas
        by_rep = defaultdict(list)
        for d in got:
            by_rep[d["rep"]].append(d)
        picks = [min(v, key=lambda d: d["imbalance"]) for v in by_rep.values()]
        lat = np.array([d["latent"] for d in picks])
        bar = np.array([d["barrier"] for d in picks])
        print("%-6d %-5d %-9.3f %-9.3f %-8d %-9.4f %-9.3f %-8.5f %-7d"
              % (n, len(picks), np.mean([d["g_c"] for d in picks]),
                 np.mean([d["shift"] for d in picks]),
                 int(np.mean([d["levels_apart"] for d in picks])),
                 lat.mean(), bar.mean(), np.mean([d["binder"] for d in picks]),
                 min(d["trips"] for d in picks)))
        summary.append(dict(N=n, L=np.sqrt(n), latent=lat.mean(), barrier=bar.mean(),
                            latent_err=lat.std(ddof=1) / np.sqrt(len(lat)) if len(lat) > 1 else np.nan,
                            barrier_err=bar.std(ddof=1) / np.sqrt(len(bar)) if len(bar) > 1 else np.nan,
                            binder=np.mean([d["binder"] for d in picks]),
                            trips=min(d["trips"] for d in picks)))

    if len(summary) >= 2:
        L = np.array([s["L"] for s in summary])
        bar = np.array([s["barrier"] for s in summary])
        lat = np.array([s["latent"] for s in summary])
        A = np.vstack([L, np.ones_like(L)]).T
        slope = np.linalg.lstsq(A, bar, rcond=None)[0][0]
        A2 = np.vstack([1.0 / L, np.ones_like(L)]).T
        lat0 = np.linalg.lstsq(A2, lat, rcond=None)[0][1]
        print("\n  barrier against L: slope %+.4f" % slope)
        print("  latent heat extrapolated to infinite size: %+.5f" % lat0)
        print("\n  (Fits shown for orientation only. The pre-registered verdict needs at least three"
              "\n   sizes passing the gates, and the gate on round trips is the one to watch here.)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "t6_lam0_control")
