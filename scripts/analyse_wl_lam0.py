"""The T6 verdict table at lambda = 0 from the one-dimensional walks. Usage:

    python scripts/analyse_wl_lam0.py

Reads every results/wl_lam0_N*<tag>.npz, re-derives the four observables from each walk's
ln g(S) with the current analysis (so a repaired analysis re-reads old walks rather than needing
them re-run), applies gates 3 and 4 of PREREGISTRATION.md T6, and fits what passes. Under
amendment 2 only multiples of 16 enter the lambda = 0 fit; other sizes are shown and marked.

Gate 3: at each size the seeds must agree on the barrier to 20 % and on the latent heat to 10 %,
measured as (max - min) / mean. Gate 4: every walk must have made at least 20 round trips.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wl_lam0 as W                          # noqa: E402

GATE3_BARRIER, GATE3_LATENT, GATE4_TRIPS = 0.20, 0.10, 20
BINDER_TOL = 0.05          # amendment 4: tolerance on the two-spike prediction at the largest size


def main(out_dir="results"):
    runs = defaultdict(list)
    for f in sorted(Path(out_dir).glob("wl_lam0_N*.npz")):
        m = re.match(r"wl_lam0_N(\d+)(.*)\.npz", f.name)
        n, tag = int(m.group(1)), m.group(2) or "(untagged)"
        d = np.load(f)
        got = W.observables(d["lng"], d["seen"], int(d["s_min"]), n)
        runs[n].append(dict(tag=tag, trips=int(d["round_trips"]), got=got))

    print("T6 at lambda = 0 by one-dimensional Wang-Landau in S. PREREGISTRATION.md T6, amendments 1-3.\n")
    print("%-5s %-6s %-11s %-6s %-8s %-15s %-9s %-9s  %s"
          % ("N", "knots", "tag", "trips", "g_c", "phi hot|cold", "latent", "barrier", "gate 4"))
    passing = []
    for n in sorted(runs):
        clean = (n % 16 == 0)
        rows = runs[n]
        for r in rows:
            g = r["got"]
            ok4 = r["trips"] >= GATE4_TRIPS
            if g:
                print("%-5d %-6s %-11s %-6d %-8.3f %-7.3f|%-7.3f %-9.3f %-9.3f  %s"
                      % (n, ("%d" % (n // 16)) if clean else "ribbon", r["tag"], r["trips"], g["g_c"],
                         g["phi_lo"], g["phi_hi"], g["latent"], g["barrier"], "pass" if ok4 else "FAIL"))
            else:
                print("%-5d %-6s %-11s %-6d %-8s %-15s %-9s %-9s  %s"
                      % (n, ("%d" % (n // 16)) if clean else "ribbon", r["tag"], r["trips"], "-", "no two humps",
                         "-", "-", "pass" if ok4 else "FAIL"))
        good = [r for r in rows if r["got"] and r["trips"] >= GATE4_TRIPS]
        if len(good) < 2:
            print("      -> %d seed(s) pass gate 4 with a fit; gate 3 needs at least two. Not fitted.\n" % len(good))
            continue
        bar = np.array([r["got"]["barrier"] for r in good])
        lat = np.array([r["got"]["latent"] for r in good])
        gc = np.array([r["got"]["g_c"] for r in good])
        sb = (bar.max() - bar.min()) / bar.mean()
        sl = (lat.max() - lat.min()) / lat.mean()
        ok3 = sb <= GATE3_BARRIER and sl <= GATE3_LATENT
        print("      -> %d seeds: barrier %.3f +/- %.3f (spread %.1f%%), latent %.3f +/- %.3f (spread %.1f%%), "
              "g_c %.3f +/- %.3f   gate 3 %s%s\n"
              % (len(good), bar.mean(), bar.std(ddof=1), 100 * sb, lat.mean(), lat.std(ddof=1), 100 * sl,
                 gc.mean(), gc.std(ddof=1), "pass" if ok3 else "FAIL",
                 "" if clean else "   [not a multiple of 16: shown, not fitted]"))
        if ok3 and clean:
            passing.append(dict(N=n, L=np.sqrt(n), barrier=bar.mean(), b_err=bar.std(ddof=1) / np.sqrt(len(bar)),
                                latent=lat.mean(), l_err=lat.std(ddof=1) / np.sqrt(len(lat)), g_c=gc.mean()))

    print("Sizes passing both gates and entering the fit: %s" % [p["N"] for p in passing])
    if len(passing) < 3:
        print("The pre-registered verdict needs at least three. No verdict from this table.")
        return

    # --- the other two observables, from the same density of states ---------------------
    # Binder energy cumulant B(g) = 1 - <H^4> / 3<H^2>^2 with H = 16(N - S), and the specific
    # heat C(g)/N = var(H) / (g^2 N); both scanned over g and the extremum kept. Per seed, then
    # averaged, so the seed scatter is the error bar here too.
    for p in passing:
        bmins, cmaxs = [], []
        for r in runs[p["N"]]:
            if not (r["got"] and r["trips"] >= GATE4_TRIPS):
                continue
            d = np.load(next(Path(out_dir).glob("wl_lam0_N%d%s.npz"
                                                % (p["N"], "" if r["tag"] == "(untagged)" else r["tag"]))))
            lng, seen, s0 = d["lng"], d["seen"], int(d["s_min"])
            bmin, cmax = np.inf, -np.inf
            for g in np.linspace(3.0, 14.0, 551):
                s, pr = W.profile(lng, seen, s0, p["N"], g)
                h = W.E_SQ * (p["N"] - s)
                m2, m4 = (pr * h ** 2).sum(), (pr * h ** 4).sum()
                m1 = (pr * h).sum()
                bmin = min(bmin, 1.0 - m4 / (3.0 * m2 * m2))
                cmax = max(cmax, (m2 - m1 * m1) / (g * g * p["N"]))
            bmins.append(bmin); cmaxs.append(cmax)
        p["binder"], p["bind_err"] = np.mean(bmins), np.std(bmins, ddof=1) / np.sqrt(len(bmins))
        p["cmax"], p["c_err"] = np.mean(cmaxs), np.std(cmaxs, ddof=1) / np.sqrt(len(cmaxs))

    print("\n%-5s %-16s %-16s %-16s %-14s" % ("N", "latent", "barrier", "Binder min", "C_max / N"))
    for p in passing:
        print("%-5d %6.3f +/- %5.3f %6.3f +/- %5.3f %7.4f +/- %6.4f %6.3f +/- %5.3f"
              % (p["N"], p["latent"], p["l_err"], p["barrier"], p["b_err"], p["binder"], p["bind_err"],
                 p["cmax"], p["c_err"]))

    # --- weighted straight-line fits, with the slope's and intercept's standard errors ------
    def wfit(x, y, sig):
        w = 1.0 / np.maximum(sig, 1e-9) ** 2
        A = np.vstack([x, np.ones_like(x)]).T
        cov = np.linalg.inv(A.T @ (w[:, None] * A))
        beta = cov @ A.T @ (w * y)
        return beta, np.sqrt(np.diag(cov))

    L = np.array([p["L"] for p in passing]); n = np.array([p["N"] for p in passing], float)
    b = np.array([p["barrier"] for p in passing]); be = np.array([p["b_err"] for p in passing])
    lat = np.array([p["latent"] for p in passing]); le = np.array([p["l_err"] for p in passing])
    bind = np.array([p["binder"] for p in passing])

    (slopeL, icL), (eL, _) = wfit(L, b, be)
    (slopeN, icN), (eN, _) = wfit(n, b, be)
    (_, lat0), (_, lat0e) = wfit(1.0 / L, lat, le)
    (bslope, _), (bslope_e, _) = wfit(1.0 / L, bind, np.array([p["bind_err"] for p in passing]))

    print("\n  barrier against L:  slope %+.3f +/- %.3f   (%.1f standard errors from zero)"
          % (slopeL, eL, slopeL / eL))
    print("  barrier against N:  slope %+.4f +/- %.4f  (%.1f standard errors from zero)"
          % (slopeN, eN, slopeN / eN))
    print("  latent heat at infinite size (intercept against 1/L): %.2f +/- %.2f per point"
          "  (%.1f standard errors from zero)" % (lat0, lat0e, lat0 / lat0e))
    print("  Binder minimum: %s ; trend with size (slope against 1/L): %+.4f +/- %.4f"
          % (", ".join("%.4f" % v for v in bind), bslope, bslope_e))
    # What a first-order transition predicts the Binder minimum tends to. For a distribution
    # that is two spikes at the phase energies e+ and e- (per point; the factor N cancels),
    # B = 1 - 2(e+^4 + e-^4) / (3 (e+^2 + e-^2)^2), which is 2/3 only when |e+| = |e-| and is
    # otherwise BELOW 2/3 by an amount fixed by the two energies. Here e+ = 16(1 - phi_hot)
    # and e- = 16(1 - phi_cold) = -8. So the limit is size-independent and computable, and the
    # question is whether the measured minima are heading for it or for 2/3.
    two_phase = []
    for p in passing:
        r0 = [r for r in runs[p["N"]] if r["got"] and r["trips"] >= GATE4_TRIPS]
        ep = np.mean([W.E_SQ * (1 - r["got"]["phi_lo"]) for r in r0])
        em = np.mean([W.E_SQ * (1 - r["got"]["phi_hi"]) for r in r0])
        a, c = ep * ep, em * em
        two_phase.append(1.0 - 2.0 * (a * a + c * c) / (3.0 * (a + c) ** 2))
    print("  Two-spike prediction for the Binder minimum at infinite size, from the measured phase")
    print("  energies at each size: %s   (2/3 = 0.6667 is what a continuous transition tends to)"
          % ", ".join("%.4f" % v for v in two_phase))

    # --- the three criteria, as pre-registered under amendments 1-4 ----------------------
    c1 = lat0 / lat0e >= 3.0
    c2 = slopeL / eL >= 3.0
    below = np.all(bind < 2.0 / 3.0)
    # Criterion 3 as originally written ("not rising towards 2/3"), kept for the record:
    rising = (bslope < 0) and (abs(bslope) > bslope_e)
    c3_original = below and not rising
    # Criterion 3 under amendment 4 (enacted 2026-09-22): below 2/3 at every size; at the largest
    # size within BINDER_TOL of the two-spike prediction from that size's measured phase energies;
    # and the gap (prediction - measured) shrinking with N.
    gaps = np.array(two_phase) - bind
    c3 = bool(below and abs(gaps[-1]) <= BINDER_TOL and np.all(np.diff(gaps) < 0))
    print("\n  Criterion 1, latent heat non-zero at 3 s.e.:      %s" % ("MET" if c1 else "not met"))
    print("  Criterion 2, barrier slope positive at 3 s.e.:    %s" % ("MET" if c2 else "not met"))
    print("  Criterion 3 (amendment 4), Binder min below 2/3, within %.2f of the two-spike value at the"
          "\n     largest size, gap shrinking with N:            %s   [gaps %s; as originally written: %s]"
          % (BINDER_TOL, "MET" if c3 else "not met", ", ".join("%.3f" % v for v in gaps),
             "met" if c3_original else "not met"))
    if c1 and c2 and c3:
        verdict = "FIRST ORDER"
    elif (lat0 / lat0e < 2.0) and (slopeL / eL < 2.0) and not below:
        verdict = "NO EVIDENCE OF FIRST ORDER AT THESE SIZES"
    else:
        verdict = "INCONCLUSIVE"
    print("\n  PRE-REGISTERED VERDICT, lambda = 0, sizes %s:  %s" % ([p["N"] for p in passing], verdict))
    print("\n  Recorded alongside, not part of the verdict: the barrier grows with L at %.1f s.e. and"
          "\n  with N at %.1f s.e.; whichever fits better says whether there is an interface (L) or"
          "\n  none (N: knots cost nothing to separate)." % (slopeL / eL, slopeN / eN))


if __name__ == "__main__":
    main()
