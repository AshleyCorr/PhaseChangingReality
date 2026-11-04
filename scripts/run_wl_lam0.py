"""T6 at lambda = 0 by a one-dimensional Wang-Landau walk in S. Usage:

    python scripts/run_wl_lam0.py 36 64          (sizes, square tori unless side x side given as 16x6)

WHY THIS INSTRUMENT HERE. Tempering across the lambda = 0 transition hits a wall that densifying
the ladder does not remove (ASSUMPTIONS O7, O9): swap acceptance collapses to about 0.26 at the
transition rung however the rungs are placed, because across a first-order transition adjacent
rungs stop sharing energies, and the gap grows with N. That is the textbook case for a
flat-histogram method, which walks straight through the valley instead of trying to hop over it.
The two-dimensional walk over (S, X) failed at N = 36 (Q17) for a reason that does not apply
here: at lambda = 0 the energy is 16(N - S) alone, X never enters, and the density of states is
one-dimensional in S -- tens of bins rather than hundreds.

TWO PASSES. Round trips are counted between the ends of the window, so a window that includes
an unreachable S would report zero trips for a walk that covered everything reachable. A short
first pass discovers the reachable range; the real run uses exactly that.

WHAT COMES OUT. ln g(S) up to a constant, from which P(S) at any g is g(S) exp(-16(N - S)/g).
The pre-registered observables follow along phi = S/N as in analyse_t6_phi.py: the coupling
where the two humps balance, the depth of the valley, and the latent heat as the energy
difference between the phases (which at lambda = 0 is just 16 times the difference in mean S).
Round trips are reported with every row, as the pre-registration requires.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import run_chain, torus, NO_CAP          # noqa: E402
from graphity.wang_landau import dos_graph                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t6_hist as H                                 # noqa: E402  (shared guards)

E_SQ = 16.0


def parse_size(tok):
    if "x" in tok:
        a, b = tok.split("x")
        return int(a), int(b)
    s = int(round(float(tok) ** 0.5))
    assert s * s == int(tok), tok
    return s, s


def profile(lng, seen, s_min, n, g):
    s = np.arange(len(lng)) + s_min
    lp = np.where(seen, lng - E_SQ * (n - s) / g, -np.inf)
    lp -= lp[seen].max()
    p = np.exp(lp)
    return s, p / p.sum()


def observables(lng, seen, s_min, n, g_lo=3.0, g_hi=14.0):
    best = None
    for g in np.linspace(g_lo, g_hi, 1101):
        s, p = profile(lng, seen, s_min, n, g)
        keep = seen
        ps = p[keep]
        hh = H.two_humps(ps, np.full(ps.shape, 1e9))      # exact profile: no counting noise
        if hh is None:
            continue
        a, b, v, sm = hh
        below = sm[:v + 1].sum()
        imb = abs(np.log(max(below, 1e-300) / max(1 - below, 1e-300)))
        if best is None or imb < best[0]:
            best = (imb, g, a, b, v, sm, s[keep])
    if best is None:
        return None
    imb, g, a, b, v, sm, sk = best
    s_lo = (sk[:v + 1] * sm[:v + 1]).sum() / sm[:v + 1].sum()
    s_hi = (sk[v + 1:] * sm[v + 1:]).sum() / sm[v + 1:].sum()
    return dict(g_c=float(g), imbalance=float(imb), phi_lo=float(sk[a]) / n, phi_hi=float(sk[b]) / n,
                latent=float(E_SQ * (s_hi - s_lo)) / n,
                barrier=float(np.log(min(sm[a], sm[b])) - np.log(sm[v])))


def main(sizes, seed=20260921, out_dir="results", moves_mult=1.0, tag="", refine=None):
    """moves_mult scales the cap on stage one, which rarely binds: stage one ends when ln f has
    converged. Round trips come mostly from stage one and then accrue with the length of stage
    two, so `refine` (sweeps per pass in stage two) is the knob that buys them. Unit walks gave
    52 (N = 64), 16 (96), 3 (128); doubling moves_mult alone left N = 96 at 15 to 17."""
    out = Path(out_dir)
    for tok in sizes:
        lx, ly = parse_size(tok)
        n = lx * ly
        adj, part = torus(lx, ly)
        side_u = np.flatnonzero(part == 0)
        # melt first so the walk starts from an ordinary state rather than the perfect sheet
        run_chain(adj, side_u, 0.0, 400, 1, seed + n, 0.0, NO_CAP, False)

        # pass one: find the reachable range of S
        wide = (max(0, int(0.05 * n)), int(1.6 * n))
        scout = dos_graph(adj, side_u, wide, None, cap=NO_CAP, seed=seed + n + 1,
                          ln_f_final=0.05, flat=0.6, sweeps_per_check=200,
                          max_moves=40_000_000 * (n // 36), refine_sweeps=0)
        seen_s = np.flatnonzero(scout["seen"][:, 0]) + scout["s_min"]
        s_range = (int(seen_s.min()), int(seen_s.max()))
        print("N=%d  reachable S %d..%d (phi %.3f..%.3f) after the scout pass"
              % (n, s_range[0], s_range[1], s_range[0] / n, s_range[1] / n), flush=True)

        # pass two: the real walk on exactly that range
        res = dos_graph(adj, side_u, s_range, None, cap=NO_CAP, seed=seed + n + 2,
                        ln_f_final=1e-6, flat=0.8, sweeps_per_check=500,
                        max_moves=int(400_000_000 * max(1, n // 36) * moves_mult),
                        refine_sweeps=int(refine if refine else 200_000 * moves_mult))
        lng, seen = res["lng"][:, 0].copy(), res["seen"][:, 0].copy()
        got = observables(lng, seen, res["s_min"], n)

        np.savez_compressed(out / ("wl_lam0_N%d%s.npz" % (n, tag)), lng=lng, seen=seen, s_min=res["s_min"],
                            N=n, lx=lx, ly=ly, round_trips=res["round_trips"],
                            round_trips_stage_one=res["round_trips_stage_one"],
                            ln_f_final=res.get("ln_f", np.nan), flatness=res.get("flatness", np.nan))
        row = dict(N=n, lx=lx, ly=ly, seed=seed, tag=tag, moves_mult=moves_mult,
                   refine_sweeps=int(refine if refine else 200_000 * moves_mult),
                   s_lo=s_range[0], s_hi=s_range[1],
                   bins=int(seen.sum()), round_trips=int(res["round_trips"]),
                   round_trips_stage_one=int(res["round_trips_stage_one"]),
                   flatness=float(res.get("flatness", np.nan)),
                   **({} if got is None else got))
        with (out / "wl_lam0.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if got:
            print("N=%d  trips=%d  g_c=%.3f  phi %.3f|%.3f  latent=%.4f  barrier=%.3f"
                  % (n, res["round_trips"], got["g_c"], got["phi_lo"], got["phi_hi"],
                     got["latent"], got["barrier"]), flush=True)
        else:
            print("N=%d  trips=%d  no two-hump structure at any g in [3, 14]"
                  % (n, res["round_trips"]), flush=True)


if __name__ == "__main__":
    # usage: run_wl_lam0.py SIZE [SIZE ...] [--seed N] [--mult X] [--tag T] [--refine SWEEPS]
    args, seed, mult, tag, refine = [], 20260921, 1.0, "", None
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--seed":
            seed = int(next(it))
        elif a == "--mult":
            mult = float(next(it))
        elif a == "--tag":
            tag = next(it)
        elif a == "--refine":
            refine = int(next(it))
        else:
            args.append(a)
    main(args or ["36"], seed=seed, moves_mult=mult, tag=tag, refine=refine)
