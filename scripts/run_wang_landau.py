"""The density of states, and the four observables of the T6 pre-registration. Usage:

    python scripts/run_wang_landau.py configs/t6_pilot_n36.json

Writes results/<name>.csv (one row per size, seed and lambda; ONE sampler run per
size and seed, because the bins are (S, X) and therefore serve every lambda at once) and .meta.json, and refuses to
overwrite either (rule 5). Every choice below is fixed in PREREGISTRATION.md, section T6; this
script implements it and adds nothing.

WHAT IS MEASURED, and why four things rather than one. All four come free from the same density
of states, and the literature treats them as the standard set for telling a weak first-order
transition from a continuous one. Measuring one and not the others is how a weak one is missed.

    latent_heat   the gap between the two humps of P(H), per point.  First order: stays finite.
    barrier       the depth of the valley between them, in ln P.     First order: grows like L.
    binder_min    the minimum of 1 - <H^4>/3<H^2>^2 over coupling.   First order: below 2/3.

A note on that third one, because it is easy to get silently wrong. The fourth-order energy
cumulant uses RAW moments, so its value depends on where the zero of energy sits. The
convention here is the model's own: H = 16(N - S) + 4 lambda X, which is zero for a perfect
flat sheet. Any comparison with a published number has to use the same convention or the
numbers are not comparable.
    c_max         the height of the heat-capacity peak, per point.   First order: grows like N.

The latent heat is the one claim 4 actually needs: the claim is that a lump of energy comes out,
and the lump IS the latent heat. A barrier with no latent heat satisfies S2's letter and does
nothing claim 4 wants.

THE TRANSITION COUPLING IS FOUND SEPARATELY AT EACH SIZE, as the coupling where the two humps
carry equal weight. That is deliberate: [Kelly22] reports the critical temperature "appears to
be asymptotically nonfinite" and our own T9 measures it drifting with ln N, so a criterion that
assumed one fixed coupling across sizes would be wrong if either holds.

Config keys: name, sides (each L for an L x L torus), lambdas, seeds, and the sampler settings
ln_f_final, sweeps_per_check, max_moves, refine_sweeps, passes. Optional: s_range and x_range to
restrict the window (pairs of inclusive bounds); the default is the whole reachable range.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

from graphity import __version__
from graphity.connectivity import connectivity
from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, is_valid, torus
from graphity.results import ResultWriter
from graphity.small_graphs import sides_first
from graphity.wang_landau import dos_graph


def humps(lng, seen, s_min, x_min, n, g, lam):
    """P(H) at this coupling, collapsed onto distinct energies. Returns (energies, probabilities)."""
    ss, xs = np.nonzero(seen)
    h = ENERGY_PER_SQUARE * (n - (ss + s_min)) + ENERGY_PER_SURPLUS * lam * (xs + x_min)
    w = lng[seen] - h / g
    w -= w.max()
    p = np.exp(w)
    levels = {}
    for e, q in zip(np.round(h, 6), p):
        levels[e] = levels.get(e, 0.0) + q
    out = np.array(sorted(levels.items()))
    out[:, 1] /= out[:, 1].sum()
    return out[:, 0], out[:, 1]


def split_weight(energies, probs, cut):
    """Weight below and above an energy cut."""
    lo = probs[energies < cut].sum()
    return lo, 1.0 - lo


def two_humps(energies, probs):
    """(low peak, high peak, valley floor) as indices, or None if the distribution has one hump.

    CHOOSING THE PAIR. Taking the two tallest local maxima is wrong: a flat top or a little
    noise puts two "maxima" a few bins apart on the SAME hump, and the valley between them is
    meaningless. Instead every pair of maxima is scored by the barrier it implies,

        ln min(p_a, p_b) - ln p_valley

    and the deepest is taken. That is unbiased in the direction that matters -- it finds the
    largest barrier present and returns something near zero when there is none, rather than
    imposing a threshold that would hide a weak one, which is exactly the case T6 is about.
    """
    interior = np.arange(1, len(probs) - 1)
    maxima = [i for i in interior if probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]]
    if len(maxima) < 2:
        return None
    best, best_depth = None, -np.inf
    for ai, a in enumerate(maxima):
        for b in maxima[ai + 1:]:
            v = a + int(np.argmin(probs[a:b + 1]))
            if v == a or v == b:
                continue                          # no genuine dip between them
            depth = np.log(min(probs[a], probs[b])) - np.log(max(probs[v], 1e-300))
            if depth > best_depth:
                best, best_depth = (a, b, v), depth
    return best


def at_coupling(lng, seen, s_min, x_min, n, g, lam):
    """Everything the pre-registration asks for at one coupling."""
    ss, xs = np.nonzero(seen)
    h = ENERGY_PER_SQUARE * (n - (ss + s_min)) + ENERGY_PER_SURPLUS * lam * (xs + x_min)
    w = lng[seen] - h / g
    w -= w.max()
    p = np.exp(w)
    p /= p.sum()
    m1 = float((p * h).sum())
    m2 = float((p * h * h).sum())
    m4 = float((p * h ** 4).sum())
    c = (m2 - m1 * m1) / (g * g * n)                      # heat capacity per point
    binder = 1.0 - m4 / (3.0 * m2 * m2) if m2 > 0 else np.nan
    return m1 / n, c, binder


def transition_coupling(lng, seen, s_min, x_min, n, lam, lo=0.5, hi=40.0):
    """The coupling where the two humps carry equal weight, by bisection on their weight ratio.

    Returns (g, cut energy) or (nan, nan) if no two humps appear anywhere in the range.
    """
    def imbalance(g):
        e, p = humps(lng, seen, s_min, x_min, n, g, lam)
        hh = two_humps(e, p)
        if hh is None:
            return None, None
        a, b, v = hh
        below, above = split_weight(e, p, e[v])
        return np.log(max(below, 1e-300) / max(above, 1e-300)), e[v]

    grid = np.geomspace(lo, hi, 60)
    vals = [(g,) + imbalance(g) for g in grid]
    ok = [(g, s, c) for g, s, c in vals if s is not None]
    if len(ok) < 2:
        return float("nan"), float("nan")
    for (g1, s1, _), (g2, s2, c2) in zip(ok, ok[1:]):
        if s1 == 0 or s1 * s2 < 0:
            for _ in range(60):                            # bisect
                gm = np.sqrt(g1 * g2)
                sm, cm = imbalance(gm)
                if sm is None:
                    break
                if s1 * sm <= 0:
                    g2, s2, c2 = gm, sm, cm
                else:
                    g1, s1 = gm, sm
            return float(np.sqrt(g1 * g2)), float(c2)
    return float("nan"), float("nan")


def measure(lng, seen, s_min, x_min, n, lam):
    """The four observables of the pre-registration, plus where they were measured."""
    g_c, cut = transition_coupling(lng, seen, s_min, x_min, n, lam)
    row = dict(g_c=g_c, latent_heat=float("nan"), barrier=float("nan"),
               binder_min=float("nan"), g_binder=float("nan"), c_max=float("nan"), g_c_max=float("nan"))
    if np.isfinite(g_c):
        e, p = humps(lng, seen, s_min, x_min, n, g_c, lam)
        hh = two_humps(e, p)
        if hh is not None:
            a, b, v = hh
            row["latent_heat"] = float(e[b] - e[a]) / n
            row["barrier"] = float(np.log(max(p[a], p[b])) - np.log(p[v]))
    grid = np.geomspace(0.5, 40.0, 400)
    stats = [at_coupling(lng, seen, s_min, x_min, n, g, lam) for g in grid]
    binders = np.array([s[2] for s in stats])
    caps = np.array([s[1] for s in stats])
    if np.isfinite(binders).any():
        i = int(np.nanargmin(binders))
        row["binder_min"], row["g_binder"] = float(binders[i]), float(grid[i])
    if np.isfinite(caps).any():
        j = int(np.nanargmax(caps))
        row["c_max"], row["g_c_max"] = float(caps[j]), float(grid[j])
    return row


def main(path):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), graphity_version=__version__,
                numpy=np.__version__, numba=numba.__version__, python=platform.python_version(),
                platform=platform.platform(),
                preregistration="PREREGISTRATION.md section T6, revised 2026-09-21")
    store = Path("results") / (cfg["name"] + "_dos")
    store.mkdir(parents=True, exist_ok=True)
    with ResultWriter(cfg["name"], meta) as out:
        for side in cfg["sides"]:
            adj0, part = torus(side, side, cap=NO_CAP)
            adj0 = sides_first(adj0, part)
            n = len(adj0)
            side_u = np.arange(n // 2)
            assert is_valid(adj0, NO_CAP)
            s_range = tuple(cfg.get("s_range") or (0, (3 * n) // 2))
            x_range = tuple(cfg.get("x_range") or (0, 2 * n))
            # ONE density of states per (size, seed) serves EVERY lambda: the bins are (S, X),
            # and the walk that fills them does not know what lambda is. Re-running the sampler
            # per lambda would be the same work four times over.
            for seed in cfg["seeds"]:
                adj = adj0.copy()
                res = dos_graph(adj, side_u, s_range, x_range, cap=NO_CAP, seed=seed,
                                ln_f_final=cfg["ln_f_final"],
                                sweeps_per_check=cfg["sweeps_per_check"],
                                max_moves=cfg["max_moves"],
                                refine_sweeps=cfg["refine_sweeps"],
                                passes=cfg.get("passes", 2))
                # the density of states is expensive to make and cheap to keep: store it so the
                # observables can be re-derived later without paying for the sampling again
                np.savez_compressed(store / ("N%d_seed%d.npz" % (n, seed)),
                                    lng=res["lng"], seen=res["seen"],
                                    s_min=res["s_min"], x_min=res["x_min"], N=n)
                pieces, largest, babies, cubes = connectivity(adj)
                common = dict(N=n, side=side, seed=seed,
                              bins=int(res["seen"].sum()), moves=res["moves"],
                              ln_f=res["ln_f"], accept=round(res["accept"], 4),
                              discovery_round=res["discovery_round"],
                              round_trips=res.get("round_trips", -1),
                              flatness=round(res.get("flatness", float("nan")), 4),
                              end_pieces=int(pieces), end_largest=largest / n,
                              end_baby=babies / n, end_cubes=cubes / n)
                print("N=%-4d seed=%-4d bins=%-6d moves=%.2e trips=%-6d flat=%.3f"
                      % (n, seed, common["bins"], common["moves"],
                         common["round_trips"], common["flatness"]), flush=True)
                for lam in cfg["lambdas"]:
                    row = dict(common, lam=lam)
                    row.update(measure(res["lng"], res["seen"], res["s_min"], res["x_min"], n, lam))
                    out.write(row)
                    print("   lam=%-5.2f g_c=%-9.3f latent=%-10.5f barrier=%-9.3f "
                          "binder=%-9.5f c_max=%-9.4f"
                          % (lam, row["g_c"], row["latent_heat"], row["barrier"],
                             row["binder_min"], row["c_max"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
