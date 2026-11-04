"""T74: the owner's slab under the layer form, fed through surroundings and a curl edge with one maximum curling speed.
Usage:

    python scripts/run_layer_outside.py configs/t74_<cell>.json

Implements PREREGISTRATION.md section T74 as amended (Amendments 2 and 3, 2026-10-20 UTC; ASSUMPTIONS Q24). The
protocol is `graphity.layer_outside.run_layer_outside` with an `Outside`: the slab 4 x L x L with T71's model arm (layer
moves plus the clean two-switch move, the gate, the release into a daughter ledger); nothing feeds the slab's stores
directly. The feed goes into a reservoir of n_out stores; in the unfed phase the reservoir relaxes each sweep a fraction
rho_unfed of (E_out - n_out g_bg) toward a far-space ledger (while fed and while digesting at rho_fed, 0 under the
amendments: isolated); energy crosses the curl edge downhill in mean energy per store, by at most c a sweep, both ways;
then the moves. Fed for n_fed sweeps; then digestion, isolated, until the two means agree within digest_tol (1 %) or
n_digest_max sweeps pass; then the unfed phase for n_unfed sweeps.

One config is one cell (one lambda, one L, one feed, one cap, one background) with its own seed. Config keys: name,
section, written, lam, L, r, feed ("steady" or "meals"), every (sweeps between meals; 1 for steady), c, n_out,
rho_fed, g_bg_fed, rho_unfed, g_bg_unfed ("equal" for the common mean of reservoir and slab stores when the unfed
phase starts, read from the run), n_fed, n_digest_max, digest_tol, n_unfed, gate, e_out0 (default 0), move_set,
release_to, max_width, log_every, log_events, replicas, seed, and the labels feed_label, cap_label, background (copied
into every row). Each row also carries, for its replica: t_unfed (the last sweep of digestion), digest_balanced,
g_common (the common mean then), s_unfed_start (the slab's stored energy then) and common_above_2a (g_common > 2a, the
side of the gate Amendment 3 item 4 reads the equal background's prediction by). A meal is R * every with
R = r N_slab, so every feed has the same mean R a sweep. c, n_out, the rho_* and g_bg_* (other than "equal"), n_fed,
n_digest_max, digest_tol, n_unfed, gate and e_out0 may be
numbers or expressions in a = 4 (lambda - 1), L, N_slab = 4 L^2, r, R, and (after c is evaluated) c, and n_unfed may
also use n_fed. Every row carries the residual F + stores + daughter_energy + E_out + far - fed + leaked - F0 - e_out0,
which must be zero to rounding. Replica seeds: SeedSequence([seed, replica]).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.contraction import build                                     # noqa: E402
from graphity.layer_feed import MOVE_SETS, RELEASE_TO                      # noqa: E402
from graphity.layer_outside import EQUAL, Outside, run_layer_outside       # noqa: E402
from graphity.results import ResultWriter, provenance                      # noqa: E402


def evaluate(value, env):
    return float(eval(str(value), {"__builtins__": {}}, dict(env, max=max, min=min)))


def cell_numbers(cfg):
    """Every per-cell number of a config, evaluated in order."""
    lam, L, r = float(cfg["lam"]), int(cfg["L"]), float(cfg["r"])
    a = 4.0 * (lam - 1.0)
    n_slab = 4 * L * L
    env = dict(a=a, L=L, N_slab=n_slab, r=r, R=r * n_slab)
    env["c"] = evaluate(cfg["c"], env)
    for key in ("rho_fed", "rho_unfed", "n_out", "g_bg_fed", "gate", "e_out0"):
        env[key] = evaluate(cfg.get(key, 0.0), env)
    bg = cfg.get("g_bg_unfed", 0.0)
    env["g_bg_unfed"] = EQUAL if str(bg) == EQUAL else evaluate(bg, env)
    env["n_fed"] = int(round(evaluate(cfg["n_fed"], env)))
    env["n_unfed"] = int(round(evaluate(cfg["n_unfed"], env)))
    env["n_digest_max"] = int(round(evaluate(cfg.get("n_digest_max", 4096), env)))
    env["digest_tol"] = evaluate(cfg.get("digest_tol", 0.01), env)
    env["every"] = int(cfg.get("every", 1))
    env["meal"] = env["R"] * env["every"]
    return env


def seed_for(seed, rep):
    return int(np.random.SeedSequence([int(seed), int(rep)]).generate_state(1)[0])


COLS = ("sweep", "N", "dims", "F", "stores", "fed", "leaked", "n_split", "n_contract", "n_clean", "open", "curled",
        "broken", "daughters", "daughter_energy", "E_out", "far", "edge_in", "flow", "g_out_before", "g_in_before",
        "n_in_before", "g_bg", "phase")


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    move_set = cfg.get("move_set", "model")
    release_to = cfg.get("release_to", "daughter")
    assert move_set in MOVE_SETS and release_to in RELEASE_TO
    env = cell_numbers(cfg)
    lam, L = float(cfg["lam"]), int(cfg["L"])
    state = build((4, L, L))
    assert state.n == env["N_slab"]
    o = Outside(n_out=int(round(env["n_out"])), c=env["c"], n_fed=env["n_fed"], feed=cfg["feed"], amount=env["meal"],
                every=env["every"], e_out0=env["e_out0"], rho_fed=env["rho_fed"], g_bg_fed=env["g_bg_fed"],
                rho_unfed=env["rho_unfed"], g_bg_unfed=env["g_bg_unfed"], n_digest_max=env["n_digest_max"],
                digest_tol=env["digest_tol"], n_unfed=env["n_unfed"])
    two_a = 2.0 * env["a"]
    meta = dict(config=cfg, config_path=str(path), evaluated={k: v for k, v in env.items()}, package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section %s, written %s" % (cfg.get("section", "T74"),
                                                                              cfg.get("written", "")))
    fixed = dict(lam=lam, L=L, N_slab=state.n, r=env["r"], R=env["R"], feed=cfg["feed"],
                 feed_label=cfg.get("feed_label", cfg["feed"]), every=env["every"], meal=env["meal"], c=env["c"],
                 cap_label=cfg.get("cap_label", ""), background=cfg.get("background", ""), g_bg_fed=env["g_bg_fed"],
                 g_bg_unfed=env["g_bg_unfed"], rho_fed=env["rho_fed"], rho_unfed=env["rho_unfed"], n_out=o.n_out, n_fed=env["n_fed"],
                 n_digest_max=env["n_digest_max"], digest_tol=env["digest_tol"], n_unfed=env["n_unfed"], gate=env["gate"], e_out0=env["e_out0"], move_set=move_set,
                 release_to=release_to)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(int(cfg["replicas"])):
            seed = seed_for(cfg["seed"], rep)
            final, rows, first = run_layer_outside(
                state, lam, None, seed, outside=o, gate=env["gate"], move_set=move_set,
                max_width=int(cfg.get("max_width", 8)), log_every=int(cfg.get("log_every", 1)),
                release_to=release_to, log_events=bool(cfg.get("log_events", True)))
            per_rep = dict(t_unfed=first["t_unfed"], digest_balanced=int(first["digest_balanced"]),
                           g_common=first["g_common"], s_unfed_start=first["s_unfed_start"],
                           common_above_2a=int(first["g_common"] > two_a))
            for row in rows:
                d = dict(zip(COLS, row))
                d["dims"] = "x".join(str(x) for x in d["dims"])
                d["residual"] = (d["F"] + d["stores"] + d["daughter_energy"] + d["E_out"] + d["far"] - d["fed"]
                                 + d["leaked"] - first["f0"] - first["e_out0"])
                out.write(dict(fixed, replica=rep, seed=seed, **per_rep, **d,
                               first_split=first["first_split"] if first["first_split"] is not None else "",
                               first_clean=first["first_clean"] if first["first_clean"] is not None else "",
                               first_contract=first["first_contract"] if first["first_contract"] is not None else ""))
            last = rows[-1]
            print("lam=%.2f L=%d feed=%s c=%.3g bg=%s rep=%-2d | daughters %d (%.1f) | digestion to %d (%s), common "
                  "mean %.3f vs 2a %.3f | stores %.1f E_out %.1f far %.1f edge_in %.1f fed %.1f | max |flow| %.3g"
                  % (lam, L, fixed["feed_label"], env["c"], fixed["background"], rep, last[13], last[14],
                     first["t_unfed"], "balanced" if first["digest_balanced"] else "limit", first["g_common"], two_a,
                     last[4], last[15], last[16], last[17], last[5], first["max_abs_flow"]), flush=True)
    print("wrote", Path(out_dir) / (cfg["name"] + ".csv"))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
