"""T71 (draft): the owner's slab under the layer form, fed at every point. Usage:

    python scripts/run_layer_feed.py configs/t71_<cell>.json

Implements PREREGISTRATION.md section T71 (drafted 2026-10-18; NOT to be launched before the owner's gate decision
and her prediction are recorded there). The protocol is `graphity.layer_feed.run_layer_feed`: the slab 4 x L x L,
untied, with one store per point; each sweep the feed adds `rate` to every store (stopping at the first split if
`feed_stops_at_split`), a fraction `leak` of every store leaves, N clean two-switch attempts are made (`move_set`
"model"; none for "layer"; the clean check off for "breaking"), then one layer proposal (split or contract) paid from
the pooled stores of the layers it touches, with the `gate` applied to any move that costs nothing. A release goes to
the stores beside the cut (`release_to` "stores") or leaves one way into a daughter ledger ("daughter", the owner's
decision of 18 October, VISION Update 66), in which case the slab reforms when the stores can pay for it; or to two twins
at once ("twins", her decision of 25 October, VISION Update 120; PREREGISTRATION T82), the second paid from the pool. The
single-switch control (arm C of T71) is T68's runner, `scripts/run_slab_feed.py`, not this one.

Config keys: name, section, written, lambdas (list), sizes (list of L), rates (list, per point per sweep), gate,
leak, n_sweeps, replicas, seed, move_set, release_to, feed_stops_at_split, max_width, log_every, log_events (default
true: every sweep in which a move fires or a daughter is born is logged besides every log_every-th). `gate`, `leak` and `n_sweeps`
may be numbers or expressions in a = 4 (lambda - 1), L, N_slab = 4 L^2 and r (the rate), evaluated per cell, so that a
registration can fix them by rule rather than by hand (for example gate "a * N_slab", n_sweeps "max(2000, 40 * a / r)").
Energy is conserved exactly; every row carries the residual F + stores + daughter_energy - fed + leaked - F0, which must
be zero to rounding. Seeds: SeedSequence([seed, N_slab, lambda index, rate index, replica]).
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
from graphity.layer_feed import FEED_RELEASES, MOVE_SETS, run_layer_feed   # noqa: E402
from graphity.results import ResultWriter, provenance                      # noqa: E402


def per_cell(value, lam, L, rate):
    """A number, or an expression in a, L, N_slab and r."""
    a = 4.0 * (lam - 1.0)
    env = dict(a=a, L=L, N_slab=4 * L * L, r=rate, max=max, min=min)
    return float(eval(str(value), {"__builtins__": {}}, env))


def seed_for(seed, n_slab, i_lam, i_rate, rep):
    return int(np.random.SeedSequence([int(seed), int(n_slab), int(i_lam), int(i_rate), int(rep)]).generate_state(1)[0])


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    move_set = cfg.get("move_set", "model")
    release_to = cfg.get("release_to", "stores")
    assert move_set in MOVE_SETS and release_to in FEED_RELEASES
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section %s, written %s" % (cfg.get("section", "T71"), cfg.get("written", "2026-10-18")))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for i_lam, lam in enumerate(cfg["lambdas"]):
            lam = float(lam)
            for L in cfg["sizes"]:
                L = int(L)
                state = build((4, L, L))
                for i_rate, rate in enumerate(cfg["rates"]):
                    rate = float(rate)
                    gate = per_cell(cfg.get("gate", 0.0), lam, L, rate)
                    leak = per_cell(cfg.get("leak", 0.0), lam, L, rate)
                    n_sweeps = int(round(per_cell(cfg["n_sweeps"], lam, L, rate)))
                    for rep in range(int(cfg["replicas"])):
                        seed = seed_for(cfg["seed"], state.n, i_lam, i_rate, rep)
                        final, rows, first = run_layer_feed(
                            state, lam, rate, n_sweeps, seed, gate=gate, leak=leak,
                            feed_stops_at_split=bool(cfg.get("feed_stops_at_split", False)), move_set=move_set,
                            max_width=int(cfg.get("max_width", 8)), log_every=int(cfg.get("log_every", 1)),
                            release_to=release_to, log_events=bool(cfg.get("log_events", True)))
                        for sweep, n, dims, f, stores, fed, leaked, n_split, n_contract, n_clean, op, cu, br, dau, dau_e in rows:
                            out.write(dict(lam=lam, L=L, N_slab=state.n, rate=rate, gate=gate, leak=leak, move_set=move_set,
                                           release_to=release_to,
                                           feed_stops_at_split=int(bool(cfg.get("feed_stops_at_split", False))),
                                           replica=rep, seed=seed, sweep=sweep, N=n, dims="x".join(str(x) for x in dims),
                                           F=f, stores=stores, fed=fed, leaked=leaked, n_split=n_split,
                                           n_contract=n_contract, n_clean=n_clean, open=op, curled=cu, broken=br,
                                           daughters=dau, daughter_energy=dau_e,
                                           residual=f + stores + dau_e - fed + leaked - first["f0"],
                                           first_split=first["first_split"] if first["first_split"] is not None else "",
                                           first_clean=first["first_clean"] if first["first_clean"] is not None else "",
                                           first_contract=first["first_contract"] if first["first_contract"] is not None else ""))
                        last = rows[-1]
                        print("lam=%.2f L=%d r=%g gate=%.1f leak=%g rep=%-2d | first split %s clean %s contract %s | end width %s F %.1f stores %.1f splits %d contractions %d clean %d open %d curled %d broken %d daughters %d (%.1f)"
                              % (lam, L, rate, gate, leak, rep, first["first_split"], first["first_clean"], first["first_contract"],
                                 "x".join(str(x) for x in last[2]), last[3], last[4], last[7], last[8], last[9], last[10], last[11], last[12], last[13], last[14]), flush=True)
    print("wrote", Path(out_dir) / (cfg["name"] + ".csv"))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
