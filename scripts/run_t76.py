"""T76: as space grows by the owner's row rule, does the density of missing squares stay constant? Usage:

    python scripts/run_t76.py configs/t76_l16_g2p5.json [out_dir]

Implements PREREGISTRATION.md section T76 (Design). One config per (g, L) cell. Not exploratory once its couplings
are fixed by amendment; nothing here chooses them (scripts/make_t76_configs.py takes them as a required argument).

Each replica starts from the flat L x L torus (graphity.row_growth.GrowthState.from_torus; four links, lambda 1.25,
no cap) and runs graphity.row_growth.run_growth: `n_settle` (200) fair sweeps with growth off, then growth, each fair
sweep one block of the chain (Metropolis at coupling g, the fair clock of O90 with `n_ref` 96) followed by one growth
step (every boundary along both axes offered once, in an order drawn from the replica's stream; a pair of flat rows
inserted wherever the grid is intact across the boundary and no link within `reach` (3) steps of its two rows sits on
more than two squares), until N reaches `cap_factor` (8) times its start or `n_growth` (400) fair sweeps of growth
have passed. The replica's streams come from SeedSequence([seed, replica]).

Rows: one per reading (before any attempt, then after every fair sweep), with the cell's g, lambda, L and the
replica, and the reading's fair_sweep, phase (start, settle, grow), attempts, acceptance, N, Lx, Ly, S, X, D (missing
squares, sum_e max(0, 2 - S_e)), rho = D / N, curled_share (links with S_e > 2 over all links) and rows_added in that
fair sweep. The verdicts are read by scripts/analyse_t76.py.

Config keys: name, section ("T76"), L, lambda, g, replicas, n_settle, n_growth, cap_factor, n_ref, reach, seed.
At lambda != 1 this is our family around the published model, not combinatorial quantum gravity.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                                  # noqa: E402
from graphity.cqg import NO_CAP, is_valid                                         # noqa: E402
from graphity.results import ResultWriter, provenance                             # noqa: E402
from graphity.row_growth import GrowthState, run_growth                           # noqa: E402

KEYS = ("name", "section", "L", "lambda", "g", "replicas", "n_settle", "n_growth", "cap_factor", "n_ref", "reach",
        "seed")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("section") != "T76":
        raise ValueError("not a T76 config")
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        raise ValueError("config lacks %s" % ", ".join(missing))
    return cfg


def run_replica(cfg, rep):
    """(readings, final state) of one replica."""
    state = GrowthState.from_torus(int(cfg["L"]))
    recs = run_growth(state, float(cfg["g"]), float(cfg["lambda"]), int(cfg["n_settle"]), int(cfg["n_growth"]),
                      int(cfg["cap_factor"]), seed=[int(cfg["seed"]), int(rep)], n_ref=int(cfg["n_ref"]),
                      reach=int(cfg["reach"]), q=float(cfg.get("q", 1.0)))
    if not is_valid(state.adj, NO_CAP):
        raise RuntimeError("replica %d ended in an invalid state" % rep)
    return recs, state


def main(path, out_dir="results"):
    cfg = load_config(path)
    for p in (Path(out_dir) / (cfg["name"] + ".csv"), Path(out_dir) / (cfg["name"] + ".meta.json")):
        if p.exists():
            raise FileExistsError("results are append-only; %s exists" % p)
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section T76")
    finals = []
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(int(cfg["replicas"])):
            t0 = time.time()
            recs, state = run_replica(cfg, rep)
            base = dict(g=float(cfg["g"]), lam=float(cfg["lambda"]), L=int(cfg["L"]), q=float(cfg.get("q", 1.0)),
                        replica=rep)
            for r in recs:
                out.write(dict(**base, **r))
            finals.append(dict(replica=rep, N=state.n, Lx=state.lx, Ly=state.ly, fair_sweeps=recs[-1]["fair_sweep"]))
            print("%s rep %d: N %d -> %d (%d x %d) after %d fair sweeps, rho %.4f, %.0f s"
                  % (cfg["name"], rep, recs[0]["N"], state.n, state.lx, state.ly, recs[-1]["fair_sweep"],
                     recs[-1]["rho"], time.time() - t0), flush=True)
        meta["finals"] = finals


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
