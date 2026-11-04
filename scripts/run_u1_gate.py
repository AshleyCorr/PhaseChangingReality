"""Gate U1 (TASKS): the Wilson-action Monte Carlo of graphity.u1 against the exact answer on flat four-link tori.
Usage:

    python scripts/run_u1_gate.py configs/gate_u1_2d.json

A reproduction check of the sampler, not a test of the hypothesis (ASSUMPTIONS Q29; VISION Update 96). For each torus
and coupling beta: the mean of cos(phi) over squares after each measured sweep, its error from equal blocks, the exact
torus value (graphity.u1.exact_mean_cos_2d), the infinite-lattice value I1/I0, and the acceptance. Writes
results/<name>.csv and .meta.json and refuses to overwrite either.

Config keys: "tori" (list of [Lx, Ly], each side at least 6 so the squares are the plaquettes), "betas", "steps"
(proposal half-width per beta), "n_equil", "n_meas", "n_blocks", "seed", and optional "winding_move" (default false:
the chain of Gate U1; true: one winding move per sweep, graphity.u1.Field.instanton, added for Gate U1').
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np
from scipy.special import iv

from graphity import __version__
from graphity.cqg_d import torus
from graphity.results import ResultWriter, provenance
from graphity.u1 import Field, exact_mean_cos_2d


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance())
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for dims in cfg["tori"]:
            f = Field(*torus(dims))
            inst = f.instanton() if cfg.get("winding_move", False) else None
            for k, (beta, step) in enumerate(zip(cfg["betas"], cfg["steps"])):
                seed = int(np.random.SeedSequence([cfg["seed"], *dims, k]).generate_state(1)[0])
                th_end, series, acc = f.metropolis(np.zeros(f.n_links), float(beta), float(step),
                                              cfg["n_equil"], cfg["n_meas"], seed,
                                              instanton=inst)
                blocks = series.reshape(cfg["n_blocks"], -1).mean(axis=1)
                err = blocks.std(ddof=1) / np.sqrt(cfg["n_blocks"])
                exact = exact_mean_cos_2d(float(beta), f.n_squares)
                out.write(dict(lx=dims[0], ly=dims[1], squares=f.n_squares, beta=beta, step=step,
                               mean_cos=series.mean(), err=err, exact_torus=exact,
                               infinite=iv(1, beta) / iv(0, beta), z_torus=(series.mean() - exact) / err,
                               z_infinite=(series.mean() - iv(1, beta) / iv(0, beta)) / err,
                               acceptance=acc, seed=seed,
                               winding_move=bool(cfg.get("winding_move", False)), final_winding=f.winding(th_end)
                               if inst is not None else ""))
                print(dims, beta, round(series.mean(), 5), round(exact, 5), round(acc, 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
