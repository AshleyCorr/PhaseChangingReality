"""The two kinds as moves on a flat torus (T103). Usage:

    python scripts/run_curl_pair_moves.py configs/t103_curl_pair_moves.json

Optional config key "price_from": "ledger" (default, T103) or "gravity" (her rule of 4 November after T103, T104).
One row per (side, lambda, barrier, replica): the census at the freeze (graphity/curl_pair_moves.py). Writes
results/<name>.csv and .meta.json and refuses to overwrite either. Seeds: seed + 1000003 * replica + 1009 * side
+ 101 * (index of lambda) + (index of barrier), every cell its own.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

from graphity import __version__
from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState
from graphity.results import ResultWriter, provenance


def main(path):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), graphity_version=__version__, numpy=np.__version__,
                python=platform.python_version(), **provenance())
    with ResultWriter(cfg["name"], meta) as out:
        for side in cfg["sides"]:
            for li, lam in enumerate(cfg["lambdas"]):
                for bi, B in enumerate(cfg["barriers"]):
                    for rep in range(cfg["replicas"]):
                        seed = cfg["seed"] + 1000003 * rep + 1009 * side + 101 * li + bi
                        adj, part = torus(side, side, NO_CAP)
                        m = PairMoves(CurlState(adj, part), lam=lam, barrier=float(B), seed=seed,
                                      price_from=cfg.get("price_from", "ledger"))
                        row = m.run(max_sweeps=cfg.get("max_sweeps", 10_000))
                        out.write(dict(side=side, n_points=side * side, lam=lam, barrier=B, replica=rep, seed=seed,
                                       **row))
                        print(f"L={side} lam={lam} B={B} rep={rep}: stores={row['stores']} (+{row['stores_plus']} "
                              f"-{row['stores_minus']}) photons={row['photons']} frozen_curls={row['frozen_curls']} "
                              f"sweeps={row['sweeps']} cluster={row['store_cluster']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
