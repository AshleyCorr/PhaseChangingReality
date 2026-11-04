"""Does space refold when refolding is free? Triangles and pentagons allowed. Usage:

    python scripts/run_general_chain.py configs/refold_sheet_left_alone.json

EXPLORATORY (VISION plan step 5; ASSUMPTIONS Q10, Q18). At the published prices a closed
30-point piece with a triangle and a pentagon on every edge (the icosidodecahedron) has exactly
the energy of the flat sheet, and both are dips. So in this corner of the published model,
folding space into a closed piece costs nothing. The question is what a run does with that.

Three starts, named in the config as "start":
  "sheet"   a flat lx x ly torus                       -- does space refold when left alone?
  "pieces"  k copies of the closed 30-point piece      -- does a refolded arrangement survive?
  "quench"  melt at `melt_g` for `melt_sweeps`, then run cold -- what does a hot patch cool into?

The third is the one that bears on the author's picture of a black hole: concentrated energy is
what gravity delivers, so the question is whether a hot region cools into a knot (her re-curled
region) or stays a random bubble ([T25]'s melted one).

Every row is a sweep: H, the census of triangles, squares and pentagons, the number of connected
pieces and the largest one's share. Sizes are small because the energy is recomputed exactly from
scratch at every attempted move (about 3 ms at 60 vertices); that is the price of allowing
triangles, and it is paid so that nothing about the energy is approximated.
"""
import json
import platform
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity import general_chain as gc                                  # noqa: E402
from graphity.full_curvature import square_sheet                          # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402


def starting_graph(cfg, rep):
    """The graph a replica starts from, and the number of sweeps already spent making it."""
    kind = cfg["start"]
    if kind == "sheet":
        return square_sheet(*cfg["sides"]), 0
    if kind == "pieces":
        g = nx.disjoint_union_all([gc.icosidodecahedron() for _ in range(int(cfg["pieces"]))])
        return g, 0
    if kind == "quench":
        seed = int(np.random.SeedSequence([int(cfg["seed"]), rep, 1]).generate_state(1)[0])
        hot, _, _ = gc.run(square_sheet(*cfg["sides"]), float(cfg["melt_g"]),
                           int(cfg["melt_sweeps"]), seed)
        return hot, int(cfg["melt_sweeps"])
    raise ValueError("start must be sheet, pieces or quench")


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__,
                networkx=nx.__version__, purpose=cfg.get("_purpose", ""))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for g_coupling in cfg["couplings"]:
            for rep in range(int(cfg["replicas"])):
                start, spent = starting_graph(cfg, rep)
                seed = int(np.random.SeedSequence(
                    [int(cfg["seed"]), rep, int(float(g_coupling) * 1000)]).generate_state(1)[0])
                _, rows, acceptance = gc.run(start, float(g_coupling), int(cfg["n_sweeps"]),
                                             seed, every=int(cfg.get("every", 10)))
                for row in rows:
                    out.write(dict(g=g_coupling, replica=rep, n=start.number_of_nodes(),
                                   melted_for=spent, acceptance=round(acceptance, 4), **row))
                print("g=%-6s rep %d: H/N %+.3f -> %+.3f, pieces %d, acceptance %.3f" % (
                    g_coupling, rep, rows[0]["h_per_vertex"], rows[-1]["h_per_vertex"],
                    rows[-1]["pieces"], acceptance), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
