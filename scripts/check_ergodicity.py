"""Task T4: list every state at the smallest sizes and check that the edge switch joins all of them. Usage:

    python scripts/check_ergodicity.py configs/ergodicity_small.json

Exhaustive, not a simulation: no random numbers, the same answer every time. The
method and the argument are in src/graphity/small_graphs.py. Config keys:
    "sizes"  vertices per side, n; the graph has N = 2n vertices.
    "caps"   list of null (no cap, the space of [T25]) and/or 2 (the capped model of [KTB19 Sec. 4]).

Writes one row per class (states that differ only by renaming vertices within a
side count as one class). Columns that describe the whole size are repeated on
every row: labelled_total (all labelled states, counted directly), reached_total
(labelled states in the classes the switch reaches from one start) and
all_joined (the two agree, which proves both that the list of classes is
complete and that the switch joins them). A size with no states at all gets one
row with class_id -1.
"""
import json
import platform
import sys
from pathlib import Path

import networkx
import numba
import numpy as np

from graphity import __version__
from graphity.cqg import CAP, NO_CAP
from graphity.results import ResultWriter
from graphity.small_graphs import check


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    for cap in cfg["caps"]:
        if cap not in (CAP, None):
            raise ValueError(f"each cap must be {CAP} or null, not {cap!r}")
    meta = dict(config=cfg, package=__version__, python=platform.python_version(), numpy=np.__version__,
                numba=numba.__version__, networkx=networkx.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for cap in cfg["caps"]:
            for n in cfg["sizes"]:
                report = check(n, NO_CAP if cap is None else CAP)
                whole = dict(N=2 * n, cap="none" if cap is None else cap, labelled_total=report["labelled"],
                             reached_total=report["reached"], all_joined=report["ergodic"])
                empty = dict(class_id=-1, squares=0, surplus=0, pieces=0, largest=0, in_babies=0, cubes=0,
                             symmetries=0, labelled_states=0, neighbouring_classes=0, valid_under_cap=False)
                for row in report["classes"] or [empty]:
                    out.write({**whole, **row})
                print(f"N = {2 * n}, cap {whole['cap']}: {report['labelled']:,} labelled states, "
                      f"{len(report['classes'])} classes, all joined by switches: {report['ergodic']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
