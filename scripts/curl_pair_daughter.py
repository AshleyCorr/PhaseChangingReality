"""T97: the curl-pair daughter in the first form of the owner's rules (PREREGISTRATION T97). Exact, no chain.

Usage: python scripts/curl_pair_daughter.py <stage>      stage is "stage1" (R = 8, 16) or "heldout" (R = 32)
"""
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                       # noqa: E402
from graphity.curl_pairs import daughter_front           # noqa: E402

STAGES = {"stage1": (8, 16), "heldout": (32,)}
LAMS = (1.10, 1.25, 1.50, 1.75)
L = 32


def main(stage, out_dir="results"):
    name = "t97_curl_pair_daughter_" + stage
    path = Path(out_dir) / (name + ".csv")
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    rows = [daughter_front(L, R, lam) for R in STAGES[stage] for lam in LAMS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    meta = dict(script="scripts/curl_pair_daughter.py", stage=stage, L=L, lams=LAMS, radii=STAGES[stage], unit=1.0,
                package=__version__, python=platform.python_version(), numpy=np.__version__,
                preregistration="PREREGISTRATION.md section T97")
    (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main(sys.argv[1])
