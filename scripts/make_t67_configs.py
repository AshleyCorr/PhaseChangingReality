"""Write T67's configs (PREREGISTRATION T67, written 2026-10-16 before any registered run): the owner's untied slab,
4 x L x L with one direction curled, lambda 1.25, sealed, fed at fixed rates. Usage:

    python scripts/make_t67_configs.py              # stage 1 (L = 8, 12); prints the queue lines
    python scripts/make_t67_configs.py --held-out   # L = 16, after the smaller sizes are read (rule 14)

Refuses to rewrite an existing config. Cells per size: shared bath with C = N and C = 4N; one point's own store
("spot"); rates 1/16, 1/4, 1 and 4 units a sweep; 8 replicas; up to 20,000 sweeps read every 50; the final graph
saved. Seeds from 20266701 in the order written (held-out from 20266801).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20266701
RATES = [0.0625, 0.25, 1.0, 4.0]
PURPOSE = ("PREREGISTRATION.md T67 (written 2026-10-16, after one sandbox look and before any registered run; VISION "
           "Updates 58 and 59; rule 11's fifth use on piece 5): does the owner's untied slab (4 x L x L, one direction "
           "curled) open from a feed, energy arriving at a fixed rate into %s? lambda 1.25; rates %s units a sweep. "
           "Not exploratory. Every six-link result carries VISION Update 24's caveat.")


def cells(held_out=False):
    sizes = (16,) if held_out else (8, 12)
    out = []
    for L in sizes:
        out.append(("t67_shared_cn_l%d" % L, L, "shared", "N", "store 0 of a shared bath of N stores"))
        out.append(("t67_shared_c4n_l%d" % L, L, "shared", "4*N", "store 0 of a shared bath of 4N stores"))
        out.append(("t67_spot_l%d" % L, L, "spot", "N", "one point's own store, with a store at every point"))
    return out


def config(name, L, mode, capacity, label, seed):
    return {"_purpose": PURPOSE % (label, RATES), "name": name, "section": "T67", "dims": [4, L, L], "lambda": 1.25,
            "mode": mode, "capacity": capacity, "rates": RATES, "replicas": 8, "n_sweeps": 20000, "record_every": 50,
            "seed": seed, "save_adjacency": True}


def main(held_out=False):
    offset = 100 if held_out else 0
    for i, (name, L, mode, capacity, label) in enumerate(cells(held_out)):
        path = ROOT / "configs" / (name + ".json")
        if path.exists():
            print("exists, not rewritten:", path.name, file=sys.stderr)
        else:
            path.write_text(json.dumps(config(name, L, mode, capacity, label, SEED0 + offset + i), indent=2) + "\n", encoding="utf-8")
        print("run_slab_feed", name)


if __name__ == "__main__":
    main("--held-out" in sys.argv[1:])
