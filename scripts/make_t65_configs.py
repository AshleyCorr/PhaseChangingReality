"""Write T65's configs (PREREGISTRATION T65, drafted 2026-10-15 and revised 16 October before any run): the owner's
slab, 4 x L x L with one direction curled, sealed under her tie "all at the last", given a spark in one store of a
shared bath, with and without a patch opened by hand, and an untied control. Usage:

    python scripts/make_t65_configs.py              # stage 1 (L = 8, 12); prints the queue lines
    python scripts/make_t65_configs.py --held-out   # L = 16, after the smaller sizes are read (rule 14)

Refuses to rewrite an existing config. Cells, stage 1 (8 configs):
  arm A  tied, lambda 1.20 and 1.25, L 8 and 12: sparks 16, 32, 64, 128; baths C = N and 4N; 8 replicas   (4 configs)
  arm B  tied, lambda 1.25, L 8 and 12, the patch k = 2 opened by hand: sparks 16, 32; C = N and 4N         (2 configs)
  control untied, lambda 1.25, L 8 and 12: spark 128; C = N and 4N                                         (2 configs)
30,000 sweeps read every 500, the final graph saved. Seeds from 20266501 in the order written.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20266501
TABLE = [0, 1.0, 2.0, 0]
PURPOSE = ("PREREGISTRATION.md T65 (drafted 2026-10-15, revised 2026-10-16, before any run; VISION Updates 47 and 57): "
           "does the owner's slab (4 x L x L, one direction curled) open into one flat space when given a push? Sealed, "
           "the spark in one store of a shared bath and conserved; %s; lambda %g; %s. Not exploratory. Every six-link "
           "result carries VISION Update 24's caveat.")


def cells(held_out=False):
    sizes = (16,) if held_out else (8, 12)
    out = []
    for L in sizes:
        for lam in (1.20, 1.25):
            out.append(("t65_a_lam%d_l%d" % (round(lam * 100), L), dict(dims=[4, L, L]), lam, True, [16.0, 32.0, 64.0, 128.0], "arm A, the push alone"))
        out.append(("t65_b_lam125_l%d" % L, dict(patch=dict(L=L, k=2)), 1.25, True, [16.0, 32.0], "arm B, the push on a patch of two columns opened by hand"))
        out.append(("t65_control_lam125_l%d" % L, dict(dims=[4, L, L]), 1.25, False, [128.0], "the untied control"))
    return out


def config(name, start, lam, tied, sparks, label, seed):
    cfg = {"_purpose": PURPOSE % (label, lam, "the tie 'all at the last', f = (0, a, 2a, 0), a = 4(lambda - 1)" if tied else "untied"),
           "name": name, "section": "T65"}
    cfg.update(start)
    cfg.update({"lambda": lam, "local_heat": False, "capacities": ["N", "4*N"]})
    if tied:
        cfg.update({"ftable_per_a": TABLE, "tie_label": "all at the last"})
    cfg.update({"sparks": sparks, "replicas": 8, "n_sweeps": 30000, "record_every": 500, "seed": seed, "save_adjacency": True})
    return cfg


def main(held_out=False):
    offset = 100 if held_out else 0
    for i, (name, start, lam, tied, sparks, label) in enumerate(cells(held_out)):
        path = ROOT / "configs" / (name + ".json")
        if path.exists():
            print("exists, not rewritten:", path.name, file=sys.stderr)
        else:
            path.write_text(json.dumps(config(name, start, lam, tied, sparks, label, SEED0 + offset + i), indent=2) + "\n", encoding="utf-8")
        print("run_sealed_curled_d", name)


if __name__ == "__main__":
    main("--held-out" in sys.argv[1:])
