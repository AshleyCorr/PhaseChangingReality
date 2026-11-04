"""Write T56's configs (PREREGISTRATION T56, written 2026-10-09 before any run). Usage:

    python scripts/make_t56_configs.py            # stage 1: the two smaller sizes of each geometry, plus controls
    python scripts/make_t56_configs.py --held-out # stage 2: the held-out sizes, after stage 1 is read (rule 14)

Stage 1, 68 configs: the slab 4 x L x L (one curled, the owner's shape for X) at L = 8 and 12 and the rod 4 x 4 x L
(two curled) at L = 18 and 36; lambda in {1.10, 1.15, 1.20, 1.25}; g in {1.5, 2.0, 2.5, 3.0}; the tie "all at the
last"; 8 replicas; 200,000 sweeps read every 1,000; final graphs saved. Four untied controls: the slab at lambda =
1.20, g = 2.0 and 2.5, both sizes. Stage 2, 32 configs: L = 16 (slab) and L = 72 (rod) at every (lambda, g), tied.
Seeds run from 20265601 in the order written; no two configs share one. The queue manifests are written beside the
configs by hand at launch, after the owner's prediction is on the record.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS, COUPLINGS = (1.10, 1.15, 1.20, 1.25), (1.5, 2.0, 2.5, 3.0)
STAGE1 = (("slab", (8, 12)), ("rod", (18, 36)))
STAGE2 = (("slab", (16,)), ("rod", (72,)))
CONTROLS = (("slab", 8, 1.20, 2.0), ("slab", 8, 1.20, 2.5), ("slab", 12, 1.20, 2.0), ("slab", 12, 1.20, 2.5))
SEED0 = 20265601


def dims_of(geometry, length):
    return [4, length, length] if geometry == "slab" else [4, 4, length]


def purpose(geometry, dims, lam, g, tie):
    shape = ("the slab, one direction curled and two open, the owner's shape for X" if geometry == "slab"
             else "the rod, two directions curled and one open, her spaghetti")
    tie_text = ("the owner's tie 'all at the last', f = (0, a, 2a, 0) with a = 4(lambda - 1)" if tie == "all_at_the_last"
                else "no tie (the untied control)")
    return ("PREREGISTRATION.md T56 (written 2026-10-09, before any run; VISION Update 47; ASSUMPTIONS O106): %s in a "
            "warm bath under %s. The exact %s torus, six links, lambda %g, the thermal chain (Metropolis in H + T_f) at "
            "the fixed coupling g = %g; named points, no push. Not exploratory. Every six-link result carries VISION "
            "Update 24's caveat: the reproduction gate is open." % (shape, tie_text, " x ".join(map(str, dims)), lam, g))


def config(name, geometry, length, lam, g, tie, seed):
    dims = dims_of(geometry, length)
    return {"_purpose": purpose(geometry, dims, lam, g, tie), "name": name, "section": "T56", "dims": dims,
            "lambda": lam, "g": g, "tie": tie, "replicas": 8, "n_sweeps": 200000, "record_every": 1000,
            "seed": seed, "save_adjacency": True}


def name_of(geometry, length, lam, g, tie):
    return "t56_%s_l%d_lam%d_g%d%s" % (geometry, length, round(lam * 100), round(g * 10),
                                       "" if tie == "all_at_the_last" else "_untied")


def stage(held_out):
    out = []
    if not held_out:
        for geometry, lengths in STAGE1:
            for length in lengths:
                for lam in LAMBDAS:
                    for g in COUPLINGS:
                        out.append((geometry, length, lam, g, "all_at_the_last"))
        for geometry, length, lam, g in CONTROLS:
            out.append((geometry, length, lam, g, "none"))
    else:
        for geometry, lengths in STAGE2:
            for length in lengths:
                for lam in LAMBDAS:
                    for g in COUPLINGS:
                        out.append((geometry, length, lam, g, "all_at_the_last"))
    return out


def main(argv):
    held_out = "--held-out" in argv
    cells = stage(held_out)
    offset = 0 if not held_out else len(stage(False))
    written = []
    for i, (geometry, length, lam, g, tie) in enumerate(cells):
        name = name_of(geometry, length, lam, g, tie)
        path = ROOT / "configs" / (name + ".json")
        if path.exists():
            raise FileExistsError("%s exists; configs are not rewritten" % path)
        path.write_text(json.dumps(config(name, geometry, length, lam, g, tie, SEED0 + offset + i), indent=2) + "\n",
                        encoding="utf-8")
        written.append(name)
    print("\n".join("run_curled_bath_tie_d " + n for n in written))
    print("%d configs written" % len(written), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
