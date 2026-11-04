"""Write T75's configs (PREREGISTRATION T75 with its Amendments 1 to 7). Usage:

    python scripts/make_t75_configs.py --couplings G1 G2 G3

The three couplings are required and are not chosen here: they are fixed by an amendment, read off the equilibrium
curve already measured for the flat sheet at lambda = 1.25, before launch. One config per (size, placement,
coupling): sizes N = 512 (T51's t51_l128_10k end states) and 1,024 (t51_l256_300k); the held-out N = 2,048 is not
written here (rule 14: its prediction comes first). Three cells per size and coupling (Amendment 7): the single flaw far from knots, the single flaw beside
a knot (Amendments 3 and 4), and the pair of flaws far from knots; 32 replicas, 2,000 fair sweeps read every fair sweep. Seeds from SEED0 in the order written. Refuses to rewrite an existing config.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20267501
SIZES = [(512, "t51_l128_10k"), (1024, "t51_l256_300k")]
PLACEMENTS = ["far", "beside", "pair"]
PURPOSE = ("PREREGISTRATION.md T75 with its Amendments 1 to 7: does %s in the product of a burp (T51's end states, "
           "N = %d) heal, stay, travel or spread, and at what speed? A flaw is the cheapest valid switch at a point "
           "costing more than zero, priced at lambda = 1.25 (beside a knot, the cheapest touching no knot point). "
           "Coupling g = %s, fixed by amendment before launch. Not exploratory. At lambda != 1 this is our family, not "
           "combinatorial quantum gravity.")
WHERE = {"far": "a single flaw far from every knot (at least 6 steps)",
         "beside": "a single flaw beside a knot (nearest point 1 or 2 steps from a knot point)",
         "pair": "a pair of flaws, both far from every knot, their nearest points 4 steps apart (Amendment 7)"}


def tag(g):
    return ("%g" % g).replace(".", "p")


def config(n, run, place, g, seed):
    return {"_purpose": PURPOSE % (WHERE[place], n, g), "name": "t75_n%d_%s_g%s" % (n, place, tag(g)),
            "section": "T75", "states_dir": "results", "states": run, "N": n, "lambda": 1.25, "g": g,
            "placement": place, "replicas": 32, "fair_sweeps": 2000, "record_every": 1, "n_ref": 96, "step": 2,
            "flaw_reach": 3, "seed": seed}


def main(couplings, out_dir=ROOT / "configs"):
    if len(couplings) != 3 or len(set(couplings)) != 3:
        raise ValueError("three different couplings are registered")
    written = []
    i = 0
    for n, run in SIZES:
        for place in PLACEMENTS:
            for g in couplings:
                cfg = config(n, run, place, float(g), SEED0 + i)
                i += 1
                path = Path(out_dir) / (cfg["name"] + ".json")
                if path.exists():
                    print("exists, not rewritten:", path.name, file=sys.stderr)
                else:
                    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                written.append(path)
                print("run_t75", cfg["name"])
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--couplings", type=float, nargs=3, required=True)
    main(ap.parse_args().couplings)
