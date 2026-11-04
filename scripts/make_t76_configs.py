"""Write T76's configs (PREREGISTRATION T76, Design). Usage:

    python scripts/make_t76_configs.py --couplings G1 G2 G3 G4
    python scripts/make_t76_configs.py --first-look G

--first-look (Amendment 4, the owner's option 2 of 20 October): the registered coupling rule cannot be met (Amendment
3, O141), so one coupling is run as a first look, at L = 16 and 24 and the four values of q, 8 cells, seeds from SEED0,
with 2,000 fair sweeps of settling (Amendment 3 found 200 too short); rho_target is null.

The four couplings are required and are not chosen here: T76 fixes them by a rule, the couplings at which the settled
flat 16 x 16 sheet has rho_eq = 0.003, 0.01, 0.03 and 0.1, found by bisection on settling runs alone
(scripts/settle_t76.py) and written into an amendment before any growth runs. Give them in that order (coldest
first, rho_eq 0.003). One config per (L, g) cell for L = 16 and 24; the held-out L = 32 is not written here (rule 14:
its numbers with ranges come first, after L = 16 and 24 are read). 16 replicas per cell; 200 fair sweeps of
settling, at most 400 of growth, stop at 8 times the starting N. Seeds from SEED0, one per cell, in the order written
(L = 16 at the four couplings, then L = 24). Refuses to rewrite an existing config.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20267601
SIZES = (16, 24)
TARGETS = (0.003, 0.01, 0.03, 0.1)
PURPOSE = ("PREREGISTRATION.md T76: as space grows by the owner's row rule (pairs of flat rows wherever the grid is "
           "intact and no link within three steps is curled, every boundary offered once per fair sweep), does the "
           "density of missing squares per point stay constant? L = %d, coupling g = %s (the coupling at which the "
           "settled flat 16 x 16 sheet has rho_eq = %s, fixed by amendment before launch). Not exploratory. At "
           "lambda != 1 this is our family, not combinatorial quantum gravity.")


def tag(g):
    return ("%g" % g).replace(".", "p")


Q_VALUES = (1 / 16, 1 / 64, 1 / 256, 1 / 1024)   # Amendment 2: the chance each clear offer is taken, scanned downward


def qtag(q):
    return "q%d" % round(1 / q)


def config(size, g, target, seed, q):
    return {"_purpose": PURPOSE % (size, "%g" % g, "%g" % target) + " Amendment 2: each clear offer taken with chance q = 1/%d." % round(1 / q),
            "name": "t76_l%d_g%s_%s" % (size, tag(g), qtag(q)),
            "section": "T76", "L": size, "lambda": 1.25, "g": g, "rho_target": target, "q": q, "replicas": 16,
            "n_settle": 200, "n_growth": 4000, "cap_factor": 8, "n_ref": 96, "reach": 3, "seed": seed}


FIRST_LOOK = ("PREREGISTRATION.md T76, Amendment 4 (the owner's option 2 of 20 October): a first look at one coupling, "
              "g = %s, the one at which the settled flat 16 x 16 sheet holds a small patchy density without melting "
              "(Amendment 3, O141), since no coupling meets the registered rule. As space grows by the owner's row "
              "rule, each clear offer taken with chance q = 1/%d, does the density of missing squares per point stay "
              "constant? L = %d. Not exploratory; labelled a first look. At lambda != 1 this is our family, not "
              "combinatorial quantum gravity.")


def first_look(g, out_dir=ROOT / "configs"):
    """Amendment 4's eight cells at one coupling."""
    g = float(g)
    if g <= 0:
        raise ValueError("couplings are positive")
    written = []
    for i, (size, q) in enumerate((s, q) for s in SIZES for q in Q_VALUES):
        cfg = config(size, g, 0, SEED0 + i, q)
        cfg["rho_target"] = None
        cfg["_purpose"] = FIRST_LOOK % ("%g" % g, round(1 / q), size)
        cfg["n_settle"] = 2000
        path = Path(out_dir) / (cfg["name"] + ".json")
        if path.exists():
            print("exists, not rewritten:", path.name, file=sys.stderr)
        else:
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        written.append(path)
        print("run_t76", cfg["name"])
    return written


def main(couplings, out_dir=ROOT / "configs"):
    couplings = [float(g) for g in couplings]
    if len(couplings) != len(TARGETS) or len(set(couplings)) != len(TARGETS):
        raise ValueError("four different couplings are registered")
    if any(g <= 0 for g in couplings):
        raise ValueError("couplings are positive")
    written = []
    i = 0
    for size in SIZES:
        for g, target in zip(couplings, TARGETS):
            for q in Q_VALUES:
                cfg = config(size, g, target, SEED0 + i, q)
                i += 1
                path = Path(out_dir) / (cfg["name"] + ".json")
                if path.exists():
                    print("exists, not rewritten:", path.name, file=sys.stderr)
                else:
                    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                written.append(path)
                print("run_t76", cfg["name"])
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--couplings", type=float, nargs=4,
                     help="the couplings for rho_eq = 0.003, 0.01, 0.03, 0.1, in that order, from the amendment")
    grp.add_argument("--first-look", type=float, help="one coupling, Amendment 4")
    args = ap.parse_args()
    if args.first_look is not None:
        first_look(args.first_look)
    else:
        main(args.couplings)
