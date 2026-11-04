"""Write the configs of T61 (PREREGISTRATION.md section T61). Usage:

    python scripts/make_t61_configs.py              # stage 1: lambda = 1.25, 1.40 at N = 96 and 192
    python scripts/make_t61_configs.py --held-out   # N = 384, one cell to a file; refuses until the prediction
                                                    # for it is written
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "configs" / "t61_heldout_prediction.json"
LAMBDAS = (1.25, 1.40)
STAGE1_LENGTHS, HELD_OUT_LENGTH = (24, 48), 96
SEED = 20266100


def configs(held_out=False):
    out = []
    ratios = [2, 1, 0.5, 0.333333, 0.25]
    for lam in LAMBDAS:
        for length, cell in ([(HELD_OUT_LENGTH, r) for r in ratios] if held_out else [(x, None) for x in STAGE1_LENGTHS]):
            n = 4 * length
            out.append({
                "_purpose": "PREREGISTRATION.md T61 (written 2026-10-11, before any run): a sealed tube whose released "
                            "energy can only go into its stores, then drained and cooled; the end state read from its "
                            "wiring. %s Not exploratory." % ("The held-out size (rule 14)." if held_out else "Stage 1."),
                "name": "t61_lam%03d_n%d" % (round(lam * 100), n) + ("" if cell is None else "_c%d" % round(cell * n)),
                "section": "T61",
                "lambda": lam,
                "lengths": [length],
                "stores_per_point": ratios if cell is None else [cell],
                "replicas": 24,
                "hold_sweeps_at_192": 20000,
                "leak": 0.02,
                "drain_stop": 0.5,
                "cold_sweeps": 2000,
                "cold_g": 0.05,
                "save_adjacency": True,
                "seed": SEED,
            })
    return out


def main(flag=""):
    held_out = flag == "--held-out"
    if held_out and not PREDICTION.exists():
        raise SystemExit("the prediction for N = 384 is not written yet (%s); nothing made" % PREDICTION.name)
    for cfg in configs(held_out):
        path = ROOT / "configs" / (cfg["name"] + ".json")
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("wrote", path.name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
