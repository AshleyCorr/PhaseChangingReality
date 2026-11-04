"""Write the configs of PREREGISTRATION T60 (the lambda map a fourth time: the wait timed with no detector, and a
size held out). Usage:

    python scripts/make_t60_configs.py              # stage 1: 28 configs, N = 64 to 192
    python scripts/make_t60_configs.py --held-out   # stage 2: 7 configs, N = 288

Stage 1 is T24's settings exactly (scripts/make_t24_configs.py), one config per (lambda, N) so that each is one
Batch job, with fresh seeds (one per lambda, 20266105 to 20266135) and one addition that reads the chain and draws
nothing: `record_detector`, which writes into each row the first look at which the decay no longer read as the
perfect tube (tested to leave every column of a run unchanged, tests/test_tube_decay_seeds.py).

Stage 2 is the held-out size, 72 x 4. It refuses to write until configs/t60_heldout_prediction.json exists, which
`scripts/analyse_t60.py --write-prediction` makes from stage 1 alone (CLAUDE.md rule 14). Its caps are doubled:
the chain offers any one local pair of links about 2/N times a sweep (the fair clock, ASSUMPTIONS O90), so a
conversion takes more chain sweeps in a longer tube, and 200,000 sweeps at N = 288 is at least the room in fair
sweeps that N = 192 has in stage 1. Named points. Refuses to overwrite a config that exists.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
LAMBDAS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]
SIDES = {64: [16, 4], 96: [24, 4], 144: [36, 4], 192: [48, 4]}
HELD_OUT = {288: [72, 4]}
PREDICTION = CONFIGS / "t60_heldout_prediction.json"


def config(lam, n, side, held_out=False):
    tag = "%03d" % round(lam * 100)
    cap = 200000 if held_out else 100000
    stage = ("stage 2, the held-out size, run after its numbers were written from stage 1"
             if held_out else "stage 1, T24's grid")
    return {
        "_purpose": ("PREREGISTRATION.md T60 (written 2026-10-10, before any run): the lambda map a fourth time, %s: "
                     "T24's protocol at lambda = %.2f, N = %d, 120 decays, fresh seeds, with the first look at which "
                     "each decay no longer read as the perfect tube recorded, so that the wait is timed with no "
                     "detector. Named points. Not exploratory." % (stage, lam, n)),
        "name": "t60_lam%s_n%d" % (tag, n),
        "sides": [side],
        "g": 1.5,
        "replicas": 120,
        "n_sweeps": cap,
        "block": 5,
        "stop_at": 0.98,
        "settle": 600,
        "settle_max": cap,
        "seed": 20266000 + round(lam * 100),
        "lambda": lam,
        "save_adjacency": True,
        "record_f_200": True,
        "record_detector": True,
    }


def configs(held_out=False):
    sides = HELD_OUT if held_out else SIDES
    return [config(lam, n, side, held_out) for lam in LAMBDAS for n, side in sides.items()]


def main(flag=""):
    held_out = flag == "--held-out"
    if held_out and not PREDICTION.exists():
        raise SystemExit("no %s: the numbers for the held-out size are written from stage 1 first" % PREDICTION.name)
    for cfg in configs(held_out):
        path = CONFIGS / (cfg["name"] + ".json")
        if path.exists():
            raise SystemExit("refusing to overwrite %s" % path)
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(path.name)


if __name__ == "__main__":
    main(*sys.argv[1:])
