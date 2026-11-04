"""Write T62's configs and its queue lines (PREREGISTRATION.md T62). Usage:

    python scripts/make_t62_configs.py              # stage 1: N = 96 and 192, twenty cells, one config each
    python scripts/make_t62_configs.py --held-out   # N = 384, ten cells; refused until the held-out numbers exist

One cell (mu, g, N) to a config, so that the cells run side by side. A replica's seed depends on (seed, N, lambda,
mu, g, replica) and not on the file it is in.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "configs" / "t62_heldout_prediction.json"
LAM = 1.25
MUS = (0.0, 0.10, 0.20, 0.25, 0.30)
GS = (1.5, 2.0)
STAGE1_LENGTHS = (24, 48)          # N = 96, 192
HELD_OUT_LENGTH = 96               # N = 384
SEED = 20266200


def name_of(mu, g, n):
    return "t62_mu%03d_g%03d_n%d" % (round(100 * mu), round(100 * g), n)


def configs(held_out=False):
    out = []
    for length in ((HELD_OUT_LENGTH,) if held_out else STAGE1_LENGTHS):
        for mu in MUS:
            for g in GS:
                out.append({
                    "_purpose": "PREREGISTRATION.md T62 (written 2026-10-12, before any run): the tube six around under a "
                                "price mu on loops of six, warm, cooled and read from its wiring. %s. Not exploratory."
                                % ("The held-out size" if held_out else "Stage 1"),
                    "name": name_of(mu, g, 4 * length), "section": "T62", "lambda": LAM, "mu": mu, "g": g,
                    "lengths": [length], "replicas": 24, "hold_sweeps_at_192": 30000, "cool_sweeps_at_192": 60000,
                    "cool_to": 0.5, "cold_sweeps": 1000, "cold_g": 0.05, "save_adjacency": True, "seed": SEED})
    return out


def main(flag=""):
    held_out = flag == "--held-out"
    if held_out and not PREDICTION.exists():
        raise SystemExit("the held-out numbers (%s) are not written yet; nothing made" % PREDICTION.name)
    lines = []
    for cfg in configs(held_out):
        path = ROOT / "configs" / (cfg["name"] + ".json")
        if path.exists():
            raise SystemExit("refusing to overwrite %s" % path)
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
        lines.append("run_sixloop_widen " + cfg["name"])
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
