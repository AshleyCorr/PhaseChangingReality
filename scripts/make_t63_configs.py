"""Write T63's configs and its queue lines (PREREGISTRATION.md T63). Usage:

    python scripts/make_t63_configs.py              # stage 1: N = 96 and 192, ten cells, one config each
    python scripts/make_t63_configs.py --held-out   # N = 384, five cells; refused until the held-out numbers exist

The start is the curled tube itself (`"start": "tube"`), run by scripts/run_sixloop_widen.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "configs" / "t63_heldout_prediction.json"
LAM = 1.25
G = 1.5
MUS = (0.0, 0.025, 0.05, 0.10, 0.20)
STAGE1_LENGTHS = (24, 48)          # N = 96, 192
HELD_OUT_LENGTH = 96               # N = 384
SEED = 20266300


def name_of(mu, n):
    return "t63_mu%04d_n%d" % (round(1000 * mu), n)


def configs(held_out=False):
    out = []
    for length in ((HELD_OUT_LENGTH,) if held_out else STAGE1_LENGTHS):
        for mu in MUS:
            out.append({
                "_purpose": "PREREGISTRATION.md T63 (written 2026-10-12, before any run): the curled tube itself under a "
                            "price mu on loops of six, warm, cooled and read from its wiring. %s. Not exploratory."
                            % ("The held-out size" if held_out else "Stage 1"),
                "name": name_of(mu, 4 * length), "section": "T63", "start": "tube", "lambda": LAM, "mu": mu, "g": G,
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
