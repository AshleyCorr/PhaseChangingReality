"""Write T64's configs and its queue lines (PREREGISTRATION.md T64). Usage:

    python scripts/make_t64_configs.py              # stage 1: N = 96 and 192, ten cells, one config each
    python scripts/make_t64_configs.py --held-out   # N = 384, five cells; refused until the held-out numbers exist

T64 is T63's design on fresh seeds: the curled tube itself (`"start": "tube"`), run by scripts/run_sixloop_widen.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_t63_configs as t63                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "configs" / "t64_heldout_prediction.json"
SEED = 20266400


def name_of(mu, n):
    return "t64_mu%04d_n%d" % (round(1000 * mu), n)


def configs(held_out=False):
    out = []
    for cfg in t63.configs(held_out):
        cfg = dict(cfg)
        n = 4 * cfg["lengths"][0]
        cfg["_purpose"] = ("PREREGISTRATION.md T64 (written 2026-10-12, before any run): T63's design on fresh seeds, the "
                           "curled tube itself under a price mu on loops of six, warm, cooled and read from its wiring. "
                           "%s. Not exploratory." % ("The held-out size" if held_out else "Stage 1"))
        cfg["name"], cfg["section"], cfg["seed"] = name_of(cfg["mu"], n), "T64", SEED
        out.append(cfg)
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
