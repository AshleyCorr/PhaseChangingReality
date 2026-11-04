"""Write the 28 configs of PREREGISTRATION T23 (the lambda map again, 120 decays per cell). Usage:

    python scripts/make_t23_configs.py

T8's settings exactly (configs/t8_lam105.json), one config per (lambda, N) so that each is one Batch job, with
fresh seeds: one per lambda, and each decay's stream from SeedSequence(seed, lx, ly, replica) as the runner
already does. Refuses to overwrite a config that exists.
"""
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LAMBDAS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]
SIDES = {64: [16, 4], 96: [24, 4], 144: [36, 4], 192: [48, 4]}


def config(lam, n):
    tag = "%03d" % round(lam * 100)
    return {
        "_purpose": ("PREREGISTRATION.md T23 (written 2026-09-24, before any run): T8's protocol at lambda = %.2f, "
                     "N = %d, with 120 decays and fresh seeds, read under the memoryless check sized to the sample. "
                     "Not exploratory." % (lam, n)),
        "name": "t23_lam%s_n%d" % (tag, n),
        "sides": [SIDES[n]],
        "g": 1.5,
        "replicas": 120,
        "n_sweeps": 100000,
        "block": 5,
        "stop_at": 0.98,
        "settle": 600,
        "settle_max": 100000,
        "seed": 20262300 + round(lam * 100),
        "lambda": lam,
        "save_adjacency": True,
        "record_f_200": True,
    }


def main():
    for lam in LAMBDAS:
        for n in SIDES:
            cfg = config(lam, n)
            path = CONFIGS / (cfg["name"] + ".json")
            if path.exists():
                raise SystemExit("refusing to overwrite %s" % path)
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            print(path.name)


if __name__ == "__main__":
    main()
