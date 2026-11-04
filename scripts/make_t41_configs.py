"""Write the configs and the Batch queue file for PREREGISTRATION T41 (2026-09-25). Usage:

    python scripts/make_t41_configs.py

Refuses to overwrite a config that exists.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
CAVEAT = "Every six-link result carries VISION Update 24's caveat: the reproduction gate is open."


def main():
    lines = []
    for lam, spark in ((1.25, 36.0), (1.40, 29.0)):
        for tag, cap in (("c4n", "4*N"), ("c2n", "2*N"), ("cn", "N"), ("cn2", "N/2"), ("cn4", "N/4"), ("cn8", "N/8")):
            name = "t41_lam%03d_n384_%s" % (round(lam * 100), tag)
            cfg = {"_purpose": ("PREREGISTRATION.md T41 (written 2026-09-25, before any run): does the new space need room "
                                "for the burp in three directions? 4 x 8 x 12 (one curled), lambda %.2f, spark %.1f (the "
                                "exact wall is %s), C = %s. Not exploratory. %s"
                                % (lam, spark, "36" if lam == 1.25 else "28.8", cap, CAVEAT)),
                   "name": name, "section": "T41", "dims": [4, 8, 12], "lambda": lam, "capacities": [cap],
                   "sparks": [spark], "replicas": 12, "n_sweeps": 100000, "record_every": 200,
                   "seed": 20264100 + len(lines), "save_adjacency": True}
            path = R / "configs" / (name + ".json")
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            lines.append("run_sealed_curled_d " + name)
    q = R / "cloud" / "queue" / "2026-09-25_t41.txt"
    if q.exists():
        raise FileExistsError(q)
    q.write_text("# PREREGISTRATION T41: room for the burp in three directions (piece 4).\n"
                 "# Submitted by the run_queue workflow from the commit that carries this file, the configs and the "
                 "pre-registration.\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("written", len(lines))


if __name__ == "__main__":
    main()
