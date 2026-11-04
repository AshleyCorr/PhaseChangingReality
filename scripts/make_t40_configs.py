"""Write the configs and the Batch queue file for PREREGISTRATION T40 (2026-09-25). Usage:

    python scripts/make_t40_configs.py

Refuses to overwrite a config that exists.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
CAVEAT = ("Every eight-link result carries VISION Update 24's caveat: no published eight-link curve exists and the kernel "
          "is validated at four and six links.")


def write(name, cfg):
    path = R / "configs" / (name + ".json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return name


def main():
    lines = []
    for spark in (20, 30, 40, 60, 80, 120, 160):
        name = "t40_three_lam125_n768_e%d" % spark
        cfg = {"_purpose": ("PREREGISTRATION.md T40 (written 2026-09-25, before any run): the push that starts the change in "
                            "four directions; 4 x 4 x 4 x 12 (three curled), lambda 1.25, spark %d (the cheapest first move "
                            "is 20), C = N/2. Not exploratory. %s" % (spark, CAVEAT)),
               "name": name, "section": "T40", "dims": [4, 4, 4, 12], "lambda": 1.25, "capacities": ["N/2"],
               "sparks": [float(spark)], "replicas": 6, "n_sweeps": 50000, "record_every": 250,
               "seed": 20264000 + spark, "save_adjacency": True}
        lines.append("run_sealed_curled_d " + write(name, cfg))
    for spark in (20, 40, 80, 160):
        name = "t40_gas_lam110_n1024_e%d" % spark
        cfg = {"_purpose": ("PREREGISTRATION.md T40 (written 2026-09-25, before any run): the push that starts the change in "
                            "four directions; a gas of four 8-cubes (all four curled), lambda 1.10, spark %d (the cheapest "
                            "first move is 19.2), C = N/2. Not exploratory. %s" % (spark, CAVEAT)),
               "name": name, "section": "T40", "gas": {"dims": [4, 4, 4, 4], "copies": 4}, "lambda": 1.10,
               "capacities": ["N/2"], "sparks": [float(spark)], "replicas": 6, "n_sweeps": 50000, "record_every": 250,
               "seed": 20264500 + spark, "save_adjacency": True}
        lines.append("run_sealed_curled_d " + write(name, cfg))
    q = R / "cloud" / "queue" / "2026-09-25_t40.txt"
    if q.exists():
        raise FileExistsError(q)
    q.write_text("# PREREGISTRATION T40: the push that starts the change in four directions (piece 13).\n"
                 "# Submitted by the run_queue workflow from the commit that carries this file, the configs and the "
                 "pre-registration.\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("written", len(lines))


if __name__ == "__main__":
    main()
