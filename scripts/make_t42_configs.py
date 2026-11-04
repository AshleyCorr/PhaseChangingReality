"""Write the configs and the Batch queue file for PREREGISTRATION T42 (2026-09-25). Usage:

    python scripts/make_t42_configs.py

Refuses to overwrite a config that exists.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
CAVEAT = "Every six-link result carries VISION Update 24's caveat: the reproduction gate is open."


def main():
    lines = []
    for weighted in (True, False):
        for e in (64, 128, 256, 512, 1024, 2048):
            name = "t42_%s_lam102_n512_e%d" % ("interchangeable" if weighted else "named", e)
            cfg = {"_purpose": ("PREREGISTRATION.md T42 (written 2026-09-25, before any run): does concentrated energy fold "
                                "flat six-link space near lambda = 1.02, where counting could pay for the fold (O55)? Flat "
                                "8 x 8 x 8, the whole energy %d in one store of a shared bath of 2N (T34's spread protocol), "
                                "%s. Not exploratory. %s"
                                % (e, "points interchangeable" if weighted else "named points, the control (same chain, no "
                                   "symmetry factor)", CAVEAT)),
                   "name": name, "section": "T42", "dims": [8, 8, 8], "lambda": 1.02, "capacities": ["2*N"],
                   "sparks": [float(e)], "replicas": 8, "n_sweeps": 50000, "record_every": 250,
                   "seed": 20264200 + e + (0 if weighted else 5000), "save_adjacency": True,
                   "interchangeable": True, "weighted": weighted}
            path = R / "configs" / (name + ".json")
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            lines.append("run_sealed_curled_d " + name)
    q = R / "cloud" / "queue" / "2026-09-25_t42.txt"
    if q.exists():
        raise FileExistsError(q)
    q.write_text("# PREREGISTRATION T42: the fold with interchangeable points near lambda = 1.02 (piece 8).\n"
                 "# Submitted by the run_queue workflow from the commit that carries this file, the configs and the "
                 "pre-registration.\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("written", len(lines))


if __name__ == "__main__":
    main()
