"""Write the configs and the Batch queue files for PREREGISTRATION T37, T38 and T39 (2026-09-25). Usage:

    python scripts/make_t37_t39_configs.py

Refuses to overwrite a config that exists (a config on the record is never changed).
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
CAVEAT_6 = "Every six-link result carries VISION Update 24's caveat: the reproduction gate is open."
CAVEAT_8 = ("Every eight-link result carries VISION Update 24's caveat: no published eight-link curve exists and the "
            "kernel is validated at four and six links.")


def write(name, cfg):
    path = R / "configs" / (name + ".json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return name


def t37():
    lines = []
    # (g, L, replicas, jobs): each job runs a contiguous block of replica ids of one seeded set
    plan = [(1.50, 64, 40, 1), (1.50, 128, 40, 1), (1.50, 256, 40, 2), (1.50, 512, 20, 2), (1.50, 1024, 24, 8),
            (1.75, 64, 40, 1), (1.75, 128, 40, 1), (1.75, 256, 40, 2), (1.75, 512, 20, 2), (1.75, 1024, 24, 8),
            (1.25, 1024, 12, 6)]
    for g, length, reps, jobs in plan:
        per = reps // jobs
        for j in range(jobs):
            ids = list(range(j * per, (j + 1) * per))
            suffix = ("_" + "abcdefgh"[j]) if jobs > 1 else ""
            name = "t37_lam125_g%03d_L%d%s" % (round(g * 100), length, suffix)
            cfg = {
                "_purpose": ("PREREGISTRATION.md T37 (written 2026-09-25, before any run): many natural seeds in a long "
                             "tube; 4 x %d at lambda 1.25, g %.2f; replicas %d to %d of one seeded set. Not exploratory."
                             % (length, g, ids[0], ids[-1])),
                "name": name, "sides": [[length, 4]], "g": g, "replica_ids": ids, "replicas": reps,
                "n_sweeps": 600000, "block": 5, "stop_at": 0.97, "settle": 600, "settle_max": 2400,
                "seed": 20263700 + round(g * 100), "lambda": 1.25, "save_adjacency": True,
                "count_patches_every": 100, "patch_min": 8,
            }
            lines.append("run_tube_decay " + write(name, cfg))
    return lines


def t38():
    lines = []
    # (lambda, lx, decays, jobs)
    plan = [(1.25, 16, 4000, 16), (1.30, 16, 4000, 16), (1.25, 48, 1000, 8), (1.30, 48, 1000, 8)]
    for lam, lx, reps, jobs in plan:
        per = reps // jobs
        for j in range(jobs):
            ids = list(range(j * per, (j + 1) * per))
            name = "t38_lam%03d_n%d_%02d" % (round(lam * 100), lx * 4, j)
            cfg = {
                "_purpose": ("PREREGISTRATION.md T38 (written 2026-09-25, before any run): the rare long wait; T24's "
                             "protocol at lambda %.2f, N = %d, decays %d to %d of one seeded set, tubes still waiting at "
                             "5,000, 10,000 and 20,000 sweeps saved. Not exploratory." % (lam, lx * 4, ids[0], ids[-1])),
                "name": name, "sides": [[lx, 4]], "g": 1.5, "replica_ids": ids, "replicas": reps,
                "n_sweeps": 100000, "block": 5, "stop_at": 0.98, "settle": 600, "settle_max": 100000,
                "seed": 20263800 + round(lam * 100), "lambda": lam, "save_adjacency": True, "record_f_200": True,
                "save_waiting_at": [5000, 10000, 20000],
            }
            lines.append("run_tube_decay " + write(name, cfg))
    return lines


def t39():
    lines = []
    cells = [  # (tag, dims, lambda, spark, capacities, replicas, n_sweeps, record_every, caveat)
        ("t39_six_lam140_n288_c2n", [4, 4, 18], 1.40, 5.0, ["2*N"], 12, 100000, 200, CAVEAT_6),
        ("t39_six_lam140_n288_cn", [4, 4, 18], 1.40, 5.0, ["N"], 12, 100000, 200, CAVEAT_6),
        ("t39_six_lam140_n288_cn2", [4, 4, 18], 1.40, 5.0, ["N/2"], 12, 100000, 200, CAVEAT_6),
        ("t39_six_lam140_n288_cn4", [4, 4, 18], 1.40, 5.0, ["N/4"], 12, 100000, 200, CAVEAT_6),
        ("t39_six_lam140_n288_cn8", [4, 4, 18], 1.40, 5.0, ["N/8"], 12, 100000, 200, CAVEAT_6),
        ("t39_six_lam140_n288_cn16", [4, 4, 18], 1.40, 5.0, ["N/16"], 12, 100000, 200, CAVEAT_6),
        ("t39_six_lam125_n288_cn3", [4, 4, 18], 1.25, 16.0, ["N/3"], 12, 300000, 500, CAVEAT_6),
        ("t39_six_lam125_n288_cn6", [4, 4, 18], 1.25, 16.0, ["N/6"], 12, 300000, 500, CAVEAT_6),
        ("t39_six_lam125_n288_cn16", [4, 4, 18], 1.25, 16.0, ["N/16"], 12, 300000, 500, CAVEAT_6),
        ("t39_eight_lam125_n1024_cn4", [4, 4, 8, 8], 1.25, 40.0, ["N/4"], 6, 50000, 250, CAVEAT_8),
        ("t39_eight_lam125_n1024_cn8", [4, 4, 8, 8], 1.25, 40.0, ["N/8"], 6, 50000, 250, CAVEAT_8),
        ("t39_eight_lam125_n1024_cn16", [4, 4, 8, 8], 1.25, 40.0, ["N/16"], 6, 50000, 250, CAVEAT_8),
        ("t39_eight_lam150_n1024_cn2", [4, 4, 8, 8], 1.50, 16.0, ["N/2"], 6, 50000, 250, CAVEAT_8),
        ("t39_eight_lam150_n1024_cn3", [4, 4, 8, 8], 1.50, 16.0, ["N/3"], 6, 50000, 250, CAVEAT_8),
        ("t39_eight_lam150_n1024_cn6", [4, 4, 8, 8], 1.50, 16.0, ["N/6"], 6, 50000, 250, CAVEAT_8),
    ]
    for tag, dims, lam, spark, caps, reps, sweeps, every, caveat in cells:
        cfg = {
            "_purpose": ("PREREGISTRATION.md T39 (written 2026-09-25, before any run): the cascade window; a torus with "
                         "two directions curled, %s, lambda %.2f, spark %.1f (the exact wall), C = %s: does the first "
                         "release pay the second wall without melting flat space? Not exploratory. %s"
                         % (" x ".join(str(d) for d in dims), lam, spark, caps[0], caveat)),
            "name": tag, "section": "T39", "dims": dims, "lambda": lam, "capacities": caps, "sparks": [spark],
            "replicas": reps, "n_sweeps": sweeps, "record_every": every, "seed": 20263900 + len(lines),
            "save_adjacency": True,
        }
        lines.append("run_sealed_curled_d " + write(tag, cfg))
    return lines


def queue(fname, header, lines):
    path = R / "cloud" / "queue" / fname
    if path.exists():
        raise FileExistsError(path)
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    head = ("# Submitted by the run_queue workflow from the commit that carries this file, the configs and the "
            "pre-registration.\n")
    queue("2026-09-25_t37.txt", "# PREREGISTRATION T37: many natural seeds in long tubes (paper 2; piece 5).\n" + head, t37())
    queue("2026-09-25_t38.txt", "# PREREGISTRATION T38: the rare long wait (paper 1; piece 2).\n" + head, t38())
    queue("2026-09-25_t39.txt", "# PREREGISTRATION T39: the cascade window (six and eight links; pieces 11 and 13).\n" + head, t39())
    print("written")
