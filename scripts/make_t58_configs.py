"""Write the 17 configs of PREREGISTRATION T58 (T38's cell at lambda 1.30, N = 64 replayed from its seeds, with the
detector's threshold and what it cannot see recorded). Usage:

    python scripts/make_t58_configs.py

Sixteen configs are the T38 originals (configs/t38_lam130_n64_XX.json) with the same seed, sides, coupling, replica
ids, block and caps, so that every decay draws the random stream it drew the first time. What changes reads the
chain and draws nothing: `record_detector` adds the detector's numbers to each row, and the saved graphs of the
original (`save_adjacency`, `save_waiting_at`) are not written again. The seventeenth, t58_targets_lam130_n64, is
the three long waits and six neighbours on their own, written block by block (`trace_replicas`), so that the
question about the three can be read in seconds while the cell runs. Named points, as in the original. Refuses to
overwrite a config.
"""
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FILES = 16
TARGETS = [553, 2748, 3072]                       # the three waits ASSUMPTIONS O109 left unexplained
CONTROLS = [551, 552, 2746, 2747, 3070, 3071]     # the two replica ids before each target, fixed before the run


KEYS = ("sides", "g", "replica_ids", "replicas", "n_sweeps", "block", "stop_at", "settle", "settle_max", "seed",
        "lambda", "record_f_200")


def config(index):
    old = json.loads((CONFIGS / ("t38_lam130_n64_%02d.json" % index)).read_text())
    ids = [int(r) for r in old["replica_ids"]]
    new = {
        "_purpose": ("PREREGISTRATION.md T58 (written 2026-10-10, before any run): T38's decays %d to %d at lambda "
                     "1.30, N = 64, replayed from their seeds with the detector's threshold, the highest square count "
                     "and the points with both directions curled recorded. Named points, as in the original. "
                     "Not exploratory." % (ids[0], ids[-1])),
        "name": "t58_lam130_n64_%02d" % index,
    }
    for key in KEYS:
        new[key] = old[key]
    new["record_detector"] = True
    return new


def targets_config():
    old = json.loads((CONFIGS / "t38_lam130_n64_00.json").read_text())
    ids = sorted(TARGETS + CONTROLS)
    new = {
        "_purpose": ("PREREGISTRATION.md T58 (written 2026-10-10, before any run): the three long waits of T38 at "
                     "lambda 1.30, N = 64 (decays %s) and the two decays before each, replayed from their seeds and "
                     "written block by block: squares, surplus squares, the count of points at each d, and the "
                     "connected pieces. Named points, as in the original. Not exploratory."
                     % ", ".join(str(r) for r in TARGETS)),
        "name": "t58_targets_lam130_n64",
    }
    for key in KEYS:
        new[key] = old[key]
    new["replica_ids"] = ids
    new["record_detector"] = True
    new["trace_replicas"] = ids
    return new


def main():
    for cfg in [targets_config()] + [config(index) for index in range(FILES)]:
        path = CONFIGS / (cfg["name"] + ".json")
        if path.exists():
            raise FileExistsError("%s exists; configs are not overwritten" % path)
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
