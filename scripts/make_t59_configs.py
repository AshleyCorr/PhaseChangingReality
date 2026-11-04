"""Write the configs of PREREGISTRATION T59 (the long waits at lambda 1.05 read from their seeds; fresh first exits
timed by the move at lambda 1.05 and 1.30). Usage:

    python scripts/make_t59_configs.py

  t59_t8wait_lam105_n96      stage A1: T8's decays 9, 10 and 11 at N = 96 replayed from T8's seed and written block
                             by block (scripts/run_tube_decay.py; decay 11 waited 78,205 sweeps)
  t59_offers_lam105          stage A2: T22's seven first exits later than four mean waits, and the replica after
                             each, replayed from T22's seeds with what the chain was offered tallied
  t59_lam105_n64_00 to _15   stage B: 2,000 fresh tubes at lambda 1.05, N = 64
  t59_lam105_n96_00 to _19   stage B: 1,000 fresh tubes at lambda 1.05, N = 96
  t59_lam130_n64_00 to _15   stage C: 4,000 fresh tubes at lambda 1.30, N = 64

Named points throughout, as in T8, T22, T38 and T58. Refuses to overwrite a config.
"""
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEED = 20265900                                   # fresh; T22 used 20261801 and 20261802, T38 and T58 20263930
G = 1.5
LOOK = 5                                          # T7's, T8's, T38's and T58's block: one look every five sweeps
T8_KEYS = ("g", "replicas", "n_sweeps", "block", "stop_at", "settle", "settle_max", "seed", "lambda", "record_f_200")
T8_TARGET, T8_CONTROLS = 11, [9, 10]              # the two decays before the target, fixed before the run
# T22's first exits later than four mean waits at lambda 1.05 (targets), and the replica after each (controls;
# where that is itself a target, the one before): fixed before the run. Seeds are T22's own.
T22_SEED = {64: 20261801, 96: 20261802}
T22_TARGETS = {64: [6, 7, 11, 17], 96: [6, 12, 29]}
T22_CONTROLS = {64: [5, 8, 12, 18], 96: [7, 13, 30]}
CELLS = (                                         # (stage, lambda, side, tubes, files, cap in sweeps)
    ("B", 1.05, [16, 4], 2000, 16, 200000),
    ("B", 1.05, [24, 4], 1000, 20, 200000),
    ("C", 1.30, [16, 4], 4000, 16, 50000),
)


def t8_wait_config():
    old = json.loads((CONFIGS / "t8_lam105.json").read_text())
    ids = sorted([T8_TARGET] + T8_CONTROLS)
    new = {
        "_purpose": ("PREREGISTRATION.md T59, stage A1 (written 2026-10-10, before any run): T8's decay 11 at lambda "
                     "1.05, N = 96, which waited 78,205 sweeps, and the two decays before it, replayed from T8's seed "
                     "and written block by block with the detector's numbers. Named points, as in the original. "
                     "Not exploratory."),
        "name": "t59_t8wait_lam105_n96",
        "sides": [[24, 4]],
    }
    for key in T8_KEYS:
        new[key] = old[key]
    new["replica_ids"] = ids
    new["record_detector"] = True
    new["trace_replicas"] = ids
    return new


def offers_config():
    targets = []
    for n, side in ((64, [16, 4]), (96, [24, 4])):
        for role, reps in (("target", T22_TARGETS[n]), ("control", T22_CONTROLS[n])):
            targets += [dict(side=side, seed=T22_SEED[n], replica=rep, role=role) for rep in reps]
    return {
        "_purpose": ("PREREGISTRATION.md T59, stage A2 (written 2026-10-10, before any run): T22's seven first exits "
                     "later than four mean waits at lambda 1.05 and the replica after each, replayed from T22's "
                     "seeds and stopped at the first exit, with the exits the chain was offered tallied in stretches "
                     "of 5,000 sweeps. Named points, as in the original. Not exploratory."),
        "name": "t59_offers_lam105",
        "mode": "offers",
        "g": G,
        "lambda": 1.05,
        "max_sweeps": 100000,
        "stretch": 5000,
        "targets": targets,
    }


def clock_configs():
    for stage, lam, side, tubes, files, cap in CELLS:
        n = side[0] * side[1]
        per = tubes // files
        assert per * files == tubes
        for index in range(files):
            yield {
                "_purpose": ("PREREGISTRATION.md T59, stage %s (written 2026-10-10, before any run): fresh tubes %d to "
                             "%d at lambda %.2f, N = %d, each run to the first exit that a look every five sweeps "
                             "sees, with the first exit also counted move by move and by a look every sweep. Named "
                             "points. Not exploratory."
                             % (stage, index * per, (index + 1) * per - 1, lam, n)),
                "name": "t59_lam%d_n%d_%02d" % (round(lam * 100), n, index),
                "mode": "clocks",
                "side": side,
                "g": G,
                "lambda": lam,
                "replica_start": index * per,
                "replicas": per,
                "max_sweeps": cap,
                "look": LOOK,
                "seed": SEED,
            }


def main():
    for cfg in [t8_wait_config(), offers_config()] + list(clock_configs()):
        path = CONFIGS / (cfg["name"] + ".json")
        if path.exists():
            raise FileExistsError("%s exists; configs are not overwritten" % path)
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
