"""Write the T88 configs (PREREGISTRATION.md section T88). Usage:

    python scripts/make_t88_configs.py              # stage 1: L = 8 and 12
    python scripts/make_t88_configs.py --held-out   # stage 2: L = 16, after the amendment that predicts it

Cells: lambda in {1.20, 1.25}; energy E in {120, 240, 480} placed in the first layer of the rod (8 stores); 4 replicas;
10,000 sweeps read every 100; seeds from 20268801 in the order written here.
"""
import json
import sys
from pathlib import Path

LAMBDAS = (1.20, 1.25)
ENERGIES = (120, 240, 480)
STAGE1 = (8, 12)
HELD_OUT = (16,)
SEED0 = 20268801


def write(held_out=False):
    lengths = HELD_OUT if held_out else STAGE1
    k = 0 if not held_out else len(LAMBDAS) * len(ENERGIES) * len(STAGE1)
    names = []
    for length in lengths:
        for lam in LAMBDAS:
            for e in ENERGIES:
                name = "t88_rod_l%d_lam%d_e%d" % (length, round(lam * 100), e)
                cfg = dict(name=name, section="T88", _purpose="T88, the pole test: a rod fed at one end under the tie",
                           dims=[4, 4, length], **{"lambda": lam}, tie="all_at_the_last", energy=e, fed_layers=1,
                           replicas=4, n_sweeps=10000, record_every=100, seed=SEED0 + k, save_adjacency=True)
                Path("configs", name + ".json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                names.append(name)
                k += 1
    return names


if __name__ == "__main__":
    for n in write("--held-out" in sys.argv):
        print(n)
