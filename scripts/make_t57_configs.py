"""Write T57's configs (PREREGISTRATION T57, written 2026-10-09 before any run): T34's protocol, flat six-link space
given concentrated energy, sealed, under the owner's tie "all at the last", with untied controls, at energies up to
twice what curling one direction of the whole space would cost under the tie. Usage:

    python scripts/make_t57_configs.py          # prints the queue lines; refuses to rewrite an existing config

Cells: tie in {all_at_the_last, none} x lambda in {1.10, 1.25} x protocol in {packed (local_heat), spread (a shared
bath of 2N stores)} x N in {216 (6 x 6 x 6), 512 (8 x 8 x 8)}: 16 configs, each with five energies and eight
replicas, 50,000 sweeps read every 250, final graphs saved. Energies: at N = 512, 256, 560, 1060 (T34's three
largest), 1600 (about 3a N at lambda = 1.25, one direction of the whole space under the tie) and 3200; at N = 216,
128, 260, 500, 680 and 1360. Seeds from 20265701 in the order written.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20265701
TIES = (("all_at_the_last", [0, 1.0, 2.0, 0]), ("none", None))
LAMBDAS = (1.10, 1.25)
PROTOCOLS = ("packed", "spread")
SIZES = ((216, [6, 6, 6], [128.0, 260.0, 500.0, 680.0, 1360.0]), (512, [8, 8, 8], [256.0, 560.0, 1060.0, 1600.0, 3200.0]))


def config(name, tie, table, lam, protocol, dims, sparks, seed):
    cfg = {"_purpose": ("PREREGISTRATION.md T57 (written 2026-10-09, before any run; VISION Update 51): does too much "
                        "concentrated energy curl flat six-link space under the owner's tie 'all at the last'? T34's "
                        "protocol: flat %s, sealed, the energy given at the start and conserved; %s; lambda %g; %s. "
                        "Not exploratory. Every six-link result carries VISION Update 24's caveat."
                        % (" x ".join(map(str, dims)),
                           "packed: the whole energy in one vertex's store under the per-vertex bath" if protocol == "packed"
                           else "spread: a shared bath of 2N stores with the whole energy in one of them",
                           lam, "the tie f = (0, a, 2a, 0), a = 4(lambda - 1)" if tie != "none" else "no tie (the control)")),
           "name": name, "section": "T57", "dims": dims, "lambda": lam}
    if protocol == "packed":
        cfg["local_heat"] = True
        cfg["capacities"] = ["N"]
    else:
        cfg["capacities"] = ["2*N"]
    if table is not None:
        cfg["ftable_per_a"] = table
        cfg["tie_label"] = "all at the last"
    cfg.update(sparks=sparks, replicas=8, n_sweeps=50000, record_every=250, seed=seed, save_adjacency=True)
    return cfg


def main():
    lines, i = [], 0
    for tie, table in TIES:
        for lam in LAMBDAS:
            for protocol in PROTOCOLS:
                for n, dims, sparks in SIZES:
                    name = "t57_%s_%s_lam%d_n%d" % ("tied" if tie != "none" else "untied", protocol, round(lam * 100), n)
                    path = ROOT / "configs" / (name + ".json")
                    if path.exists():
                        raise FileExistsError("%s exists; configs are not rewritten" % path)
                    path.write_text(json.dumps(config(name, tie, table, lam, protocol, dims, sparks, SEED0 + i), indent=2) + "\n",
                                    encoding="utf-8")
                    lines.append("run_sealed_curled_d " + name)
                    i += 1
    print("\n".join(lines))
    print("%d configs written" % i, file=sys.stderr)


if __name__ == "__main__":
    main()
