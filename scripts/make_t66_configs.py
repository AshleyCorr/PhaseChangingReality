"""Write T66's configs (PREREGISTRATION T66, written 2026-10-16 before any registered run): does flat three-direction
space heal damage in fewer sweeps than flat two-direction space? Usage:

    python scripts/make_t66_configs.py              # stage 1 (N = 216 and 512); prints the queue lines
    python scripts/make_t66_configs.py --held-out   # N = 1000, after the smaller sizes are read (rule 14)

Refuses to rewrite an existing config. Cells: two directions (12 x 18, 16 x 32; held out 20 x 50) and three
(6 x 6 x 6, 8 x 8 x 8; held out 10 x 10 x 10), lambda 1.25, untied, E = rho N with rho = 1 and 2 packed into one
store of a bath of 4N, 16 replicas, 8,000 sweeps read every 10. Seeds from 20266601 in the order written.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20266601
LAM = 1.25
RHO = [1.0, 2.0]
SIZES = {216: ([12, 18], [6, 6, 6]), 512: ([16, 32], [8, 8, 8]), 1000: ([20, 50], [10, 10, 10])}
PURPOSE = ("PREREGISTRATION.md T66 (written 2026-10-16, before any registered run; the owner's prediction of the same "
           "day): does flat three-direction space heal damage in fewer sweeps than flat two-direction space? Flat %s "
           "torus (%d directions, N = %d), untied, sealed, E = rho N packed into one store of a bath of 4N empty stores, "
           "rho = 1 and 2. Not exploratory. Every six-link result carries VISION Update 24's caveat.")


def cells(held_out=False):
    sizes = (1000,) if held_out else (216, 512)
    out = []
    for n in sizes:
        for dims in SIZES[n]:
            out.append(("t66_d%d_n%d" % (len(dims), n), dims))
    return out


def config(name, dims, seed):
    n = 1
    for x in dims:
        n *= x
    return dict(_purpose=PURPOSE % ("x".join(str(x) for x in dims), len(dims), n), name=name, section="T66",
                dims=dims, **{"lambda": LAM}, rho_per_point=RHO, capacity="4*N", replicas=16, n_sweeps=8000,
                record_every=10, seed=seed)


def main(held_out=False):
    offset = 100 if held_out else 0
    lines = []
    for i, (name, dims) in enumerate(cells(held_out)):
        path = ROOT / "configs" / (name + ".json")
        if path.exists():
            raise FileExistsError("%s exists; configs are not rewritten" % path)
        path.write_text(json.dumps(config(name, dims, SEED0 + offset + i), indent=2) + "\n")
        lines.append("run_heal_time %s" % name)
    print("\n".join(lines))


if __name__ == "__main__":
    main("--held-out" in sys.argv[1:])
