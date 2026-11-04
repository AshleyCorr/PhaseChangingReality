"""T11: is the leftover ring a seam? Usage:

    python scripts/run_seam_check.py configs/t11_seam_n64.json

Implements PREREGISTRATION.md section T11. A cold sealed box exactly as T10 (a bath of 2N demons,
one holding the 12-unit spark), run in short blocks so that the first departure from the tube is
caught. Per replica it records where along the tube the change started (the columns of the first
non-tube vertices), where the leftover ring sits at the end (the columns of the vertices at d = 1),
and the circular distance between them. Only replicas that end with exactly one ring carry a
distance. Nothing is stopped before 90 % conversion plus a settling stretch.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                        # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus, total_squares      # noqa: E402
from graphity.dimension import local_dimension, pieces_of               # noqa: E402
from graphity.results import ResultWriter                               # noqa: E402
from graphity.sealed import run_sealed_bath                             # noqa: E402

PHI_TUBE, PHI_SHEET = 1.25, 1.0


def circular_mean(cols, lx):
    """Mean position of a set of columns on a loop of length lx, in columns."""
    ang = 2.0 * np.pi * np.asarray(cols, float) / lx
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m * lx / (2.0 * np.pi)) % lx)


def circular_distance(a, b, lx):
    d = abs(float(a) - float(b)) % lx
    return float(min(d, lx - d))


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, spark = float(cfg["lambda"]), float(cfg["spark"])
    block, n_sweeps, settle = int(cfg["block"]), int(cfg["n_sweeps"]), int(cfg["settle_after"])
    lx, ly = cfg["side"]
    n = lx * ly
    c = max(1, int(round(eval(str(cfg["capacity"]), {"N": n}))))
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T11, written 2026-09-22")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(int(cfg["replicas"])):
            adj, part = torus(lx, ly, NO_CAP)
            side_u = np.flatnonzero(part == 0)
            demons = np.zeros(c)
            demons[0] = spark
            e0 = hamiltonian(adj, lam) + demons.sum()
            s_tube = total_squares(adj)
            done, blk, depart_sweep, converted_at, start_cols, last = 0, 0, -1, -1, [], None
            while done < n_sweeps:
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, rep, blk]).generate_state(1)[0])
                s, x, mean, tot, acc = run_sealed_bath(adj, side_u, demons, block, seed, lam, NO_CAP)
                done += block
                blk += 1
                last = (int(s[-1]), int(x[-1]), float(mean[-1]), float(tot[-1]))
                if depart_sweep < 0 and s[-1] < s_tube:
                    depart_sweep = done
                    d = local_dimension(adj)
                    start_cols = sorted({int(v // ly) for v in np.flatnonzero(d != 1)})
                f = (PHI_TUBE - s[-1] / n) / (PHI_TUBE - PHI_SHEET)
                if converted_at < 0 and f >= 0.9:
                    converted_at = done
                if converted_at >= 0 and done >= converted_at + settle:
                    break
            s_end, x_end, bath_t, tot_end = last
            drift = float(abs(16.0 * (n - s_end) + 4.0 * lam * x_end + tot_end - e0))
            d = local_dimension(adj)
            ring_cols = sorted({int(v // ly) for v in np.flatnonzero(d == 1)})
            pieces = pieces_of(adj, d == 1)
            rings = len(pieces)
            start = circular_mean(start_cols, lx) if start_cols else float("nan")
            one_ring = rings == 1 and len(ring_cols) == 1 and bool(start_cols)
            dist = circular_distance(ring_cols[0], start, lx) if one_ring else float("nan")
            out.write(dict(N=n, C=c, replica=rep, depart_sweep=depart_sweep, n_start=len(start_cols),
                           start_cols=" ".join(map(str, start_cols)), start=start,
                           converted_at=converted_at, end_sweep=done, rings=rings,
                           largest_ring_piece=(pieces[0] if pieces else 0),
                           ring_cols=" ".join(map(str, ring_cols)), dist=dist, half_length=lx / 2,
                           f_final=(PHI_TUBE - s_end / n) / (PHI_TUBE - PHI_SHEET),
                           bath_T=bath_t, drift=drift))
            print("N=%-4d rep=%-2d  start %s (%.1f)  converted at %d  rings %d at cols %s  dist %.1f of %d  drift %.1e"
                  % (n, rep, start_cols, start, converted_at, rings, ring_cols, dist, lx // 2, drift), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
