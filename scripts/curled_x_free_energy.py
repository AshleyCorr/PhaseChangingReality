"""T92: does a curled X pay an entropy penalty against flat space at λ = 1? Free energy by thermodynamic integration.

Usage:
    python scripts/curled_x_free_energy.py <stage>        stage is "stage1" (N = 144) or "heldout" (N = 192)

THE QUESTION (the owner's, 2 November: "Whether a curled X charges the same penalty has not been measured. Do this."):
in the published model at λ = 1, where a curled arrangement and a flat one have exactly the same energy, does the curled
one have lower or higher free energy once the warmth g is above zero, that is, does entropy favor opening or curling?
T78 and T79 (O147, O148) measured a penalty on a curl inside a warm melted network; O150 found, counting one switch out,
that the curled tube has the lowest free energy of the lattice tori at every coupling. This measures it beyond one switch.

THE METHOD (standard thermodynamic integration; exact identity, numerical integral). For a basin of states,
d(βF)/dβ = <H>, with β = 1/g. Both basins have ground energy 0 at λ = 1, so

    βF(β) = −ln D − ∫_β^β_max <H>(β') dβ'      (β_max = 1/g_min; <H> is essentially zero beyond it)

with D the number of labeled ground states in the basin. Then

    ΔF(g) = F_curled − F_flat = g [ −ln(D_curled / D_flat) − ∫_β^β_max (<H>_curled − <H>_flat) dβ' ].

ΔF > 0: open space has the lower free energy (entropy favors opening). ΔF < 0: the curl is favored.
The degeneracy term is exact (O150): the tube's zero-cost family holds 4 labeled copies per tube labeling and the tube has
2N renamings; a square lattice torus has 4N, a non-square one 2N. So ln(D_curled / D_flat) = ln 8 against a square flat
torus and ln 4 against a non-square one. The integral is measured.

THE RUN. Named points, λ = 1, no cap, the hard-core rule, Metropolis (graphity.cqg.run_chain). For each basin and each
coupling g in COUPLINGS, R replicas each start from the built torus (fresh), run n_equil sweeps, then n_meas measured
sweeps; <H> = 16 (N − S) + 4 X averaged over the measured sweeps; X / N and S / N are kept to check the basin held
(curled: X / N ≥ 0.8 throughout the mean; flat: X / N ≤ 0.2). A replica whose mean leaves its basin is flagged and its
coupling is the end of that basin's window. Stage 1: the 4 × 36 tube against the 12 × 12 flat torus (N = 144). Held out:
the 4 × 48 tube against the 12 × 16 flat torus (N = 192).
Pure functions, tested in tests/test_curled_x_free_energy.py.
"""
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                       # noqa: E402
from graphity.cqg import NO_CAP, run_chain, torus      # noqa: E402

COUPLINGS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5)
STAGES = {"stage1": dict(curled=(4, 36), flat=(12, 12), ln_ratio=math.log(8)),
          "heldout": dict(curled=(4, 48), flat=(12, 16), ln_ratio=math.log(4))}
REPLICAS, N_EQUIL, N_MEAS, SEED = 4, 2000, 10000, 20269201


def energy(n, s, x):
    return 16.0 * (n - s) + 4.0 * x


def measure(shape, g, rep):
    """<H>, <S>/N, <X>/N for one replica at coupling g, from a fresh built torus."""
    adj, part = torus(shape[0], shape[1], NO_CAP)
    n = len(adj)
    side_u = np.flatnonzero(part == 0)
    seed = int(np.random.SeedSequence([SEED, shape[0], shape[1], int(round(g * 1000)), rep]).generate_state(1)[0])
    s, x, acc = run_chain(adj, side_u, 1.0 / g, N_EQUIL, N_MEAS, seed, 1.0, NO_CAP, False)
    s = np.asarray(s, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.mean(energy(n, s, x))), float(np.mean(s) / n), float(np.mean(x) / n), float(acc)


def delta_f(gs, h_curled, h_flat, ln_ratio):
    """ΔF(g) at every g in `gs` (increasing), by the trapezoid rule in β from the coldest coupling."""
    beta = 1.0 / np.asarray(gs, dtype=float)
    dh = np.asarray(h_curled, dtype=float) - np.asarray(h_flat, dtype=float)
    out = []
    for k in range(len(gs)):
        # integral from beta[k] up to beta[0] (the coldest), beta decreasing with k
        integral = 0.0
        for i in range(k):
            integral += 0.5 * (dh[i] + dh[i + 1]) * (beta[i] - beta[i + 1])
        out.append(gs[k] * (-ln_ratio - integral))
    return out


def main(stage, out_dir="results"):
    cfg = STAGES[stage]
    name = "t92_curled_x_free_energy_" + stage
    path = Path(out_dir) / (name + ".csv")
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    rows = []
    for basin in ("curled", "flat"):
        shape = cfg[basin]
        for g in COUPLINGS:
            for rep in range(REPLICAS):
                h, s, x, acc = measure(shape, g, rep)
                rows.append(dict(stage=stage, basin=basin, lx=shape[0], ly=shape[1], N=shape[0] * shape[1], g=g,
                                 replica=rep, h_mean=h, s_per_n=s, x_per_n=x, acceptance=acc))
            print(stage, basin, g, "done", flush=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(script="scripts/curled_x_free_energy.py", stage=stage, couplings=COUPLINGS, replicas=REPLICAS,
                n_equil=N_EQUIL, n_meas=N_MEAS, seed=SEED, shapes=dict(curled=cfg["curled"], flat=cfg["flat"]),
                ln_degeneracy_ratio=cfg["ln_ratio"], package=__version__, python=platform.python_version(),
                numpy=np.__version__, preregistration="PREREGISTRATION.md section T92")
    (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
