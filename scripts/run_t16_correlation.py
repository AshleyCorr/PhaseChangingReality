"""T16: the correlation length of [KTB19] Fig. 9a, at the model author's request. Usage:

    python scripts/run_t16_correlation.py configs/t16_<name>.json REPLICA

Runs one replica (0, 1, ...) of the config, so replicas can run side by side, and writes
results/<name>_rep<REPLICA>.csv, .meta.json and results/<name>_rep<REPLICA>_curves/*.npz.
A replica's seeds depend only on (seed, lx, ly, replica), so replica k run alone is replica k.

Two protocols, named by the config's "protocol":
  "tempering"   parallel tempering from melted tori, as T13 protocol E, snapshotting the graph at
                every coupling every `snapshot_every` rounds.
  "sequential"  the author's procedure, as T13 protocol P: cold descent from a melted torus, then
                cold ascent from a fresh lattice torus, each coupling starting from the previous
                one's final graph; `n_equil` sweeps of warm-up, then `n_meas` measured sweeps in
                blocks of `snapshot_every`, one snapshot after each block. The random stream is
                seeded once and carried on (seed -1), never re-seeded per block (ASSUMPTIONS Q14).

What each row holds (PREREGISTRATION T16; the definitions are in src/graphity/correlation.py and
ASSUMPTIONS Q19): phi and the susceptibility chi = N var(phi) from every sweep; from the
snapshots, the mean and standard error of xi / diameter by [KTB19] Eq. (4.15), how many snapshots
had no fluctuations (a perfect lattice) or no usable distance, and xi / diameter computed once
from the averaged C(r). The averaged C(r) itself is stored per row in the _curves directory.
"""
import json
import platform
import sys
import warnings
from pathlib import Path

import numba
import numpy as np

from graphity import __version__
from graphity.analysis import autocorr_time
from graphity.correlation import correlation_by_distance, correlation_curve, xi_literal
from graphity.cqg import NO_CAP, is_valid, run_chain, torus
from graphity.results import ResultWriter
from graphity.tempering import temper

MAX_R = 64
# Distances beyond a graph's diameter are NaN in every snapshot; their mean is NaN, as it should be.
warnings.filterwarnings("ignore", "Mean of empty slice")


class Snapshots:
    """Accumulates the correlation readings of one coupling."""

    def __init__(self):
        self.curves, self.xi, self.no_var, self.no_xi, self.diam = [], [], 0, 0, []

    def add(self, adj):
        num, count, edge_var, _, diam = correlation_by_distance(adj, MAX_R)
        self.diam.append(diam)
        if edge_var <= 0:
            self.no_var += 1
            return
        c = correlation_curve(num, count, edge_var)
        self.curves.append(c)
        xi, used, _ = xi_literal(c, diam)
        if used:
            self.xi.append(xi / diam)
        else:
            self.no_xi += 1

    def row(self):
        xi = np.array(self.xi)
        out = dict(snapshots=len(self.diam), snap_no_fluct=self.no_var, snap_no_xi=self.no_xi,
                   diam_mean=float(np.mean(self.diam)),
                   xi_over_diam=(float(xi.mean()) if xi.size else ""),
                   xi_over_diam_err=(float(xi.std(ddof=1) / np.sqrt(xi.size)) if xi.size > 1 else ""),
                   xi_pooled_over_diam="")
        if self.curves:
            mean_c = np.nanmean(np.array(self.curves), axis=0)
            diam = int(round(np.mean(self.diam)))
            xi_p, used, _ = xi_literal(mean_c, diam)
            if used:
                out["xi_pooled_over_diam"] = float(xi_p / diam)
        return out

    def save(self, path, **extra):
        mean_c = np.nanmean(np.array(self.curves), axis=0) if self.curves else np.full(MAX_R + 1, np.nan)
        np.savez_compressed(path, c_mean=mean_c, xi_over_diam=np.array(self.xi),
                            diam=np.array(self.diam), **extra)


def shape_of(entry):
    return (entry, entry) if isinstance(entry, int) else (int(entry[0]), int(entry[1]))


def run_tempering(cfg, n, lx, ly, rep, side_u, couplings, lam, out, store):
    melt_seeds, temper_seeds = np.random.SeedSequence([cfg["seed"], lx, ly, rep]).spawn(2)
    graphs = []
    for melt_seed in melt_seeds.generate_state(len(couplings)):
        adj, _ = torus(lx, ly, NO_CAP)
        run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, int(melt_seed), lam, NO_CAP, False)
        graphs.append(adj)
    snaps = [Snapshots() for _ in couplings]
    every = cfg["snapshot_every"]

    def on_round(r, gs):
        if (r - cfg["rounds_equil"]) % every == every - 1:
            for k, g in enumerate(gs):
                snaps[k].add(g)

    res = temper(graphs, side_u, couplings, cfg["rounds_equil"] + cfg["rounds_meas"],
                 cfg["sweeps_per_round"], temper_seeds, lam, NO_CAP, False,
                 measure_from=cfg["rounds_equil"], on_round=on_round)
    assert all(is_valid(g, NO_CAP) for g in graphs)
    for k, g in enumerate(couplings):
        s = res["squares"][k].astype(np.int64)
        snaps[k].save(store / ("N%d_k%d.npz" % (n, k)), g=g, N=n)
        out.write(dict(N=n, replica=rep, protocol="tempering", leg="", k=k, g=g, lam=lam,
                       phi=float(s.mean() / n), chi=float(s.var() / n), tau_int=autocorr_time(s / n),
                       acceptance=float(res["acceptance"][k]),
                       swap_rate=(float(res["swap_rate"][k]) if k < len(couplings) - 1 else ""),
                       round_trips=int(res["round_trips"]), **snaps[k].row()))
    print("N=%d rep=%d trips=%d" % (n, rep, res["round_trips"]), flush=True)


def run_sequential(cfg, n, lx, ly, rep, side_u, couplings, lam, out, store):
    seed = int(np.random.SeedSequence([cfg["seed"], lx, ly, rep]).generate_state(1)[0])
    first = True
    block = cfg["snapshot_every"]
    for leg in ("cool", "heat"):
        adj, _ = torus(lx, ly, NO_CAP)
        order = couplings if leg == "cool" else couplings[::-1]
        if leg == "cool":
            run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, seed, lam, NO_CAP, False)
            first = False
        for k, g in enumerate(order):
            run_chain(adj, side_u, 1.0 / g, cfg["n_equil"], 0, seed if first else -1, lam, NO_CAP, False)
            first = False
            snaps = Snapshots()
            s_all, acc_sum = [], 0.0
            for _ in range(cfg["n_meas"] // block):
                s, _, acc = run_chain(adj, side_u, 1.0 / g, 0, block, -1, lam, NO_CAP, False)
                s_all.append(s)
                acc_sum += acc
                snaps.add(adj)
            assert is_valid(adj, NO_CAP)
            s = np.concatenate(s_all).astype(np.int64)
            snaps.save(store / ("N%d_%s_k%d.npz" % (n, leg, k)), g=g, N=n)
            out.write(dict(N=n, replica=rep, protocol="sequential", leg=leg, k=k, g=g, lam=lam,
                           phi=float(s.mean() / n), chi=float(s.var() / n), tau_int=autocorr_time(s / n),
                           acceptance=acc_sum / len(s_all), swap_rate="", round_trips="",
                           **snaps.row()))
        print("N=%d rep=%d leg=%s done" % (n, rep, leg), flush=True)


def main(path, replica, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    if replica not in range(cfg["replicas"]):
        raise ValueError("replica must be one of 0 .. %d" % (cfg["replicas"] - 1))
    lam = float(cfg.get("lambda", 1.0))
    if cfg.get("cap", None) is not None or cfg.get("acceptance", "metropolis") != "metropolis":
        raise ValueError("T16 runs the author's model only: no cap, Metropolis")
    couplings = [float(g) for g in cfg["couplings"]]
    if any(a <= b for a, b in zip(couplings, couplings[1:])):
        raise ValueError("couplings must be strictly decreasing (hottest first)")
    name = "%s_rep%d" % (cfg["name"], replica)
    meta = dict(config=cfg, config_path=str(path), replica=replica, package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T16")
    store = Path(out_dir) / (name + "_curves")
    with ResultWriter(name, meta, out_dir) as out:
        store.mkdir(parents=True, exist_ok=True)
        for entry in cfg["sides"]:
            lx, ly = shape_of(entry)
            _, part = torus(lx, ly, NO_CAP)
            side_u = np.flatnonzero(part == 0)
            run = {"tempering": run_tempering, "sequential": run_sequential}[cfg["protocol"]]
            run(cfg, lx * ly, lx, ly, replica, side_u, couplings, lam, out, store)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "results")
