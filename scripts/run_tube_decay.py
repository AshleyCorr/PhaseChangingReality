"""T7: watch a tube uncurl into a sheet, and look at it while it happens. Usage:

    python scripts/run_tube_decay.py configs/t7_lam125.json

Implements PREREGISTRATION.md section T7. Each replica starts from the exact tube (a torus with
one side of length 4), runs at fixed coupling, and is followed in short blocks. When the
conversion fraction first crosses 25 %, 50 % and 75 % the adjacency is read -- not copied, the
chain is simply paused -- and three things are recorded: the histogram of the local dimension
d(v) over vertices, the connected pieces of the vertices that already read as sheet (d = 2),
and the sweep. The waiting time is the pre-registered one: the first sweep at which phi has
left the tube's value by more than three times its own resting fluctuation. The energy
released is checked against the exact 4(lambda - 1).

Conversion fraction: f = (phi_tube - phi) / (phi_tube - phi_sheet), with phi_tube = 1.25 (one
curled side adds a quarter of a square per vertex) and phi_sheet = 1.

The chain is advanced with seed < 0 after the first block, which carries the random stream on
rather than re-seeding (ASSUMPTIONS Q14). Looking at the graph between blocks uses no random
numbers, so the chain is the same as if it had run uninterrupted.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                           # noqa: E402
from graphity.connectivity import connectivity                             # noqa: E402
from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP,     # noqa: E402
                          run_chain, surplus, torus, total_squares)
from graphity.dimension import local_dimension, new_seeds, piece_labels, pieces_of  # noqa: E402
from graphity.results import ResultWriter                                  # noqa: E402

PHI_TUBE, PHI_SHEET = 1.25, 1.0
MARKS = (0.25, 0.50, 0.75)
D_BINS = 7                     # d(v) runs 0..6


def snapshot(adj):
    d = local_dimension(adj)
    hist = np.bincount(d, minlength=D_BINS)[:D_BINS]
    pieces = pieces_of(adj, d == 2)
    converted = int((d == 2).sum())
    return hist, len(pieces), (pieces[0] / converted if converted else 0.0), converted


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    g = float(cfg["g"])
    block = int(cfg.get("block", 5))
    n_sweeps = int(cfg["n_sweeps"])
    stop_at = float(cfg.get("stop_at", 0.98))
    settle = int(cfg.get("settle", 300))     # sweeps to average the final energy over
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T7, written 2026-09-22")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for lx, ly in cfg["sides"]:
            n = lx * ly
            # T7 amendment 3 (i): a config may name the replicas to run, to replay chosen decays
            # with a longer settle cap. The seed derivation below does not change, so a replayed
            # replica reproduces its original random stream exactly up to the old cap.
            reps = [int(r) for r in cfg["replica_ids"]] if "replica_ids" in cfg else range(int(cfg["replicas"]))
            for rep in reps:
                adj, part = torus(lx, ly, NO_CAP)
                side_u = np.flatnonzero(part == 0)
                seed = int(np.random.SeedSequence([int(cfg["seed"]), lx, ly, rep]).generate_state(1)[0])
                h0 = (ENERGY_PER_SQUARE * (n - total_squares(adj))
                      + ENERGY_PER_SURPLUS * lam * surplus(adj)) / n
                phis, sweeps_done, first = [], 0, True
                f_200 = ""
                # T37 (2026-09-25): count separate seeds. Every `count_patches_every` sweeps the sheet points
                # (d = 2) are split into connected pieces; a piece of at least `patch_min` points touching no point
                # of a piece counted before is a new seed. Reading only: no random numbers are drawn.
                save_waiting_at = set(int(v) for v in cfg.get("save_waiting_at", []))
                patch_every = int(cfg.get("count_patches_every", 0))
                patch_min = int(cfg.get("patch_min", 8))
                counted = np.zeros(n, dtype=bool)
                seeds, seed_sweeps, patch_series, converted_series = 0, [], [], []
                hit = {}
                waited = None
                thresh = None
                # T58 (2026-10-10): the detector's own numbers, and what it cannot see. With `record_detector` the
                # row gains the threshold and the resting spread it came from, the highest square count reached,
                # the most points with both directions curled (d = 0), and how the stretch between sweep 200 and
                # the detection was spent. A replica named in `trace_replicas` has every block written to a file
                # and its final graph saved. Reading only: no random numbers are drawn.
                detect = bool(cfg.get("record_detector"))
                traced = rep in set(int(r) for r in cfg.get("trace_replicas", []))
                s_tube, x_tube = int(total_squares(adj)), int(surplus(adj))
                rest_sd = ""
                phi_max, sweep_phi_max, d0_max, d0_blocks = PHI_TUBE, 0, 0, 0
                tube_at_200, first_left, first_left_after_200 = "", "", ""
                exits_unseen, off_blocks, deepest_unseen, was_tube = 0, 0, "", True
                trace = []
                # 10 October 2026, the owner's question (could a second curl come and go between two looks?): with
                # `trace_sweeps` a traced replica also has every sweep written, with the connected pieces, baby
                # universes and 4-cubes after it, and the number of moves accepted in its block of sweeps. A block
                # with none accepted is a block in which the graph did not change at all. Reading only.
                sweep_trace = []
                conn = np.zeros((block, 4), dtype=np.int64) if (traced and cfg.get("trace_sweeps")) else None
                while sweeps_done < n_sweeps:
                    if conn is None:
                        s, x, acc = run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1,
                                              lam, NO_CAP, False)
                    else:               # looking after every sweep draws no random numbers: the chain is the same
                        s, x, acc = run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1,
                                              lam, NO_CAP, False, conn)
                    first = False
                    sweeps_done += block
                    if conn is not None:
                        moves = int(round(float(acc) * block * 2 * n))
                        for i in range(block):
                            sweep_trace.append([sweeps_done - block + i + 1, int(s[i]), int(x[i])]
                                               + [int(v) for v in conn[i]] + [moves])
                    phi = float(s[-1]) / n
                    phis.append(phi)
                    if detect or traced:
                        d_hist = np.bincount(local_dimension(adj), minlength=D_BINS)[:D_BINS]
                        is_tube = int(s[-1]) == s_tube and int(x[-1]) == x_tube and int(d_hist[1]) == n
                        if phi > phi_max:
                            phi_max, sweep_phi_max = phi, sweeps_done
                        d0_max = max(d0_max, int(d_hist[0]))
                        d0_blocks += int(d_hist[0] > 0)
                        if first_left == "" and not is_tube:
                            first_left = sweeps_done
                        if sweeps_done == 200:
                            tube_at_200 = int(is_tube)
                        if traced:
                            trace.append([sweeps_done, int(s[-1]), int(x[-1])] + [int(v) for v in d_hist]
                                         + [int(v) for v in connectivity(adj)])
                    if patch_every and sweeps_done % patch_every == 0:
                        sheet = local_dimension(adj) == 2
                        converted_series.append(int(sheet.sum()))
                        labels = piece_labels(adj, sheet)
                        k_new, counted = new_seeds(labels, counted, patch_min)
                        if k_new:
                            seeds += k_new
                            seed_sweeps.append("%d:%d" % (sweeps_done, k_new))
                        sizes = np.bincount(labels[labels >= 0]) if labels.max() >= 0 else np.zeros(0, dtype=np.int64)
                        patch_series.append(int((sizes >= patch_min).sum()))
                    if sweeps_done == 200:
                        # T8 (2026-09-23): how far the tube had converted when its resting stretch
                        # ended, so that "still a tube at sweep 200" can be read. Reading only.
                        f_200 = (PHI_TUBE - phi) / (PHI_TUBE - PHI_SHEET)
                    if sweeps_done <= 200:
                        was_tube = (not (detect or traced)) or is_tube
                        continue
                    if thresh is None:
                        rest = np.array(phis[: max(1, 200 // block)])
                        thresh = PHI_TUBE - 3.0 * max(rest.std(), 1e-6) - 1e-9
                        rest_sd = float(rest.std())
                    if waited is None and phi < thresh:
                        waited = sweeps_done
                    if detect or traced:
                        if first_left_after_200 == "" and not is_tube:
                            first_left_after_200 = sweeps_done
                        if waited is None and not is_tube:       # away from the tube, and the detector silent
                            off_blocks += 1
                            exits_unseen += int(was_tube)
                            deepest_unseen = phi if deepest_unseen == "" else min(deepest_unseen, phi)
                    # T38 (2026-09-25): a tube still waiting at a named sweep has its graph saved, so a long wait can
                    # be read from its wiring. Reading only: no random numbers are drawn.
                    if waited is None and sweeps_done in save_waiting_at:
                        wait_dir = Path(out_dir) / (cfg["name"] + "_waiting")
                        wait_dir.mkdir(parents=True, exist_ok=True)
                        target = wait_dir / ("N%d_rep%d_sweep%d.npz" % (n, rep, sweeps_done))
                        if target.exists():
                            raise FileExistsError("results are append-only; %s exists" % target)
                        np.savez(target, adj=adj, part=part, lam=lam, g=g, h0=h0 * n)
                    f = (PHI_TUBE - phi) / (PHI_TUBE - PHI_SHEET)
                    for m in MARKS:
                        if m not in hit and f >= m:
                            hit[m] = (sweeps_done,) + snapshot(adj)
                    if detect or traced:
                        was_tube = is_tube
                    if f >= stop_at:
                        break
                # T7 amendment 1: settle until the released energy has stopped changing. The
                # first runs settled for a fixed 300 sweeps and caught half the decays on a ledge
                # -- a defected sheet holding 0.22 per point -- which then released the rest in
                # one step hundreds to thousands of sweeps later. So the energy is read in windows
                # of `settle` sweeps and accepted when two consecutive windows agree to 0.5 % of
                # 4(lambda - 1), up to `settle_max` sweeps. The ledge is recorded: the released
                # energy after the first window, and how long the plateau lasted.
                settle_max = int(cfg.get("settle_max", 20000))
                windows, spent = [], 0
                while spent < settle_max:
                    hs = []
                    for _ in range(max(1, settle // block)):
                        s, x, _ = run_chain(adj, side_u, 1.0 / g, 0, block, -1, lam, NO_CAP, False)
                        hs.append((ENERGY_PER_SQUARE * (n - s[-1]) + ENERGY_PER_SURPLUS * lam * x[-1]) / n)
                    spent += max(1, settle // block) * block
                    windows.append(h0 - float(np.mean(hs)))
                    if len(windows) >= 2 and abs(windows[-1] - windows[-2]) <= 0.005 * 4.0 * (lam - 1.0)                             and abs(windows[-1] - 4.0 * (lam - 1.0)) <= 0.01 * 4.0 * (lam - 1.0):
                        break
                sweeps_done += spent
                # T7 amendment 4 (a): keep the resting state, so it can be read from its wiring
                # instead of assumed. Saving reads the graph and uses no random numbers.
                if cfg.get("save_adjacency"):
                    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
                    adj_dir.mkdir(parents=True, exist_ok=True)
                    target = adj_dir / ("N%d_rep%d.npz" % (n, rep))
                    if target.exists():
                        raise FileExistsError("results are append-only; %s exists" % target)
                    np.savez(target, adj=adj, part=part, lam=lam, g=g, h0=h0 * n)
                h1 = h0 - windows[-1]
                first_window = windows[0]
                ledge = sum(1 for w in windows[:-1] if abs(w - windows[-1]) > 0.05 * 4.0 * (lam - 1.0)) * settle
                row = dict(N=n, lx=lx, ly=ly, replica=rep, lam=lam, g=g, seed=seed,
                           sweeps=sweeps_done, waiting=(waited if waited is not None else ""),
                           phi_final=phis[-1], released=(h0 - h1), expected=4.0 * (lam - 1.0),
                           released_first_window=first_window, ledge_sweeps=ledge, settle_sweeps=spent,
                           reached=max([0.0] + [m for m in hit]))
                if cfg.get("record_f_200"):
                    row["f_200"] = f_200                   # T8; absent from T7's files, which predate it
                if detect:                                 # T58; absent from earlier files
                    row.update(thresh=(thresh if thresh is not None else ""), rest_sd=rest_sd,
                               tube_at_200=tube_at_200, first_left=first_left,
                               first_left_after_200=first_left_after_200, exits_unseen=exits_unseen,
                               off_blocks=off_blocks, deepest_unseen=deepest_unseen, phi_max=phi_max,
                               sweep_phi_max=sweep_phi_max, d0_max=d0_max, d0_blocks=d0_blocks)
                if traced:
                    trace_dir = Path(out_dir) / (cfg["name"] + "_trace")
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    target = trace_dir / ("N%d_rep%d.csv" % (n, rep))
                    final = trace_dir / ("N%d_rep%d_final.npz" % (n, rep))
                    if target.exists() or final.exists():
                        raise FileExistsError("results are append-only; %s exists" % target)
                    header = "sweep,S,X," + ",".join("d%d" % k for k in range(D_BINS)) + ",pieces,largest,baby,cubes"
                    lines = [header] + [",".join(str(v) for v in line) for line in trace]
                    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                    np.savez(final, adj=adj, part=part, lam=lam, g=g, h0=h0 * n)
                    if sweep_trace:                        # every sweep up to the end of the detection loop
                        by_sweep = trace_dir / ("N%d_rep%d_sweeps.csv" % (n, rep))
                        if by_sweep.exists():
                            raise FileExistsError("results are append-only; %s exists" % by_sweep)
                        lines = ["sweep,S,X,pieces,largest,baby,cubes,accepted_in_block"]
                        lines += [",".join(str(v) for v in line) for line in sweep_trace]
                        by_sweep.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                if patch_every:                            # T37; absent from earlier files
                    row["seeds"] = seeds
                    row["seed_sweeps"] = " ".join(seed_sweeps)
                    row["patches_max"] = max(patch_series) if patch_series else 0
                    row["patch_series"] = " ".join(str(v) for v in patch_series)
                    row["converted_series"] = " ".join(str(v) for v in converted_series)
                for m in MARKS:
                    tag = "%d" % int(100 * m)
                    if m in hit:
                        sw, hist, npieces, largest, conv = hit[m]
                        row["sweep_" + tag] = sw
                        row["pieces_" + tag] = npieces
                        row["largest_" + tag] = largest
                        row["converted_" + tag] = conv
                        for k in range(D_BINS):
                            row["d%d_%s" % (k, tag)] = int(hist[k])
                    else:
                        row["sweep_" + tag] = ""
                        row["pieces_" + tag] = ""
                        row["largest_" + tag] = ""
                        row["converted_" + tag] = ""
                        for k in range(D_BINS):
                            row["d%d_%s" % (k, tag)] = ""
                out.write(row)
                print("N=%-4d rep=%-3d waited=%-6s reached=%.2f  released=%.4f (expect %.4f)%s"
                      % (n, rep, waited, row["reached"], h0 - h1, 4 * (lam - 1),
                         ("  at 50%%: d-hist %s, %d pieces, largest %.2f"
                          % ([int(v) for v in hit[0.5][1]], hit[0.5][2], hit[0.5][3])) if 0.5 in hit else ""),
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
