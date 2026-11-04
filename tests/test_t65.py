"""T65: the sealed runner's start from a slab with a hand-opened patch (2026-10-15)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exact_t54_patch import patch, tie_table                     # noqa: E402
from run_sealed_curled_d import start_of                          # noqa: E402
from graphity.cqg_d import is_valid, torus                        # noqa: E402
from graphity.sealed_tie_d import energy_table_d                  # noqa: E402
from graphity.dimension import local_dimension_d                  # noqa: E402


def test_patch_start_matches_exact_t54_and_is_valid():
    cfg = {"patch": {"L": 8, "k": 3}}
    adj, part, label = start_of(cfg)
    adj0, _ = torus([4, 8, 8])
    ref, _ = patch(adj0, 8, 3)
    assert (adj == ref).all() and label == "4 8 8 patch 3"
    assert is_valid(adj)
    lam = 1.25
    ftab = tie_table("all_at_the_last", lam)
    # the slab holds 3a per point under the tie; the patch state lies above it by T54's exact dH(3) at L = 8
    e_slab = energy_table_d(adj0, lam, ftab)
    assert abs(e_slab / adj0.shape[0] - 3 * 4 * (lam - 1)) < 1e-9
    # T54's exact quadratic under the tie: dH(k) = (176 - 112 lam) k - 48 (lam - 1) k^2, which is 24 at k = 1 and
    # exactly 0 at k = 3 for lam = 1.25 (O118)
    assert abs(energy_table_d(adj, lam, ftab) - e_slab) < 1e-9
    adj1, _, _ = start_of({"patch": {"L": 8, "k": 1}})
    assert abs(energy_table_d(adj1, lam, ftab) - e_slab - 24.0) < 1e-9


def test_full_patch_is_flat_space():
    adj, part, _ = start_of({"patch": {"L": 8, "k": 8}})
    d = local_dimension_d(adj)
    assert (d == 3).all()
    assert abs(energy_table_d(adj, 1.25, tie_table("all_at_the_last", 1.25))) < 1e-9


def test_patch_absent_leaves_the_runner_as_it_was():
    adj, part, label = start_of({"dims": [4, 6, 6]})
    ref, _ = torus([4, 6, 6])
    assert (adj == ref).all() and label == "4 6 6"


def _row(n, d3, dam, sweep=30000, tie="all at the last", lam=1.25, h0=3.0, h1=0.0, total=0.0, spark=16.0, rep=0, dims="4 8 8", c=256):
    d2 = n - d3 - dam
    return dict(N=str(n), dims=dims, C=str(c), spark=str(spark), replica=str(rep), lam=str(lam), tie_label=tie, sweep=str(sweep),
                d0="0", d1="0", d2=str(d2), d3=str(d3), damaged_d=str(dam), h_tie_per_vertex=str(h1 if sweep else h0),
                h_per_vertex=str(h1 if sweep else h0), total=str(total))


def test_t65_replica_words_and_lambda_verdicts():
    from analyse_t65 import replica_outcome, lambda_verdict, cells, k_of
    assert replica_outcome(_row(256, 240, 2)) == "OPENS"
    assert replica_outcome(_row(256, 240, 8)) == "PARTIAL"          # damaged above 2 % of N
    assert replica_outcome(_row(256, 10, 0)) == "FALLS BACK"
    assert replica_outcome(_row(256, 100, 40)) == "DAMAGED"
    assert replica_outcome(_row(256, 100, 10)) == "PARTIAL"
    assert lambda_verdict(["PARTIAL", "OPENS"]) == "OPENS IN ROOM"
    assert lambda_verdict(["FALLS BACK", "FALLS BACK"]) == "FALLS BACK"
    assert lambda_verdict(["PARTIAL", "DAMAGED"]) == "DAMAGED"
    assert lambda_verdict(["PARTIAL", "MIXED"]) == "PARTIAL"
    assert k_of({"dims": "4 8 8 patch 2"}) == 2 and k_of({"dims": "4 8 8"}) == 0
    rows = []
    for rep in range(8):
        rows.append(_row(256, 0, 0, sweep=0, rep=rep))
        rows.append(_row(256, 250, 1, sweep=30000, rep=rep, total=3.0 * 256 * 0.9))
    c = cells(rows)
    (maj, counts, rel, share, firsts), = c.values()
    assert maj == "OPENS" and counts["OPENS"] == 8 and abs(rel - 3.0) < 1e-9 and abs(share - 0.9) < 1e-9 and firsts == [30000] * 8


def test_t65_configs_match_the_registration():
    from make_t65_configs import cells, config
    stage1 = cells()
    names = [c[0] for c in stage1]
    assert len(stage1) == 8 and len(set(names)) == 8
    for i, (name, start, lam, tied, sparks, label) in enumerate(stage1):
        cfg = config(name, start, lam, tied, sparks, label, 1 + i)
        assert cfg["capacities"] == ["N", "4*N"] and cfg["replicas"] == 8 and cfg["n_sweeps"] == 30000
        assert ("ftable_per_a" in cfg) == tied
        if "patch" in cfg:
            assert cfg["patch"]["k"] == 2 and cfg["lambda"] == 1.25 and cfg["sparks"] == [16.0, 32.0]
        elif not tied:
            assert cfg["sparks"] == [128.0]
        else:
            assert cfg["sparks"] == [16.0, 32.0, 64.0, 128.0] and cfg["lambda"] in (1.20, 1.25)
    held = cells(held_out=True)
    assert len(held) == 4 and all(c[1].get("dims", [None, 16])[1] == 16 or c[1]["patch"]["L"] == 16 for c in held)
