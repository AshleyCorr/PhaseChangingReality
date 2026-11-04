"""`scripts/t71_t74_ledger_checks.py` (item 4 of `docs/design/next_experiments_2026-10-26.md`): the ledger identity,
the wiring-free share, the shortfall test, the clock count and the bucket null, on tiny synthetic files and against the
registered code with its wiring moves off. Fast."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t71_t74_ledger_checks as C  # noqa: E402

LAM, L = 1.1, 8
N_SLAB = 4 * L * L
A_N = 4.0 * (LAM - 1.0) * N_SLAB     # 102.4


def t71_row(replica, sweep, F, stores, fed, n_split, n_contract, n_clean, daughters, dau_e):
    if fed is None:                      # what the slab's ledger says was fed: F + stores + daughter - F0
        fed = F + stores + dau_e - A_N
    return dict(lam=LAM, L=L, N_slab=N_SLAB, rate=0.015625, gate=A_N, leak=0.0, move_set="model",
                release_to="daughter", feed_stops_at_split=0, replica=replica, seed=11 + replica, sweep=sweep, N=384,
                dims="6x8x8", F=F, stores=stores, fed=fed, leaked=0.0, n_split=n_split, n_contract=n_contract,
                n_clean=n_clean, open=384, curled=0, broken=0, daughters=daughters, daughter_energy=dau_e,
                residual=F + stores + dau_e - fed - A_N, first_split="", first_clean="", first_contract="")


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_t71_identity_share_and_remainder(tmp_path):
    # replica 0: three births, no clean move, the stores carried differ between births
    # replica 1: two births, a clean move of cost 3 before the second, which is short by 3
    rows = [
        t71_row(0, 10, 0.0, 20.0, None, 1, 0, 0, 1, A_N),                   # in 20 (F0 paid the daughter), carry 20
        t71_row(0, 30, A_N, 5.0, None, 1, 1, 0, 1, A_N),                    # the contraction (not a birth)
        t71_row(0, 40, 0.0, 30.0, None, 2, 1, 0, 2, 2 * A_N),               # in a N_slab + 10, carry 10
        t71_row(0, 60, 0.0, 30.0, None, 4, 2, 0, 3, 3 * A_N),               # a widening between, carry 0
        t71_row(1, 10, 0.0, 20.0, None, 1, 0, 0, 1, A_N),
        t71_row(1, 20, 3.0, 17.0, None, 1, 0, 1, 1, A_N),                   # a clean move costing 3
        t71_row(1, 40, 3.0, 20.0, None, 2, 1, 1, 2, 2 * A_N - 3.0),         # short by 3: the clean move's cost
    ]
    p = tmp_path / "t71_x.csv"
    write(p, rows)
    kind, got = C.read_rows(str(p))
    assert kind == "t71"
    ac = C.check_a_c(kind, C.group(kind, got))
    (cell, c), = ac.items()
    assert c["identity_word"] == "IDENTITY HOLDS" and c["max_identity"] <= 1e-9
    assert c["births"] == 5 and c["short"] == 1 and c["widenings"] == 1 and c["clean_fed"] == 1
    # the item's remainder is the carry: 20, 10, 0 for replica 0 (in units of a N_slab), 20 and 3 for replica 1
    rems = sorted(b["remainder"] for x in c["reads"].values() for b in x["births"])
    assert rems == pytest.approx([0.0, 0.0, 10.0, 20.0, 20.0])
    assert c["wiring_free"] == 4 and c["pooled_share"] == pytest.approx(0.8)
    assert c["share_word"] == "WIRING-FREE SHARE BELOW 90 %" and c["reps_below"] == 1
    assert c["per_replica"][1]["share"] == pytest.approx(0.5)


def test_identity_fails_when_a_daughter_is_miscounted(tmp_path):
    rows = [t71_row(0, 10, 0.0, 20.0, None, 1, 0, 0, 1, A_N),
            t71_row(0, 40, 0.0, 30.0, 30.0 + A_N - 1.0, 2, 1, 0, 2, 2 * A_N)]
    rows[1]["residual"] = 0.0          # a file whose feed column is off by one unit and whose residual hides it
    p = tmp_path / "t71_bad.csv"
    write(p, rows)
    kind, got = C.read_rows(str(p))
    with pytest.raises(ValueError):    # F0 recovered from the first row is fine; but the identity check reads it...
        C.start_row(kind, dict(got[0], residual=5.0))
    ac = C.check_a_c(kind, C.group(kind, got))
    assert next(iter(ac.values()))["identity_word"] == "IDENTITY FAILS"


def test_shortfall_test_words():
    d = C.check_d([0, 0, 0], [100, 100, 100])
    assert d["word"] == "NO SHORTFALL"
    d = C.check_d([1, 0, 1, 0], [500, 500, 500, 500], draws=2000)
    assert d["word"] == "MEMORYLESS" and 0 < d["p_value"] <= 1
    d = C.check_d([40, 0, 0, 0, 0, 0, 0, 0], [500] * 8, draws=2000)
    assert d["word"] == "SEED-LINKED" and d["p_value"] < 0.05
    assert C.check_d_events([0, 0, 0])["word"] == "NO MOVES"
    assert C.check_d_events([1, 0, 2, 1, 0, 1, 0, 1], draws=2000)["word"] == "MEMORYLESS"
    assert C.check_d_events([12, 0, 0, 0, 0, 0, 0, 0], draws=2000)["word"] == "SEED-LINKED"


def test_clock_exponent_fit():
    fit = C.poisson_exponent([10, 90], [1000, 1000], [256, 2304])      # rate x 9 over N x 9: w = 1
    assert fit["w"] == pytest.approx(1.0, abs=0.01) and fit["lo"] < 1.0 < fit["hi"] and not (fit["lo"] <= 0 <= fit["hi"])
    fit = C.poisson_exponent([50, 50], [1000, 1000], [256, 2304])
    assert fit["w"] == pytest.approx(0.0, abs=0.01) and fit["lo"] < 0.0 < fit["hi"] and not (fit["lo"] <= 1 <= fit["hi"])
    fit = C.poisson_exponent([7, 7], [1000, 1000], [256, 576])         # the stage-1 pair: too few to tell
    assert fit["lo"] <= 0.0 <= fit["hi"] and fit["lo"] <= 1.0 <= fit["hi"]
    assert C.poisson_exponent([0, 0], [10, 10], [256, 576]) is None


def test_take_evenly_per_layer_matches_the_point_version():
    from graphity.layer_outside import take_evenly
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = sorted(rng.random(5).tolist())
        rng.shuffle(s)
        m = 7
        x = float(rng.random() * sum(s) * m)
        pts = np.repeat(np.array(s), m)
        take_evenly(pts, x)
        C.take_evenly_layers(s, m, x)
        assert np.repeat(np.array(s), m) == pytest.approx(pts, abs=1e-9)


@pytest.mark.parametrize("feed,every,rho", [("steady", 1, 0.0), ("meals", 16, 0.0), ("meals", 4, 0.02)])
def test_bucket_null_is_the_registered_code_with_its_wiring_moves_off(feed, every, rho):
    from graphity.contraction import build
    from graphity.layer_outside import Outside, run_layer_outside
    r, n_fed = 0.0625, 240
    R = r * N_SLAB
    c = 2 * R
    o = Outside(n_out=N_SLAB, c=c, n_fed=n_fed, feed=feed, amount=R * every, every=every, e_out0=0.0, rho_fed=rho,
                g_bg_fed=0.3, n_digest_max=0, n_unfed=0)
    seed = 20261026
    _, rows, first = run_layer_outside(build((4, L, L)), LAM, None, seed, outside=o, gate=A_N, move_set="layer",
                                       log_every=1, release_to="daughter")
    births, trace = C.bucket_null(LAM, L, N_SLAB, c, n_fed, feed, R * every, every, 0.0, rho, 0.3, A_N, seed)
    assert len(first["births"]) >= 5
    assert len(births) == len(first["births"])
    for (s1, e1), (s2, e2) in zip(births, first["births"]):
        assert s1 == s2 and e1 == pytest.approx(e2, abs=1e-9)
    by_sweep = {r_[0]: r_ for r_ in rows}
    assert any(r_[8] > 0 for r_ in rows)                       # a contraction happened (the wrap case is exercised below)
    for sweep, stores, e_out, edge_in, w in trace:
        row = by_sweep[sweep]
        assert row[4] == pytest.approx(stores, abs=1e-9)
        assert row[15] == pytest.approx(e_out, abs=1e-9)
        assert row[17] == pytest.approx(edge_in, abs=1e-9)
        assert row[2][0] == w


def test_bucket_null_contraction_wrap_case_matches_the_code():
    """A seed under which a contraction is proposed at x0 = w - 1 inside the first 240 sweeps."""
    from graphity.contraction import build
    from graphity.layer_outside import Outside, run_layer_outside
    r, n_fed = 0.0625, 240
    R = r * N_SLAB
    o = Outside(n_out=N_SLAB, c=2 * R, n_fed=n_fed, feed="steady", amount=R, n_digest_max=0, n_unfed=0)
    found = False
    for seed in range(1, 40):
        rng = np.random.default_rng(seed)
        rng.integers(2 ** 31 - 1)
        rng.integers(2 ** 31 - 1)
        # replay the proposal draws on the null to see whether a wrap contraction fires
        births, trace = C.bucket_null(LAM, L, N_SLAB, 2 * R, n_fed, "steady", R, 1, 0.0, 0.0, 0.0, A_N, seed)
        widths = [w for _, _, _, _, w in trace]
        rng = np.random.default_rng(seed)
        rng.integers(2 ** 31 - 1)
        rng.integers(2 ** 31 - 1)
        prev_w = 4
        for sweep in range(1, n_fed + 1):
            rng.integers(1)
            x0 = int(rng.integers(prev_w))
            contract = bool(rng.integers(2))
            w = widths[sweep - 1]
            if contract and w == prev_w - 2 and x0 == prev_w - 1:
                found = True
            prev_w = w
        if found:
            break
    assert found, "no seed below 40 exercised the wrap case"
    _, rows, first = run_layer_outside(build((4, L, L)), LAM, None, seed, outside=o, gate=A_N, move_set="layer",
                                       log_every=1, release_to="daughter")
    assert [(s, pytest.approx(e, abs=1e-9)) for s, e in births] == first["births"]
    by_sweep = {r_[0]: r_ for r_ in rows}
    for sweep, stores, e_out, edge_in, w in trace:
        assert by_sweep[sweep][4] == pytest.approx(stores, abs=1e-9)
