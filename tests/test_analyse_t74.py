"""The T74 reader on synthetic rows (every registered word, as amended by Amendments 2 and 3, and our prediction for
the equal background read against 2a), the config generator (108 registered cells, seeds one per cell, no L = 16), and
the runner end to end on a tiny config written to a temporary folder."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t74 as A            # noqa: E402
import make_t74_configs as M       # noqa: E402
import run_layer_outside as RLO    # noqa: E402

LAM, NS = 1.25, 64
AN = 4.0 * (LAM - 1.0) * NS          # a N_slab = 64


def replica(rep, births, s_start, s_end, feed="steady", cap="c2R", bg="cold", n_fed=100, digest=0, n_unfed=100,
            c=1.0, energy=AN, fed=100.0, edge_in=50.0, s_fed=None, g_common=1.0):
    """Rows for one replica: a row at sweep 1, one per birth, the rows at n_fed, at the end of digestion
    (t_unfed = n_fed + digest) and at the end. The stores are s_fed (default s_start) up to n_fed, s_start through
    digestion, s_end after."""
    t_unfed = n_fed + digest
    s_fed = s_start if s_fed is None else s_fed
    base = dict(lam=LAM, L=4, N_slab=NS, feed_label=feed, cap_label=cap, background=bg, replica=rep, n_fed=n_fed,
                n_unfed=n_unfed, c=c, residual=0.0, t_unfed=t_unfed, digest_balanced=1, g_common=g_common)
    rows, d, e = [], 0, 0.0
    sweeps = sorted(set([1] + births + [n_fed, t_unfed, t_unfed + n_unfed]))
    for s in sweeps:
        if s in births:
            d += 1
            e += energy
        stores = s_fed if s <= n_fed else (s_start if s <= t_unfed else s_end)
        rows.append(dict(base, sweep=s, daughters=d, daughter_energy=e, fed=fed if s >= n_fed else fed * s / n_fed,
                         edge_in=edge_in, stores=stores))
    return rows


def test_fed_words():
    burps = [r for k in range(16) for r in replica(k, [10, 30, 50], 10.0, 10.0)]
    cells = A.read_cells(burps)
    assert next(iter(cells.values()))["fed_word"] == "BURPS"
    never = [r for k in range(16) for r in replica(k, [], 10.0, 10.0)]
    assert next(iter(A.read_cells(never).values()))["fed_word"] == "NEVER BURPS"
    mixed = [r for k in range(16) for r in replica(k, [10] if k else [], 10.0, 10.0)]
    assert next(iter(A.read_cells(mixed).values()))["fed_word"] == "MIXED"
    off = [r for k in range(16) for r in replica(k, [10, 30], 10.0, 10.0, energy=AN + 1e-6)]
    assert next(iter(A.read_cells(off).values()))["fed_word"] == "MIXED"   # a daughter not exactly a N_slab
    # Amendment 3, item 1: a daughter born during digestion (after n_fed = 100, before t_unfed = 120) is fed
    digesting = [r for k in range(16) for r in replica(k, [110], 10.0, 10.0, digest=20)]
    c = next(iter(A.read_cells(digesting).values()))
    assert c["fed_word"] == "BURPS" and c["median_digestion_daughters"] == 1 and c["median_unfed_daughters"] == 0
    assert c["n_periods"] == 0                                 # the timing reads the fed phase proper only


def word(rows):
    return next(iter(A.read_cells(rows).values()))["unfed_word"]


def test_unfed_words():
    """Amendment 2, item 3, as amended by Amendment 3 (items 2 and 3)."""
    waits = [r for k in range(16) for r in replica(k, [10], 10.0, 9.6 if k else 1.0)]   # 15 of 16 within 5 %
    assert word(waits) == "WAITS"
    up = [r for k in range(16) for r in replica(k, [10], 10.0, 10.4)]                   # up 4 %: still waits
    assert word(up) == "WAITS"
    few = [r for k in range(16) for r in replica(k, [10], 10.0, 9.6 if k > 1 else 1.0)]  # only 14 of 16
    assert word(few) == "MIXED"
    drained = [r for k in range(16) for r in replica(k, [10], 10.0, 5.0 if k else 6.0)]  # 15 end at half or less
    assert word(drained) == "DISSOLVES"
    rich = [r for k in range(16) for r in replica(k, [10], 10.0, 10.6)]                 # up more than 5 %
    assert word(rich) == "FED BY ITS SURROUNDINGS"
    # a daughter born in the unfed phase (after t_unfed = 100) is fed by its surroundings even with the stores
    # unchanged, and rules out WAITS
    born = [r for k in range(16) for r in replica(k, [10, 150], 10.0, 10.0)]
    c = next(iter(A.read_cells(born).values()))
    assert c["unfed_word"] == "FED BY ITS SURROUNDINGS" and c["n_wait"] == 0
    assert c["median_unfed_daughters"] == 1
    # born on the way down: both bars met, MIXED with both noted (Amendment 3, item 3)
    both = [r for k in range(16) for r in replica(k, [10, 150], 10.0, 2.0)]
    c = next(iter(A.read_cells(both).values()))
    assert c["unfed_word"] == "MIXED (bars met: DISSOLVES and FED BY ITS SURROUNDINGS)"
    assert c["bars_met"] == ["DISSOLVES", "FED BY ITS SURROUNDINGS"]
    middle = [r for k in range(16) for r in replica(k, [10], 10.0, 8.0)]                # down 20 %: no word
    assert word(middle) == "MIXED"


def test_the_unfed_words_are_read_against_the_end_of_digestion():
    """Amendment 3, item 2: the start is the slab's stored energy at t_unfed, not at n_fed. Stores 100 at the end of
    the meals, 10 when digestion ends (a refold spent them), 9.8 at the end: WAITS, not DISSOLVES."""
    rows = [r for k in range(16) for r in replica(k, [10], 10.0, 9.8, digest=30, s_fed=100.0)]
    c = next(iter(A.read_cells(rows).values()))
    assert c["unfed_word"] == "WAITS" and c["replicas"][0]["s_unfed_start"] == 10.0
    assert c["median_digestion"] == 30


def test_the_equal_prediction_is_read_against_2a():
    """Amendment 3, item 4 (ours): equal predicts WAITS below 2a (here 2a = 2) and FED BY ITS SURROUNDINGS above; a cell
    with replicas on both sides is SPLIT and not scored; cold predicts DISSOLVES, warm FED BY ITS SURROUNDINGS."""
    below = [r for k in range(16) for r in replica(k, [10], 10.0, 10.0, bg="equal", g_common=1.5)]
    c = next(iter(A.read_cells(below).values()))
    assert c["predicted"] == "WAITS" and c["prediction_holds"] is True and c["n_common_above_2a"] == 0
    above = [r for k in range(16) for r in replica(k, [10], 10.0, 10.0, bg="equal", g_common=2.5)]
    c = next(iter(A.read_cells(above).values()))
    assert c["predicted"] == "FED BY ITS SURROUNDINGS" and c["prediction_holds"] is False
    split = [r for k in range(16) for r in replica(k, [10], 10.0, 10.0, bg="equal", g_common=2.5 if k < 4 else 1.5)]
    c = next(iter(A.read_cells(split).values()))
    assert c["predicted"] == "SPLIT (4 of 16 above 2a)" and c["prediction_holds"] is None
    cold = [r for k in range(16) for r in replica(k, [10], 10.0, 1.0, bg="cold")]
    c = next(iter(A.read_cells(cold).values()))
    assert c["predicted"] == "DISSOLVES" and c["prediction_holds"] is True
    warm = [r for k in range(16) for r in replica(k, [10, 150], 10.0, 9.0, bg="warm")]
    c = next(iter(A.read_cells(warm).values()))
    assert c["predicted"] == "FED BY ITS SURROUNDINGS" and c["prediction_holds"] is True


def test_timing_words():
    steady = [r for k in range(16) for r in replica(k, [10, 20, 30, 40], 1.0, 1.0)]          # periods 10, cv 0
    same = [r for k in range(16) for r in replica(k, [10, 21, 30, 41], 1.0, 1.0, feed="meals16")]  # mean 10.33
    spread = [r for k in range(16) for r in replica(k, [10, 15, 30, 35, 50], 1.0, 1.0, feed="meals256")]  # mean 10
    t = A.timing(A.read_cells(steady + same + spread))
    assert t[(LAM, 4, "c2R", "meals16")][0] == "SPREAD ONLY"     # steady cv is 0, so any spread doubles it
    assert t[(LAM, 4, "c2R", "meals256")][0] == "SPREAD ONLY"
    steady2 = [r for k in range(16) for r in replica(k, [10, 18, 30, 38, 50], 1.0, 1.0)]     # mean 10, cv 0.2
    same2 = [r for k in range(16) for r in replica(k, [10, 19, 30, 39, 50], 1.0, 1.0, feed="meals16")]
    diff = [r for k in range(16) for r in replica(k, [10, 30, 50], 1.0, 1.0, feed="meals256")]  # mean 20
    t = A.timing(A.read_cells(steady2 + same2 + diff))
    assert t[(LAM, 4, "c2R", "meals16")][0] == "SAME TIMING"
    assert t[(LAM, 4, "c2R", "meals256")][0] == "DIFFERENT TIMING"


def test_timing_pools_the_backgrounds():
    """The fed phase is isolated (Amendment 2), so the three backgrounds' cells are pooled for the timing words: a
    steady feed whose cold cells have period 10 and warm cells period 14 pools to 12, against meals at 12."""
    steady = ([r for k in range(16) for r in replica(k, [10, 20, 30], 1.0, 1.0, bg="cold")]
              + [r for k in range(16) for r in replica(k, [10, 24, 38], 1.0, 1.0, bg="warm")])
    meals = [r for bg in ("cold", "warm") for k in range(16)
             for r in replica(k, [10, 22, 34], 1.0, 1.0, feed="meals16", bg=bg)]
    t = A.timing(A.read_cells(steady + meals))
    word_, ratio, _ = t[(LAM, 4, "c2R", "meals16")]
    assert ratio == pytest.approx(1.0) and word_ == "SAME TIMING"


def test_the_generator_writes_the_registered_cells(tmp_path):
    """Amendments 2 and 3: 108 cells, seeds 20267401 to 20267508 in the order lambda, L, feed, cap, background; fed
    4,096 sweeps isolated, digestion isolated up to 4,096 sweeps to a 1 % balance, then 4,096 relaxing at 1/64."""
    M.main(str(tmp_path))
    cfgs = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("t74_*.json"))]
    assert len(cfgs) == 108
    assert sorted(c["seed"] for c in cfgs) == list(range(20267401, 20267509))
    order = sorted(cfgs, key=lambda c: c["seed"])
    keys = [(c["lam"], c["L"], c["feed_label"], c["cap_label"], c["background"]) for c in order]
    assert keys == [(lam, L, f, cap, bg) for lam in (1.10, 1.25, 1.40) for L in (8, 12)
                    for f in ("steady", "meals16", "meals256") for cap in ("c2R", "cRhalf")
                    for bg in ("cold", "equal", "warm")]
    assert {c["L"] for c in cfgs} == {8, 12} and {c["lam"] for c in cfgs} == {1.10, 1.25, 1.40}
    assert {(c["feed"], c["every"]) for c in cfgs} == {("steady", 1), ("meals", 16), ("meals", 256)}
    assert all(c["replicas"] == 16 and c["r"] == 1 / 16 and c["rho_fed"] == 0.0 and c["rho_unfed"] == 1 / 64 and c["gate"] == "a * N_slab"
               and c["move_set"] == "model" and c["release_to"] == "daughter" for c in cfgs)
    for c in cfgs:
        env = RLO.cell_numbers(c)
        a, R = 4 * (c["lam"] - 1), env["R"]
        assert env["c"] == pytest.approx(2 * R if c["cap_label"] == "c2R" else R / 2)
        if c["background"] == "equal":
            assert env["g_bg_unfed"] == "equal"
        else:
            assert env["g_bg_unfed"] == pytest.approx(4 * a if c["background"] == "warm" else 0.0)
        assert env["rho_fed"] == 0.0 and env["rho_unfed"] == pytest.approx(1 / 64)
        assert env["n_fed"] == env["n_unfed"] == env["n_digest_max"] == 4096 and env["digest_tol"] == 0.01
        assert env["meal"] == pytest.approx(R * c["every"])
        assert env["n_fed"] % c["every"] == 0                          # every feed delivers the same energy
    M.main(str(tmp_path))                                              # identical: left alone, no error
    p = tmp_path / (cfgs[0]["name"] + ".json")
    p.write_text(p.read_text().replace('"replicas": 16', '"replicas": 3'))
    with pytest.raises(FileExistsError):
        M.main(str(tmp_path))


def test_runner_end_to_end(tmp_path):
    cfg = dict(name="t74_tiny", section="T74", written="test", lam=LAM, L=6, r=1 / 16, feed="meals", feed_label="meals16",
               every=16, c="2 * R", cap_label="c2R", n_out="N_slab", background="equal", rho_fed=0.0, g_bg_fed="0",
               rho_unfed=1 / 64, g_bg_unfed="equal", n_fed="64", n_digest_max="64", digest_tol=0.01, n_unfed="64",
               gate="a * N_slab", e_out0=0.0, move_set="model",
               release_to="daughter", max_width=8, log_every=25, log_events=True, replicas=2, seed=1)
    p = tmp_path / "t74_tiny.json"
    p.write_text(json.dumps(cfg))
    RLO.main(str(p), str(tmp_path / "res"))
    rows = list(csv.DictReader(open(tmp_path / "res" / "t74_tiny.csv", newline="")))
    assert (tmp_path / "res" / "t74_tiny.meta.json").exists()
    assert all(abs(float(r["residual"])) < 1e-9 * max(1.0, float(r["fed"])) for r in rows)
    for rep_ in ("0", "1"):
        mine = [r for r in rows if r["replica"] == rep_]
        t = int(mine[0]["t_unfed"])
        assert 64 <= t <= 128 and {int(r["sweep"]) for r in mine} >= {64, t, t + 64}
        assert max(int(r["sweep"]) for r in mine) == t + 64
        assert all(float(r["far"]) == 0.0 for r in mine if int(r["phase"]) in (1, 2))   # isolated fed and digesting
        for r in mine:
            if int(r["phase"]) == 3:                                   # equal: the common mean at the unfed start
                assert float(r["g_bg"]) == pytest.approx(float(r["g_common"]))
        assert int(mine[0]["common_above_2a"]) == int(float(mine[0]["g_common"]) > 2 * 4 * (LAM - 1))
    cells = A.read_cells(rows)
    assert len(cells) == 1
    with pytest.raises(FileExistsError):
        RLO.main(str(p), str(tmp_path / "res"))                        # results are append-only
