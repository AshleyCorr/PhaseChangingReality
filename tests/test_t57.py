"""T57's configs and reading (scripts/make_t57_configs.py, scripts/analyse_t57.py), written before any run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a57 = _load("analyse_t57")


def _rows(tie, lam, proto, n, e, replica, folded, melted, sweeps=(250, 500)):
    rows = []
    for s in sweeps:
        r = dict(N=str(n), lam=str(lam), spark=str(e), replica=str(replica), sweep=str(s), melted=str(melted),
                 local_heat=str(proto == "packed"), largest_flat="0")
        d = [0] * 8
        d[2] = folded
        d[3] = n - folded - melted
        d[4] = melted
        r.update({"d%d" % k: str(v) for k, v in enumerate(d)})
        if tie != "none":
            r["tie_label"] = tie
        rows.append(r)
    return rows


def test_groups_cells_and_verdicts():
    rows = []
    for rep in range(8):
        rows += _rows("all at the last", 1.25, "packed", 216, 500.0, rep, folded=40, melted=0)   # FOLDED
        rows += _rows("none", 1.25, "packed", 216, 500.0, rep, folded=0, melted=40)              # MELTED
        rows += _rows("none", 1.10, "spread", 216, 128.0, rep, folded=0, melted=0)               # HEALED
    c, v = a57.summarize(rows)
    assert c[("all at the last", 1.25, "packed", 216, 500.0)][1] == "FOLDED"
    assert c[("none", 1.25, "packed", 216, 500.0)][1] == "MELTED"
    assert v[("all at the last", 1.25)] == "RE-CURLS"
    assert v[("none", 1.25)] == "MELTS"
    assert v[("none", 1.10)] == "HEALS"
    assert a57.group_verdict(["MIXED", "HEALED"]) == "MIXED"


def test_a_cell_with_too_few_replicas_is_mixed():
    rows = []
    for rep in range(3):
        rows += _rows("none", 1.25, "spread", 512, 256.0, rep, folded=0, melted=40)
    c, _ = a57.summarize(rows)
    assert c[("none", 1.25, "spread", 512, 256.0)][1] == "MIXED"


def test_the_t57_configs_if_present():
    cfgs = sorted((ROOT / "configs").glob("t57_*.json"))
    if not cfgs:
        pytest.skip("no T57 configs yet")
    assert len(cfgs) == 16
    seeds, names = set(), set()
    for p in cfgs:
        c = json.loads(p.read_text())
        assert c["name"] == p.stem and c["section"] == "T57" and c["name"] not in names
        names.add(c["name"])
        assert c["seed"] not in seeds
        seeds.add(c["seed"])
        assert c["replicas"] == 8 and c["n_sweeps"] == 50000 and c["save_adjacency"] is True
        assert ("ftable_per_a" in c) == c["name"].startswith("t57_tied_")
        if "ftable_per_a" in c:
            assert c["ftable_per_a"] == [0, 1.0, 2.0, 0] and c["tie_label"] == "all at the last"
        assert ("local_heat" in c) == ("packed" in c["name"])
        assert len(c["sparks"]) == 5 and c["sparks"] == sorted(c["sparks"])
