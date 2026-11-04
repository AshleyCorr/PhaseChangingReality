"""T26 and T27 reading rules (scripts/analyse_t26.py) on rows whose answers are known."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyse_t26", SCRIPTS / "analyse_t26.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def run(rep, path, protocol="stores", points="named", E=32.0, radius="2", leak=0.0, n=144):
    """path: list of (folded, melted) per block."""
    rows = []
    for b, (f, m) in enumerate(path):
        rows.append(dict(N=str(n), protocol=protocol, points=points, E=str(E), radius=radius, leak=str(leak),
                         replica=str(rep), block=str(b), folded=str(f), melted=str(m), largest_folded=str(f)))
    return rows


def test_outcomes_from_the_final_block():
    assert a.outcome(run(0, [(0, 0), (2, 6), (8, 2)])) == "FOLDED"
    assert a.outcome(run(0, [(0, 0), (2, 6), (2, 6)])) == "MELTED"
    assert a.outcome(run(0, [(0, 0), (2, 6), (2, 1)])) == "HEALED"
    assert a.outcome(run(0, [(0, 0), (2, 2)])) == "FOLDED"          # a tie counts as folded: 4 damaged, half folded


def test_melt_then_fold_needs_a_melt_first_and_a_fold_after():
    assert a.melt_then_fold(run(0, [(0, 0), (1, 7), (6, 2)]))
    assert not a.melt_then_fold(run(0, [(0, 0), (6, 2), (1, 7)]))
    assert not a.melt_then_fold(run(0, [(0, 0), (6, 0), (6, 0)]))


def test_cells_take_the_majority_and_need_enough_replicas():
    rows = []
    for rep in range(8):
        rows += run(rep, [(0, 0), (0, 8), (8 if rep < 5 else 0, 0 if rep < 5 else 8)])   # 5 fold, 3 melt
    for rep in range(3):
        rows += run(rep, [(0, 0), (0, 8)], E=64.0)                                       # too few
    cm = a.cells(rows)
    k32 = ("stores", "named", 32.0, "2", 0.0, 144, "False")
    k64 = ("stores", "named", 64.0, "2", 0.0, 144, "False")
    assert cm[k32]["majority"] == "FOLDED" and cm[k32]["counts"] == {"FOLDED": 5, "MELTED": 3}
    assert abs(cm[k32]["melt_then_fold"] - 5 / 8) < 1e-9
    assert cm[k64]["majority"] == "NOT READ"


def test_verdicts_sealed_and_leaking():
    fold = {("stores", "named", 32.0, "2", 0.0, 144, "False"): dict(majority="FOLDED"),
            ("patch", "named", 32.0, "2", 0.0, 144, "False"): dict(majority="MELTED")}
    assert a.verdict(fold, leaking=False) == "RE-CURLS"
    melt = {("stores", "named", 32.0, "2", 0.0, 144, "False"): dict(majority="MELTED"),
            ("stores", "named", 64.0, "2", 0.0, 144, "False"): dict(majority="HEALED")}
    assert a.verdict(melt, leaking=False) == "MELTS"
    heal = {("stores", "named", 32.0, "2", 0.0, 144, "False"): dict(majority="HEALED")}
    assert a.verdict(heal, leaking=False) == "HEALS"
    mixed = {("stores", "named", 32.0, "2", 0.0, 144, "False"): dict(majority="MIXED")}
    assert a.verdict(mixed, leaking=False) == "MIXED"
    leaky = {("bath", "named", 576.0, "", 0.01, 144, "False"): dict(majority="HEALED"),
             ("bath", "named", 576.0, "", 0.1, 144, "False"): dict(majority="FOLDED"),
             ("bath", "named", 576.0, "", 0.0, 144, "False"): dict(majority="MELTED")}   # the control: not scored
    assert a.verdict(leaky, leaking=True) == "FOLDS BEFORE IT FLATTENS"
    assert a.verdict(leaky, leaking=False) == "NOT READ"           # bath cells never count for T26
