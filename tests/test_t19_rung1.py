"""T19's and T15 rung 1's reading rules on known inputs."""
import importlib.util
from pathlib import Path

import numpy as np


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t19 = load("analyse_t19")
r1 = load("analyse_t15_rung1")


def b(disp, n):
    return dict(displacement=disp, n_d1=str(n))


def test_t19_outcomes_and_verdict():
    assert t19.outcome([b("0", 4), b("3", 4)]) == "moved"
    assert t19.outcome([b("0", 4), b("", 0)]) == "annealed"
    assert t19.outcome([b("0", 4), b("1", 4)]) == "stayed"
    assert t19.verdict(["stayed"] * 7 + ["annealed"] * 3) == "STAYS"
    assert t19.verdict(["stayed"] * 5 + ["annealed"] * 5) == "MIXED"
    assert t19.verdict(["moved"] * 5) == "NOT READ"


def test_rung1_check():
    p = np.array([0.5, 0.3, 0.2])
    assert r1.check(p + np.array([0.01, -0.01, 0.0]), np.full(3, 0.01), p)[2]
    assert not r1.check(np.array([0.6, 0.2, 0.2]), np.full(3, 0.001), p)[2]
