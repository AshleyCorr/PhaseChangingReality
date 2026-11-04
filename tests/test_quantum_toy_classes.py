"""The quantum toy on the class space (scripts/quantum_toy_classes.py): construction checks at N = 16."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quantum_toy_classes as q  # noqa: E402


@pytest.fixture(scope="module")
def space16():
    return q.class_space(8)


def test_five_classes_and_the_labelled_count(space16):
    classes, counts = space16
    assert len(classes) == 5
    assert sum(c["size"] for c in classes) == 635040000          # T4's exhaustive count (results/ergodicity_small.csv)


def test_switch_counts_are_reversible_so_the_matrix_is_symmetric(space16):
    classes, counts = space16
    assert q.balance_holds(classes, counts)
    for lam in (0.0, 1.0):
        h = q.hamiltonian(classes, counts, 8, lam, 0.7)
        assert np.allclose(h, h.T)


def test_gamma_zero_is_the_classical_toy(space16):
    classes, counts = space16
    h = q.hamiltonian(classes, counts, 8, 0.0, 0.0)
    e0, _, psi = q.ground_state(h)
    energies = [16 * (16 - c["squares"]) for c in classes]
    assert e0 == min(energies)
    k = int(np.argmax(psi ** 2))
    assert energies[k] == min(energies)


def test_ground_state_is_positive_when_gamma_is_positive(space16):
    classes, counts = space16
    for lam in (0.0, 0.5, 1.0, 1.25):
        _, _, psi = q.ground_state(q.hamiltonian(classes, counts, 8, lam, 0.3))
        assert psi.min() > 0                                       # stoquastic and connected: Perron-Frobenius


def test_weights_sum_to_one(space16):
    classes, counts = space16
    _, _, psi = q.ground_state(q.hamiltonian(classes, counts, 8, 1.0, 1.0))
    s, x, p = q.readings(classes, psi, 8)
    assert abs(p.sum() - 1) < 1e-12
    assert 0 < s < 2 and 0 <= x


def test_symmetric_sector_matches_the_labelled_matrix_on_a_hand_case():
    """Three labeled states, two classes: the projected matrix equals the formula."""
    classes = [dict(class_id=0, squares=0, surplus=0, symmetries=2, size=1),
               dict(class_id=1, squares=0, surplus=0, symmetries=1, size=2)]
    counts = {(0, 1): 2, (1, 0): 1}                                 # one state joined to each of two
    h = q.hamiltonian(classes, counts, 1, 0.0, 1.0)
    full = np.array([[0, -1, -1], [-1, 0, 0], [-1, 0, 0]], dtype=float)
    a = np.array([1, 0, 0.0]); b = np.array([0, 1, 1.0]) / np.sqrt(2)
    proj = np.array([[a @ full @ a, a @ full @ b], [b @ full @ a, b @ full @ b]])
    diag = 16.0 * 2 * np.eye(2)
    assert np.allclose(h - diag, proj)
