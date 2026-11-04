"""Stage B benchmarks: the simulator against the closed forms of orh.darwinism for the controlled-tilt record model.

Visibility, the partial-information plot I(S:F), its classical plateau, redundancy, the quantum eraser and
recombination, and the disagreement rate of two observers reading different fragments."""
import numpy as np
import pytest

from orh import darwinism as dw
from orh.statevector import Register, binary_entropy, visibility


def model(p, thetas):
    names = ["s"] + [f"e{i}" for i in range(len(thetas))]
    reg = Register(names).prepare("s", p)
    for i, th in enumerate(thetas):
        reg.controlled_tilt("s", f"e{i}", th)
    return reg


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_visibility_is_the_overlap_of_the_environment_branches(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95)
    thetas = rng.uniform(0, np.pi / 2, size=5)
    reg = model(p, thetas)
    assert abs(visibility(reg.reduced(["s"])) - dw.visibility(p, thetas)) < 1e-12
    assert abs(reg.norm() - 1) < 1e-12


def test_balanced_visibility_equals_the_environment_overlap_exactly():
    thetas = [0.2, 0.9, 1.3]
    reg = model(0.5, thetas)
    assert abs(visibility(reg.reduced(["s"])) - np.prod(np.cos(thetas))) < 1e-12


@pytest.mark.parametrize("seed", [3, 4])
def test_mutual_information_matches_the_closed_form_for_every_fragment(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9)
    thetas = rng.uniform(0.1, np.pi / 2, size=4)
    reg = model(p, thetas)
    env = [f"e{i}" for i in range(4)]
    for mask in range(1 << 4):
        frag = [i for i in range(4) if mask >> i & 1]
        rest = [i for i in range(4) if i not in frag]
        sim = reg.mutual_information(["s"], [env[i] for i in frag]) if frag else 0.0
        assert abs(sim - dw.mutual_information(p, thetas, frag)) < 1e-10
        # the complement check that follows from purity: H(SF) = H(rest)
        if frag and rest:
            assert abs(reg.entropy(["s"] + [env[i] for i in frag]) - reg.entropy([env[i] for i in rest])) < 1e-10


def test_perfect_records_give_the_classical_plateau_and_the_quantum_rise():
    p, n = 0.3, 6
    plot = dw.partial_information_plot(p, np.pi / 2, n)
    assert abs(plot[0]) < 1e-12
    assert np.allclose(plot[1:n], binary_entropy(p))
    assert abs(plot[n] - 2 * binary_entropy(p)) < 1e-12
    reg = model(p, [np.pi / 2] * n)
    for m in range(n + 1):
        sim = reg.mutual_information(["s"], [f"e{i}" for i in range(m)]) if m else 0.0
        assert abs(sim - plot[m]) < 1e-10


def test_redundancy_counts_how_many_fragments_each_hold_nearly_everything():
    assert dw.redundancy(0.3, np.pi / 2, 8, 0.1) == 8.0           # every single qubit is a full record
    assert dw.redundancy(0.3, 0.05, 8, 0.1) == float("inf")        # too weak: no fragment short of all suffices
    r = dw.redundancy(0.5, 0.6, 20, 0.1)
    assert 1 < r < 20


def test_quantum_eraser_recovers_fringes_in_each_sorted_subensemble_and_none_overall():
    reg = Register(["s", "e"]).h("s").cnot("s", "e")
    assert abs(visibility(reg.reduced(["s"]))) < 1e-12           # a perfect record kills the fringes
    reg.h("e")                                                   # read the record in the conjugate basis
    for value in (0, 1):
        branch = reg.projected({"e": value})
        rho = branch.reduced(["s"])
        assert abs(visibility(rho) - 1) < 1e-12                  # full visibility in each sorted subensemble
    assert abs(visibility(reg.reduced(["s"]))) < 1e-12           # and none without sorting


def test_recombination_undoes_the_record_exactly():
    reg = Register(["s", "e"]).h("s").controlled_tilt("s", "e", 0.8)
    assert abs(visibility(reg.reduced(["s"])) - np.cos(0.8)) < 1e-12
    reg.controlled_tilt("s", "e", -0.8)                          # the inverse interaction
    assert abs(visibility(reg.reduced(["s"])) - 1) < 1e-12
    assert abs(reg.entropy(["e"])) < 1e-12


@pytest.mark.parametrize("ti,tj", [(np.pi / 2, np.pi / 2), (0.4, 1.1), (1.2, 0.3)])
def test_two_observers_disagree_at_the_closed_form_rate(ti, tj):
    p = 0.35
    reg = Register(["s", "e0", "e1", "oa", "ob"]).prepare("s", p)
    reg.controlled_tilt("s", "e0", ti).controlled_tilt("s", "e1", tj)
    reg.cnot("e0", "oa").cnot("e1", "ob")
    pr = reg.probabilities(["oa", "ob"])
    assert abs((pr[0, 1] + pr[1, 0]) - dw.disagreement_rate(p, ti, tj)) < 1e-12
    if ti == tj == np.pi / 2:
        assert pr[0, 1] < 1e-30 and pr[1, 0] < 1e-30
