"""The simulator against hand-built matrices: gates, partial traces, entropies, projections (orh.statevector)."""
import numpy as np

from orh.statevector import (CNOT_GATE, H_GATE, Register, controlled, entropy_bits, inner, partial_trace,
                             tilt, visibility, binary_entropy)


def kron(*mats):
    out = np.eye(1, dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out


def random_register(names, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2 ** len(names)) + 1j * rng.normal(size=2 ** len(names))
    return Register(names, v / np.linalg.norm(v))


def test_one_qubit_gate_matches_kron_on_any_position():
    for pos in range(3):
        reg = random_register(["a", "b", "c"], seed=pos)
        before = reg.vector().copy()
        reg.h(["a", "b", "c"][pos])
        mats = [np.eye(2)] * 3
        mats[pos] = H_GATE
        assert np.allclose(reg.vector(), kron(*mats) @ before)


def test_two_qubit_gate_matches_kron_including_reversed_order():
    reg = random_register(["a", "b", "c"], seed=7)
    before = reg.vector().copy()
    reg.cnot("a", "c")                                         # control a, target c, b untouched
    full = np.zeros((8, 8), dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                i = 4 * a + 2 * b + c
                j = 4 * a + 2 * b + (c ^ a)
                full[j, i] = 1.0
    assert np.allclose(reg.vector(), full @ before)
    reg2 = random_register(["a", "b", "c"], seed=7)
    before2 = reg2.vector().copy()
    reg2.cnot("c", "a")                                        # control c, target a
    full2 = np.zeros((8, 8), dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                full2[4 * (a ^ c) + 2 * b + c, 4 * a + 2 * b + c] = 1.0
    assert np.allclose(reg2.vector(), full2 @ before2)


def test_gates_are_unitary_and_h_squared_is_identity():
    reg = random_register(["a", "b"], seed=1)
    v = reg.vector().copy()
    reg.h("a").h("a")
    assert np.allclose(reg.vector(), v)
    assert abs(reg.norm() - 1) < 1e-12
    assert np.allclose(CNOT_GATE @ CNOT_GATE, np.eye(4))
    u = controlled(tilt(0.4))
    assert np.allclose(u @ u.conj().T, np.eye(4))


def test_controlled_tilt_writes_the_record_only_when_the_control_is_one():
    reg = Register(["s", "e"]).controlled_tilt("s", "e", 0.3)
    assert np.allclose(reg.vector(), [1, 0, 0, 0])
    reg = Register(["s", "e"]).x("s").controlled_tilt("s", "e", 0.3)
    assert np.allclose(reg.vector(), [0, 0, np.cos(0.3), np.sin(0.3)])


def test_prepare_gives_the_requested_born_weights():
    reg = Register(["s"]).prepare("s", 0.3)
    assert np.allclose(reg.probabilities(["s"]), [0.7, 0.3])


def test_bell_pair_reduced_states_entropies_and_mutual_information():
    reg = Register(["a", "b"]).h("a").cnot("a", "b")
    assert np.allclose(reg.probabilities(["a", "b"]), [[0.5, 0], [0, 0.5]])
    assert np.allclose(reg.reduced(["a"]), np.eye(2) / 2)
    assert abs(reg.entropy(["a"]) - 1) < 1e-12
    assert abs(reg.entropy(["a", "b"])) < 1e-12
    assert abs(reg.mutual_information(["a"], ["b"]) - 2) < 1e-12
    assert abs(visibility(reg.reduced(["a"]))) < 1e-12


def test_reduced_state_respects_the_requested_qubit_order():
    reg = Register(["a", "b"]).x("a")                          # |10>
    rho_ab = reg.reduced(["a", "b"])
    rho_ba = reg.reduced(["b", "a"])
    assert abs(rho_ab[2, 2] - 1) < 1e-12                       # index 2*a + b = 2
    assert abs(rho_ba[1, 1] - 1) < 1e-12                       # index 2*b + a = 1


def test_partial_trace_of_a_product_state_is_the_factor():
    reg = Register(["a", "b", "c"]).h("a").ry(0.7, "c")
    rho_c = reg.reduced(["c"])
    v = np.array([np.cos(0.35), np.sin(0.35)])
    assert np.allclose(rho_c, np.outer(v, v))
    assert abs(entropy_bits(rho_c)) < 1e-12
    assert abs(reg.mutual_information(["a"], ["c"])) < 1e-12


def test_projection_zeroes_the_other_branch_without_renormalizing():
    reg = Register(["a", "b"]).h("a").cnot("a", "b")
    proj = reg.projected({"a": 1})
    assert np.allclose(proj.vector(), [0, 0, 0, 1 / np.sqrt(2)])
    assert abs(inner(proj, proj).real - 0.5) < 1e-12
    assert abs(inner(reg.projected({"a": 0}), proj)) < 1e-12


def test_partial_trace_matches_an_explicit_sum():
    reg = random_register(["a", "b", "c"], seed=3)
    psi = reg.psi
    rho = partial_trace(psi, [1])                              # keep b
    explicit = np.zeros((2, 2), dtype=complex)
    for a in (0, 1):
        for c in (0, 1):
            v = psi[a, :, c]
            explicit += np.outer(v, v.conj())
    assert np.allclose(rho, explicit)


def test_binary_entropy_endpoints_and_middle():
    assert binary_entropy(0) == 0 and binary_entropy(1) == 0
    assert abs(binary_entropy(0.5) - 1) < 1e-12
