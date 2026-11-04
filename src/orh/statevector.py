"""A pure-state simulator for a handful of named qubits.

The state is an n-index tensor of shape (2,)*n with complex entries, one index per qubit, so a gate on one or two
qubits is a tensordot over those indices. Up to about twenty qubits is comfortable in NumPy.

Everything here is unitary. There is no operation that collapses the state: a "measurement" in the toy is an
entangling gate that writes a record into another qubit (Everett 1957, the relative-state move), and a projector is
applied only to build branch vectors for the decoherence functional in `histories.py`, never to update the state.

Conventions. Basis states of a qubit are |0> and |1>; the computational basis of a set of qubits is ordered with the
first-named qubit most significant, so a reduced density matrix over qubits (a, b) has row index 2*a + b. Entropies
are in bits.
"""
from __future__ import annotations

import numpy as np

SQRT_HALF = 1 / np.sqrt(2.0)
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) * SQRT_HALF
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
I_GATE = np.eye(2, dtype=complex)
SWAP_GATE = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def ry(theta):
    """Rotation about y by `theta`: |0> goes to cos(theta/2)|0> + sin(theta/2)|1>."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def tilt(theta):
    """The gate that sends |0> to cos(theta)|0> + sin(theta)|1>: a record of strength theta (theta = pi/2 is a perfect
    copy, the controlled-NOT; theta = 0 records nothing)."""
    return ry(2 * theta)


def controlled(u):
    """The 4 x 4 controlled-U on (control, target), basis index 2*control + target."""
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = u
    return out


CNOT_GATE = controlled(X_GATE)


class Register:
    """Named qubits holding one pure state. All gates act in place and return the register, so calls chain."""

    def __init__(self, names, psi=None):
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("qubit names must be distinct")
        self.n = len(self.names)
        self.index = {name: i for i, name in enumerate(self.names)}
        if psi is None:
            psi = np.zeros((2,) * self.n, dtype=complex)
            psi[(0,) * self.n] = 1.0
        self.psi = np.asarray(psi, dtype=complex).reshape((2,) * self.n)

    # ---- housekeeping -------------------------------------------------------------------------------------------
    def copy(self):
        return Register(self.names, self.psi.copy())

    def axis(self, name):
        return self.index[name]

    def vector(self):
        """The state as a flat vector in the computational basis (first-named qubit most significant)."""
        return self.psi.reshape(-1)

    def norm(self):
        return float(np.sqrt(np.vdot(self.vector(), self.vector()).real))

    # ---- gates ----------------------------------------------------------------------------------------------------
    def apply(self, u, name):
        """A 2 x 2 unitary on one qubit."""
        q = self.axis(name)
        out = np.tensordot(np.asarray(u, dtype=complex), self.psi, axes=([1], [q]))
        self.psi = np.moveaxis(out, 0, q)
        return self

    def apply2(self, u4, name_a, name_b):
        """A 4 x 4 unitary on two qubits, basis index 2*a + b."""
        a, b = self.axis(name_a), self.axis(name_b)
        if a == b:
            raise ValueError("a two-qubit gate needs two different qubits")
        u = np.asarray(u4, dtype=complex).reshape(2, 2, 2, 2)      # (out_a, out_b, in_a, in_b)
        out = np.tensordot(u, self.psi, axes=([2, 3], [a, b]))     # (out_a, out_b, ...rest...)
        self.psi = np.moveaxis(out, [0, 1], [a, b])
        return self

    def h(self, name):
        return self.apply(H_GATE, name)

    def x(self, name):
        return self.apply(X_GATE, name)

    def z(self, name):
        return self.apply(Z_GATE, name)

    def ry(self, theta, name):
        return self.apply(ry(theta), name)

    def prepare(self, name, p_one):
        """Take a qubit from |0> to sqrt(1 - p_one)|0> + sqrt(p_one)|1> (real amplitudes)."""
        return self.apply(ry(2 * np.arcsin(np.sqrt(p_one))), name)

    def cnot(self, control, target):
        return self.apply2(CNOT_GATE, control, target)

    def controlled_tilt(self, control, target, theta):
        """If the control is |1>, send the target from |0> to cos(theta)|0> + sin(theta)|1>: a record of strength
        theta written on the target. theta = pi/2 is a perfect copy."""
        return self.apply2(controlled(tilt(theta)), control, target)

    def swap(self, name_a, name_b):
        return self.apply2(SWAP_GATE, name_a, name_b)

    # ---- readouts (none of them changes the state) --------------------------------------------------------------
    def probabilities(self, names):
        """Born probabilities of the computational-basis outcomes of `names`, as an array of shape (2,)*len(names)
        indexed in the order given."""
        keep = [self.axis(nm) for nm in names]
        p = np.abs(self.psi) ** 2
        rest = tuple(i for i in range(self.n) if i not in keep)
        p = p.sum(axis=rest) if rest else p
        kept_order = [i for i in range(self.n) if i in keep]      # axes come out in register order
        perm = [kept_order.index(i) for i in keep]
        return np.transpose(p, perm)

    def reduced(self, names):
        """The reduced density matrix of `names`, rows indexed in the order given."""
        return partial_trace(self.psi, [self.axis(nm) for nm in names])

    def entropy(self, names):
        return entropy_bits(self.reduced(names))

    def mutual_information(self, names_a, names_b):
        """I(A:B) = S(A) + S(B) - S(AB) in bits."""
        return (self.entropy(names_a) + self.entropy(names_b)
                - self.entropy(list(names_a) + list(names_b)))

    def projected(self, assignments):
        """A copy of the state with the amplitudes outside the computational-basis assignment `{name: value}` set to
        zero (not renormalized). This is a projector applied to a branch vector, for `histories.py`; it is not a
        state update of this register."""
        psi = self.psi.copy()
        for name, value in assignments.items():
            q = self.axis(name)
            idx = [slice(None)] * self.n
            idx[q] = 1 - int(value)
            psi[tuple(idx)] = 0.0
        return Register(self.names, psi)


def partial_trace(psi, keep):
    """Reduced density matrix of the axes `keep` (in that order) of the pure state tensor `psi`."""
    psi = np.asarray(psi)
    n = psi.ndim
    rest = [i for i in range(n) if i not in keep]
    mat = np.transpose(psi, list(keep) + rest).reshape(2 ** len(keep), -1)
    return mat @ mat.conj().T


def entropy_bits(rho):
    """The von Neumann entropy of a density matrix, in bits."""
    w = np.linalg.eigvalsh(np.asarray(rho))
    w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())


def binary_entropy(p):
    """h(p) in bits."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def coherence(rho):
    """|rho_01| of a qubit density matrix."""
    return float(abs(rho[0, 1]))


def visibility(rho):
    """The fringe visibility a balanced interferometer would show for a qubit in state rho: 2|rho_01| / tr rho.
    For rho_00 = rho_11 = 1/2 this is |<E_B|E_A>|, the overlap of the two environment states (handoff section 3)."""
    return float(2 * abs(rho[0, 1]) / np.trace(rho).real)


def inner(reg_a, reg_b):
    """<a|b> for two registers over the same qubits."""
    return complex(np.vdot(reg_a.vector(), reg_b.vector()))


def maximally_mixed_pair(reg, name, partner):
    """Entangle `name` with `partner` into a Bell pair, so that `name` alone is maximally mixed (used to show that a
    record cannot be written on a register that is not in a low-entropy state)."""
    return reg.h(partner).cnot(partner, name)
