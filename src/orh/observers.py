"""Observers as registers inside one unitary world (Stage D of the handoff).

An observer is a named set of register qubits. It reads a record by a controlled-NOT from the source (an environment
fragment, another observer's register); it tells another observer by a controlled-NOT from its own register; it
forgets by swapping a register into a fresh environment qubit (the record leaves the observer but not the world; the
branch weight is untouched); it is duplicated by copying a register into a fresh one. Birth is the first coupling,
death is every register swapped away. None of these is a collapse: the world's state stays pure throughout.

What the module measures is joint distributions of record values, read from the Born probabilities of the register
qubits (`World.joint`), never products of marginals (handoff section 4). `World.history_family` hands the same world to
`histories.py` so that the Born weights of the branches can be compared before and after an observer is duplicated.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import histories as hist
from .statevector import Register


class World:
    def __init__(self, names):
        self.reg = Register(names)
        self.observers = {}       # name -> list of register qubits, in the order written
        self.log = []

    # ---- building the world -----------------------------------------------------------------------------------
    def prepare(self, qubit, p_one):
        self.reg.prepare(qubit, p_one)
        self.log.append(("prepare", qubit, p_one))
        return self

    def tilt_record(self, source, target, theta):
        """The environment qubit `target` records `source` with strength theta (pi/2 is a perfect copy)."""
        self.reg.controlled_tilt(source, target, theta)
        self.log.append(("tilt_record", source, target, theta))
        return self

    def observe(self, observer, source, register):
        """Observer `observer` reads `source` into its register qubit `register` (a controlled-NOT)."""
        self.reg.cnot(source, register)
        self.observers.setdefault(observer, [])
        if register not in self.observers[observer]:
            self.observers[observer].append(register)
        self.log.append(("observe", observer, source, register))
        return self

    def tell(self, speaker, speaker_register, listener, listener_register):
        """`speaker` tells `listener` the value in one of its registers."""
        return self.observe(listener, speaker_register, listener_register)

    def forget(self, observer, register, sink):
        """The record in `register` leaves the observer into the fresh environment qubit `sink`; the register comes
        back as |0>. Unitary; the world keeps the information, the observer does not."""
        self.reg.swap(register, sink)
        self.observers[observer].remove(register)
        self.log.append(("forget", observer, register, sink))
        return self

    def unrecord(self, source, register):
        """Undo a controlled-NOT record while the source is intact: true erasure (recoherence)."""
        self.reg.cnot(source, register)
        for regs in self.observers.values():
            if register in regs:
                regs.remove(register)
        self.log.append(("unrecord", source, register))
        return self

    def duplicate(self, observer, register, copy_register, as_observer=None):
        """Copy one of an observer's registers into a fresh qubit. With `as_observer` the copy is a new observer
        (an identical twin holding the same record); without, it is one more register of the same observer."""
        self.reg.cnot(register, copy_register)
        owner = as_observer or observer
        self.observers.setdefault(owner, []).append(copy_register)
        self.log.append(("duplicate", observer, register, copy_register, owner))
        return self

    def die(self, observer, sinks):
        """Every register of `observer` is swapped into fresh environment qubits `sinks` (one per register)."""
        regs = list(self.observers.get(observer, []))
        if len(sinks) < len(regs):
            raise ValueError("one sink per register is needed")
        for r, s in zip(regs, sinks):
            self.forget(observer, r, s)
        self.log.append(("die", observer))
        return self

    # ---- reading the world (nothing here changes the state) ---------------------------------------------------
    def alive(self):
        return [name for name, regs in self.observers.items() if regs]

    def register_count(self, observer):
        return len(self.observers.get(observer, []))

    def joint(self, groups):
        """The joint Born distribution of the register values of several observers, as
        {(values of group 1, values of group 2, ...): probability}. `groups` is a dict observer -> list of qubits
        (or a list of qubit lists). This is the object the handoff asks for in place of a product of marginals."""
        groups = list(groups.values()) if isinstance(groups, dict) else [list(g) for g in groups]
        flat = [q for g in groups for q in g]
        p = self.reg.probabilities(flat)
        out = {}
        for values in itertools.product((0, 1), repeat=len(flat)):
            prob = float(p[values])
            if prob > 1e-15:
                key, k = [], 0
                for g in groups:
                    key.append(tuple(values[k:k + len(g)]))
                    k += len(g)
                out[tuple(key)] = prob
        return out

    def marginal(self, qubits):
        p = self.reg.probabilities(list(qubits))
        return {values: float(p[values]) for values in itertools.product((0, 1), repeat=len(qubits))
                if p[values] > 1e-15}

    def product_of_marginals(self, groups):
        """What the handoff forbids: the product of the observers' marginals, for comparison with `joint`."""
        groups = list(groups.values()) if isinstance(groups, dict) else [list(g) for g in groups]
        margs = [self.marginal(g) for g in groups]
        out = {}
        for combo in itertools.product(*[m.items() for m in margs]):
            key = tuple(k for k, _ in combo)
            out[key] = float(np.prod([v for _, v in combo]))
        return out

    def record_mutual_information(self, qubits_a, qubits_b):
        """I(A:B) between two observers' registers, in bits (zero if their reports are independent)."""
        return self.reg.mutual_information(list(qubits_a), list(qubits_b))

    def outcome_weights(self, qubit):
        """The Born weights of the two values of `qubit`, read from the world as it stands: the branch weights that
        duplication of an observer must not change."""
        p = self.reg.probabilities([qubit])
        return float(p[0]), float(p[1])


def disagreement(joint, group_a=0, group_b=1):
    """From a joint distribution over two single-qubit records, the probability that they differ."""
    return float(sum(v for k, v in joint.items() if k[group_a] != k[group_b]))
