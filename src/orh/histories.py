"""Decoherent histories on the simulator: class operators, the decoherence functional, sum rules, compatible sets,
nested measures and record surprisal (handoff section 3; Gell-Mann and Hartle 1990/2018 and 1993, whose equations
are transcribed in docs/reading/notes/2026-10-12_orh_decoherent_histories_gmh.md).

A history family is a list of steps. Each step is a pair (evolve, alternatives): `evolve` is a function that applies
the unitary evolution from the previous time to this one to a Register, in place; `alternatives` is a list of
projector specifications, each a dict {qubit name: value}, that must be exhaustive and exclusive (they are, whenever
they run over every value of one fixed set of qubits; `[{}]` is the trivial alternative, no projection at this time).

For the history alpha = (alpha_1, ..., alpha_n) the branch vector is

    |psi_alpha> = P_{alpha_n} U_n ... P_{alpha_1} U_1 |psi_0>  =  C_alpha |psi_0>,

and the decoherence functional of a pure initial state is the Gram matrix D(alpha, beta) = <psi_beta | psi_alpha>.
Its diagonal is the candidate probabilities p_alpha = D(alpha, alpha). The family is (medium) decoherent when the
off-diagonal entries vanish; then and only then do the p_alpha obey every probability sum rule.

Two facts the tests lean on, both exact and both easy to get wrong in words:

1. sum over all alpha, beta of D(alpha, beta) = <psi_0| U^dagger U |psi_0> = 1 always, because the class operators
   sum to the full evolution. So the total of the diagonal, sum_alpha p_alpha, equals 1 minus the sum of the
   off-diagonal entries, and it can equal 1 while the family interferes (the two-slit family does exactly this:
   the off-diagonal terms cancel in pairs). "The weights add to one" is therefore weaker than decoherence; the
   test of a family is the full set of sum rules, one per coarse graining, `sum_rule_violations`.

2. A record kept in a register that nothing later touches decoheres every pair of histories that differ in the
   recorded value, exactly, because the two branch vectors are orthogonal on that register for ever after. The same
   register, once its content is undone or swapped away, no longer does: the family that includes the earlier
   record value stops being decoherent (Gell-Mann and Hartle's caveat about the permanence of the past).

Compatible sets. A record R is a dict {step index: alternative index}. C(R) is the set of history labels that carry
those alternatives at those steps. mu(C) is the sum of p_alpha over C, I(R) = -log2 mu(C(R)) is the record surprisal
of the handoff (its section 3), and the conditional measure within C is p_alpha / mu(C). All of this is bookkeeping
once the family is fixed and decoherent; `nested_record_table` just carries it out and reports whether the family
qualifies.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .statevector import Register, inner


@dataclass(frozen=True)
class Step:
    evolve: object          # callable Register -> None (in place), the unitary from the previous time to this one
    alternatives: tuple     # tuple of dicts {qubit: value}; exhaustive and exclusive at this time
    label: str = ""


def step(evolve, alternatives, label=""):
    return Step(evolve, tuple(dict(a) for a in alternatives), label)


def alternatives_over(names):
    """Every computational-basis assignment of the qubits `names`: an exhaustive, exclusive set of projectors."""
    names = list(names)
    return [dict(zip(names, values)) for values in itertools.product((0, 1), repeat=len(names))]


def branch_vectors(reg0, steps):
    """{alpha: Register} for every history alpha of the family, alpha a tuple of alternative indices."""
    branches = {(): reg0.copy()}
    for s in steps:
        new = {}
        for label, reg in branches.items():
            r = reg.copy()
            s.evolve(r)
            for k, assignment in enumerate(s.alternatives):
                new[label + (k,)] = r.projected(assignment)
        branches = new
    return branches


def decoherence_functional(reg0, steps):
    """(labels, D): the history labels in a fixed order and the Gram matrix D[a, b] = <psi_b | psi_a>."""
    branches = branch_vectors(reg0, steps)
    labels = sorted(branches)
    vecs = np.array([branches[l].vector() for l in labels])
    d = vecs.conj() @ vecs.T          # d[b, a] = <psi_b|psi_a>; we want D[a, b] = <psi_b|psi_a>
    return labels, d.T


def probabilities(d):
    """The diagonal of D, real: the candidate history weights p_alpha."""
    return np.real(np.diag(d))


def max_offdiagonal(d):
    """The largest |D(alpha, beta)| with alpha != beta: zero for an exactly decoherent family."""
    off = np.abs(d - np.diag(np.diag(d)))
    return float(off.max()) if off.size else 0.0


def is_decoherent(d, tol=1e-12):
    return max_offdiagonal(d) <= tol


def total_weight(d):
    """sum_alpha p_alpha. Equal to 1 minus the sum of the off-diagonal entries, so 1 for a decoherent family and
    sometimes 1 for an interfering one too (see the module docstring)."""
    return float(probabilities(d).sum())


def coarse_probability(labels, d, members):
    """The probability of the coarse-grained history made of `members`: Re sum_{a, b in members} D(a, b), which is
    what the Born rule gives for the sum of their class operators."""
    idx = [labels.index(m) for m in members]
    sub = d[np.ix_(idx, idx)]
    return float(np.real(sub.sum()))


def sum_rule_violations(labels, d, classes):
    """For each coarse graining in `classes` (a dict name -> list of member labels), the fine-grained sum
    sum_{a in class} p_a, the coarse-grained probability Re sum_{a, b in class} D(a, b), and their difference. A
    decoherent family has every difference zero; an interfering one does not."""
    p = probabilities(d)
    out = {}
    for name, members in classes.items():
        idx = [labels.index(m) for m in members]
        fine = float(p[idx].sum())
        coarse = coarse_probability(labels, d, members)
        out[name] = dict(fine=fine, coarse=coarse, violation=fine - coarse)
    return out


def compatible(labels, record):
    """The labels alpha whose alternative at step k equals record[k] for every k in the record."""
    return [a for a in labels if all(a[k] == v for k, v in record.items())]


def measure(labels, d, subset):
    """mu(subset) = sum of p_alpha over the subset."""
    p = probabilities(d)
    return float(sum(p[labels.index(a)] for a in subset))


def surprisal_bits(mu):
    """I = -log2 mu, for mu > 0."""
    if mu <= 0:
        return float("inf")
    return float(-np.log2(mu))


def nested_record_table(labels, d, records, tol=1e-12):
    """For a sequence of records R_1 subset R_2 subset ... (each a dict {step: alternative} extending the last):
    the size of C(R_n), its measure mu, its surprisal, the sum of the conditional measure inside it, whether it is
    nested in the previous set, and whether the family is decoherent (without which mu is not a probability).

    Returns a list of dicts, one per record. The caller decides what to make of a non-decoherent row; this function
    reports rather than refuses, so that the failure can be shown."""
    rows = []
    prev = None
    deco = is_decoherent(d, tol)
    p = probabilities(d)
    for n, record in enumerate(records, start=1):
        c = compatible(labels, record)
        mu = measure(labels, d, c)
        cond = float(sum(p[labels.index(a)] for a in c) / mu) if mu > 0 else float("nan")
        nested = True if prev is None else set(c) <= set(prev)
        rows.append(dict(n=n, record=dict(record), size=len(c), mu=mu, surprisal_bits=surprisal_bits(mu),
                         conditional_total=cond, nested=nested, family_decoherent=deco,
                         total_weight=total_weight(d)))
        prev = c
    return rows
