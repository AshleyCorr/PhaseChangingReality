"""The quantum toy on the smallest class spaces, exact (docs/design/quantum_toy_brief.md, step 1). Usage:

    python scripts/quantum_toy_classes.py <n> [<n> ...]        (n vertices a side; N = 2n; writes results/)

THE MODEL. Basis: the valid four-link bipartite wirings on n + n labeled vertices (graphity.small_graphs). Hamiltonian
H = H0 + H1, with H0 the toy's energy unchanged, H0|G> = [16 (N - S) + 4 lam X] |G>, and H1 = -Gamma * sum over valid
edge switches s of |s(G)><G|, which is the toy's move made coherent. Every off-diagonal element is -Gamma or 0, so H is
stoquastic and, the switch graph being connected (T4, Q9), its ground state is unique and positive (Perron-Frobenius).

THE CLASS SPACE (exact, ours). Renaming vertices within a side commutes with H. A unique ground state is therefore
invariant under renaming, so it lies in the symmetric sector, spanned by |C> = |C|^(-1/2) sum over g in C of |g> for each
class C (the classes of graphity.small_graphs: wirings equal up to renaming within sides). In that basis

    <C|H0|C>  = E_C
    <C'|H1|C> = -Gamma * sqrt(|C| / |C'|) * n(C -> C')

where n(C -> C') is the number of valid switches from any one wiring of C that land in C' (the same for every member
of C, by renaming), and |C| = (n!)^2 / A_C with A_C the class's renamings. The matrix is symmetric because a switch is
undone by a switch, so |C| n(C -> C') = |C'| n(C' -> C), which the code checks. Switches that stay inside a class give
the diagonal term -Gamma n(C -> C). The weight of class C in the ground state is |psi_C|^2.

WHAT IS COMPUTED, per (n, lam, Gamma): the ground energy, the gap to the first excited level, <S>/N and <X>/N in the
ground state, and the weight of every class. Gamma = 0 must give the classical ground set (tested); at large Gamma the
ground state is the top eigenvector of the switch matrix alone.
Pure functions, tested in tests/test_quantum_toy_classes.py.
"""
import csv
import json
import math
import platform
import sys
from collections import Counter
from math import factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                                  # noqa: E402
from graphity.cqg import surplus, total_squares                                     # noqa: E402
from graphity.small_graphs import ClassList, count_labelled_states, one_switch_away  # noqa: E402

LAMBDAS = (0.0, 0.5, 1.0, 1.25)
GAMMAS = (0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)


def class_space(n, start=None):
    """Every class reachable from `start` by switches, with the switch counts between them.

    Returns (classes, counts), classes a list of dicts (squares, surplus, symmetries, size, energy inputs) and counts a
    dict {(k, j): n(k -> j)} over ordered pairs, self pairs included.
    """
    if start is None:
        _, start = count_labelled_states(n)
    cl = ClassList()
    counts = Counter()
    todo = [cl.index_of(start)]
    seen = set(todo)
    while todo:
        k = todo.pop()
        for other in one_switch_away(cl.reps[k][0]):
            j = cl.index_of(other)
            counts[(k, j)] += 1
            if j not in seen:
                seen.add(j)
                todo.append(j)
    classes = []
    for k, (adj, _) in enumerate(cl.reps):
        a = cl.symmetries(k)
        classes.append(dict(class_id=k, squares=int(total_squares(adj)), surplus=int(surplus(adj)), symmetries=a,
                            size=factorial(n) ** 2 // a))
    return classes, dict(counts)


def hamiltonian(classes, counts, n, lam, gamma):
    """The symmetric-sector matrix of H = H0 + H1 (see the module docstring)."""
    m = len(classes)
    N = 2 * n
    h = np.zeros((m, m))
    for c in classes:
        k = c["class_id"]
        h[k, k] = 16.0 * (N - c["squares"]) + 4.0 * lam * c["surplus"]
    for (k, j), cnt in counts.items():
        h[j, k] += -gamma * math.sqrt(classes[k]["size"] / classes[j]["size"]) * cnt
    return h


def balance_holds(classes, counts):
    """|C| n(C -> C') = |C'| n(C' -> C) for every pair: the matrix is symmetric (exact integers)."""
    for (k, j), cnt in counts.items():
        if classes[k]["size"] * cnt != classes[j]["size"] * counts.get((j, k), 0):
            return False
    return True


def ground_state(h):
    """Lowest eigenvalue, the gap above it, and the ground vector with a positive sign."""
    w, v = np.linalg.eigh(h)
    psi = v[:, 0]
    if psi.sum() < 0:
        psi = -psi
    return float(w[0]), float(w[1] - w[0]) if len(w) > 1 else float("nan"), psi


def readings(classes, psi, n):
    """<S>/N, <X>/N and the class weights in the ground state."""
    p = psi ** 2
    N = 2 * n
    s = sum(p[c["class_id"]] * c["squares"] for c in classes) / N
    x = sum(p[c["class_id"]] * c["surplus"] for c in classes) / N
    return s, x, p


def labelled_shares(classes):
    """Each class's share of all labeled wirings: the classical infinite-temperature weights, for comparison."""
    tot = sum(c["size"] for c in classes)
    return np.array([c["size"] / tot for c in classes])


def main(ns, out_dir="results"):
    for n in ns:
        classes, counts = class_space(n)
        assert balance_holds(classes, counts), "switch counts are not reversible"
        shares = labelled_shares(classes)
        name = "quantum_toy_classes_n%d" % (2 * n)
        rows = []
        for lam in LAMBDAS:
            for gamma in GAMMAS:
                e0, gap, psi = ground_state(hamiltonian(classes, counts, n, lam, gamma))
                s, x, p = readings(classes, psi, n)
                for c in classes:
                    k = c["class_id"]
                    rows.append(dict(N=2 * n, lam=lam, gamma=gamma, e0=e0, gap=gap, s_per_n=s, x_per_n=x,
                                     class_id=k, squares=c["squares"], surplus=c["surplus"], symmetries=c["symmetries"],
                                     labelled_share=shares[k], weight=p[k], min_amplitude=float(psi.min())))
        path = Path(out_dir) / (name + ".csv")
        if path.exists():
            raise SystemExit("refusing to overwrite " + str(path))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        meta = dict(script="scripts/quantum_toy_classes.py", n=n, N=2 * n, classes=len(classes),
                    switch_pairs=len(counts), lambdas=LAMBDAS, gammas=GAMMAS, package=__version__,
                    python=platform.python_version(), numpy=np.__version__,
                    preregistration="PREREGISTRATION.md section T90")
        (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("N = %d: %d classes, %d ordered switch pairs, written %s" % (2 * n, len(classes), len(counts), path))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [8, 9])
