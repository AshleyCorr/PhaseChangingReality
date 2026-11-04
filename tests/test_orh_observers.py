"""Stage D benchmarks: interacting observer registers in one unitary world (orh.observers): shared records agree
exactly, told records correlate, forgotten records leave the observer but not the world, a register that is not in a
low-entropy state cannot hold a record, and the forward circuit run backwards unwrites every record."""
import numpy as np

from orh import darwinism as dw
from orh import histories as hist
from orh.observers import World, disagreement
from orh.statevector import binary_entropy, maximally_mixed_pair


def test_two_observers_reading_one_error_free_record_never_disagree():
    w = World(["s", "e", "oa", "ob"]).prepare("s", 0.3).tilt_record("s", "e", np.pi / 2)
    w.observe("A", "e", "oa").observe("B", "e", "ob")
    joint = w.joint({"A": ["oa"], "B": ["ob"]})
    assert set(joint) == {((0,), (0,)), ((1,), (1,))}
    assert abs(joint[((0,), (0,))] - 0.7) < 1e-12 and abs(joint[((1,), (1,))] - 0.3) < 1e-12
    assert disagreement(joint) == 0.0
    product = w.product_of_marginals({"A": ["oa"], "B": ["ob"]})
    assert abs(product[((0,), (1,))] - 0.21) < 1e-12            # what multiplying marginals would have said
    assert abs(w.record_mutual_information(["oa"], ["ob"]) - binary_entropy(0.3)) < 1e-12


def test_imperfect_fragments_disagree_at_the_closed_form_rate():
    ti, tj, p = 0.5, 1.0, 0.4
    w = World(["s", "e0", "e1", "oa", "ob"]).prepare("s", p)
    w.tilt_record("s", "e0", ti).tilt_record("s", "e1", tj)
    w.observe("A", "e0", "oa").observe("B", "e1", "ob")
    joint = w.joint({"A": ["oa"], "B": ["ob"]})
    assert abs(disagreement(joint) - dw.disagreement_rate(p, ti, tj)) < 1e-12


def test_telling_correlates_an_observer_who_had_no_record():
    w = World(["s", "e", "oa", "ob"]).prepare("s", 0.3).tilt_record("s", "e", np.pi / 2).observe("A", "e", "oa")
    assert abs(w.reg.mutual_information(["ob"], ["s"])) < 1e-12
    w.tell("A", "oa", "B", "ob")
    assert disagreement(w.joint({"A": ["oa"], "B": ["ob"]})) == 0.0
    assert "B" in w.alive()


def test_forgetting_leaves_the_record_in_the_world_and_widens_the_observers_compatible_set():
    p = 0.3
    w = World(["s", "e", "oa", "sink"]).prepare("s", p).tilt_record("s", "e", np.pi / 2).observe("A", "e", "oa")
    # with a third copy in e, the pair (oa, s) is classically correlated: I = h(p) (it would be 2h(p) were oa the
    # only record, because then (oa, s) would be pure and entangled)
    assert abs(w.reg.mutual_information(["oa"], ["s"]) - binary_entropy(p)) < 1e-12
    w.forget("A", "oa", "sink")
    assert abs(w.reg.mutual_information(["oa"], ["s"])) < 1e-12               # the observer knows nothing now
    assert abs(w.reg.mutual_information(["sink"], ["s"]) - binary_entropy(p)) < 1e-12   # the world still does
    assert w.register_count("A") == 0 and "A" not in w.alive()
    assert w.outcome_weights("s") == (0.7, 0.3) or np.allclose(w.outcome_weights("s"), (0.7, 0.3))
    # the family of A's register value: before forgetting it split the weight; after, one value carries it all
    before = [hist.step(lambda r: r.prepare("s", p).cnot("s", "oa"), hist.alternatives_over(["oa"]))]
    after = before + [hist.step(lambda r: r.swap("oa", "sink"), hist.alternatives_over(["oa"]))]
    lb, db = hist.decoherence_functional(hist.Register(["s", "oa", "sink"]), before)
    la, da = hist.decoherence_functional(hist.Register(["s", "oa", "sink"]), after)
    assert abs(hist.measure(lb, db, hist.compatible(lb, {0: 1})) - p) < 1e-12
    assert abs(hist.measure(la, da, hist.compatible(la, {1: 0})) - 1.0) < 1e-12   # A's later register says 0 always
    assert hist.is_decoherent(da)


def test_duplicating_an_observer_changes_the_register_count_and_not_the_born_weights():
    w = World(["s", "e", "oa", "oa2"]).prepare("s", 0.3).tilt_record("s", "e", np.pi / 2).observe("A", "e", "oa")
    weights = w.outcome_weights("s")
    w.duplicate("A", "oa", "oa2", as_observer="A-twin")
    assert np.allclose(w.outcome_weights("s"), weights)
    assert sorted(w.alive()) == ["A", "A-twin"]
    joint = w.joint({"A": ["oa"], "A-twin": ["oa2"]})
    assert disagreement(joint) == 0.0


def test_a_register_that_is_not_in_a_low_entropy_state_cannot_hold_a_record():
    p = 0.3
    fresh = World(["s", "o", "q"]).prepare("s", p).observe("A", "s", "o").observe("B", "o", "q")
    assert disagreement(fresh.joint({"S": ["s"], "B": ["q"]})) == 0.0            # a fresh register copies S exactly
    mixed = World(["s", "o", "p", "q"])
    maximally_mixed_pair(mixed.reg, "o", "p")                                   # o is now maximally mixed
    mixed.prepare("s", p).observe("A", "s", "o").observe("B", "o", "q")
    rho_given = [mixed.reg.projected({"s": v}).reduced(["o"]) for v in (0, 1)]
    assert np.allclose(rho_given[0] / np.trace(rho_given[0]), np.eye(2) / 2)
    assert np.allclose(rho_given[1] / np.trace(rho_given[1]), np.eye(2) / 2)    # o looks the same either way
    assert abs(disagreement(mixed.joint({"S": ["s"], "B": ["q"]})) - 0.5) < 1e-12   # reading o is a coin toss


def test_the_forward_circuit_run_backwards_unwrites_the_records():
    w = World(["s", "e", "oa", "ob"]).prepare("s", 0.3).tilt_record("s", "e", np.pi / 2)
    w.observe("A", "e", "oa").observe("B", "oa", "ob")
    counts = [abs(w.reg.mutual_information([q], ["s"])) > 1e-9 for q in ("e", "oa", "ob")]
    assert counts == [True, True, True]
    w.reg.cnot("oa", "ob").cnot("e", "oa").controlled_tilt("s", "e", -np.pi / 2)   # the inverse, in reverse order
    for q in ("e", "oa", "ob"):
        assert abs(w.reg.mutual_information([q], ["s"])) < 1e-12
    assert np.allclose(w.reg.probabilities(["e", "oa", "ob"]).reshape(-1), [1, 0, 0, 0, 0, 0, 0, 0])


def test_the_number_of_records_grows_along_the_forward_circuit_from_the_blank_start():
    w = World(["s", "e0", "e1", "oa", "ob"]).prepare("s", 0.5)
    registers = ["e0", "e1", "oa", "ob"]

    def n_records():
        return sum(1 for q in registers if w.reg.mutual_information([q], ["s"]) > 1e-9)

    seen = [n_records()]
    w.tilt_record("s", "e0", np.pi / 2); seen.append(n_records())
    w.tilt_record("s", "e1", np.pi / 2); seen.append(n_records())
    w.observe("A", "e0", "oa"); seen.append(n_records())
    w.tell("A", "oa", "B", "ob"); seen.append(n_records())
    assert seen == [0, 1, 2, 3, 4]
