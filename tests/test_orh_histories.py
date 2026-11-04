"""Stage C benchmarks: the decoherence functional, the sum rules, nested compatible sets, record surprisal, the
permanence of a kept record and the impermanence of an erased one, Born weights against branch counts."""
import numpy as np

from orh import histories as hist
from orh.statevector import Register, binary_entropy


def two_slit_steps():
    return [hist.step(lambda r: r.h("s"), hist.alternatives_over(["s"]), "path"),
            hist.step(lambda r: r.h("s"), hist.alternatives_over(["s"]), "output")]


def test_two_slit_family_interferes_total_weight_one_but_sum_rules_fail():
    labels, d = hist.decoherence_functional(Register(["s"]), two_slit_steps())
    p = hist.probabilities(d)
    assert np.allclose(p, 0.25)
    assert abs(hist.total_weight(d) - 1) < 1e-12                   # off-diagonal terms cancel in pairs
    assert abs(hist.max_offdiagonal(d) - 0.25) < 1e-12             # yet the family is not decoherent
    classes = {"out0": [(0, 0), (1, 0)], "out1": [(0, 1), (1, 1)]}
    v = hist.sum_rule_violations(labels, d, classes)
    assert abs(v["out0"]["fine"] - 0.5) < 1e-12 and abs(v["out0"]["coarse"] - 1.0) < 1e-12
    assert abs(v["out1"]["fine"] - 0.5) < 1e-12 and abs(v["out1"]["coarse"] - 0.0) < 1e-12
    assert not hist.is_decoherent(d)


def test_a_kept_record_makes_the_same_family_exactly_decoherent():
    steps = [hist.step(lambda r: r.h("s").cnot("s", "e"), hist.alternatives_over(["s"]), "path"),
             hist.step(lambda r: r.h("s"), hist.alternatives_over(["s"]), "output")]
    labels, d = hist.decoherence_functional(Register(["s", "e"]), steps)
    assert hist.is_decoherent(d)
    assert np.allclose(hist.probabilities(d), 0.25)
    classes = {"out0": [(0, 0), (1, 0)], "out1": [(0, 1), (1, 1)]}
    for v in hist.sum_rule_violations(labels, d, classes).values():
        assert abs(v["violation"]) < 1e-12


def test_partial_record_gives_off_diagonal_terms_equal_to_the_environment_overlap_over_two():
    theta = 0.7
    steps = [hist.step(lambda r: r.h("s").controlled_tilt("s", "e", theta), hist.alternatives_over(["s"])),
             hist.step(lambda r: r.h("s"), hist.alternatives_over(["s"]))]
    labels, d = hist.decoherence_functional(Register(["s", "e"]), steps)
    i, j = labels.index((0, 0)), labels.index((1, 0))
    # branch (path 0, out 0) = 1/2 |0>|0>; branch (path 1, out 0) = 1/2 |0>|theta>: overlap cos(theta)/4
    assert abs(d[i, j] - np.cos(theta) / 4) < 1e-12
    assert abs(hist.max_offdiagonal(d) - np.cos(theta) / 4) < 1e-12


def test_nested_records_shrink_the_compatible_set_and_keep_the_total_at_one():
    p1, p2 = 0.3, 0.8
    steps = [hist.step(lambda r: r.prepare("s1", p1).cnot("s1", "o1"), hist.alternatives_over(["o1"]), "first"),
             hist.step(lambda r: r.prepare("s2", p2).cnot("s2", "o2"), hist.alternatives_over(["o2"]), "second")]
    labels, d = hist.decoherence_functional(Register(["s1", "s2", "o1", "o2"]), steps)
    assert hist.is_decoherent(d)
    rows = hist.nested_record_table(labels, d, [{0: 0}, {0: 0, 1: 1}])
    assert rows[0]["size"] == 2 and rows[1]["size"] == 1
    assert abs(rows[0]["mu"] - (1 - p1)) < 1e-12
    assert abs(rows[1]["mu"] - (1 - p1) * p2) < 1e-12
    assert rows[1]["mu"] <= rows[0]["mu"] and rows[1]["nested"]
    assert rows[1]["surprisal_bits"] > rows[0]["surprisal_bits"]
    assert abs(rows[0]["surprisal_bits"] + np.log2(1 - p1)) < 1e-12
    for row in rows:
        assert abs(row["conditional_total"] - 1) < 1e-12
        assert abs(row["total_weight"] - 1) < 1e-12
        assert row["family_decoherent"]


def test_two_observers_of_one_record_have_a_joint_measure_not_a_product():
    p = 0.3
    steps = [hist.step(lambda r: r.prepare("s", p).cnot("s", "oa"), hist.alternatives_over(["oa"])),
             hist.step(lambda r: r.cnot("s", "ob"), hist.alternatives_over(["ob"]))]
    labels, d = hist.decoherence_functional(Register(["s", "oa", "ob"]), steps)
    assert hist.is_decoherent(d)
    mu_a = hist.measure(labels, d, hist.compatible(labels, {0: 0}))
    mu_ab = hist.measure(labels, d, hist.compatible(labels, {0: 0, 1: 0}))
    mu_mixed = hist.measure(labels, d, hist.compatible(labels, {0: 0, 1: 1}))
    assert abs(mu_a - (1 - p)) < 1e-12
    assert abs(mu_ab - mu_a) < 1e-12                 # B's agreeing record adds no surprisal
    assert mu_mixed == 0.0                           # A says 0 and B says 1 never happens
    assert abs(mu_ab - mu_a * mu_a) > 0.1            # the product of marginals would be wrong


def test_an_erased_record_is_not_a_permanent_fact_about_the_past():
    steps = [hist.step(lambda r: r.h("s").cnot("s", "e"), hist.alternatives_over(["e"]), "record"),
             hist.step(lambda r: r.cnot("s", "e").h("s"), hist.alternatives_over(["s"]), "output after erasure")]
    labels, d = hist.decoherence_functional(Register(["s", "e"]), steps)
    i, j = labels.index((0, 0)), labels.index((1, 0))
    assert abs(d[i, j] - 0.25) < 1e-12               # the two record values interfere once the record is undone
    assert not hist.is_decoherent(d)
    # the same world with the record kept (no erasure) decoheres exactly
    kept = [hist.step(lambda r: r.h("s").cnot("s", "e"), hist.alternatives_over(["e"]), "record"),
            hist.step(lambda r: r.h("s"), hist.alternatives_over(["s"]), "output")]
    _, d_kept = hist.decoherence_functional(Register(["s", "e"]), kept)
    assert hist.is_decoherent(d_kept)


def test_born_weights_are_not_branch_counts_and_survive_duplication_of_a_register():
    p = 0.9
    one = [hist.step(lambda r: r.prepare("s", p).cnot("s", "o"), hist.alternatives_over(["o"]))]
    labels1, d1 = hist.decoherence_functional(Register(["s", "o", "o2"]), one)
    assert np.allclose(hist.probabilities(d1), [1 - p, p])      # two histories, not 1/2 each
    two = one + [hist.step(lambda r: r.cnot("o", "o2"), hist.alternatives_over(["o2"]))]
    labels2, d2 = hist.decoherence_functional(Register(["s", "o", "o2"]), two)
    p2 = hist.probabilities(d2)
    weight_s1 = sum(p2[labels2.index(a)] for a in hist.compatible(labels2, {0: 1}))
    assert abs(weight_s1 - p) < 1e-12                           # the duplicated register changes no weight
    assert hist.measure(labels2, d2, hist.compatible(labels2, {0: 1, 1: 0})) == 0.0
    assert sum(1 for x in p2 if x > 1e-12) == 2                 # and no new histories carry weight
