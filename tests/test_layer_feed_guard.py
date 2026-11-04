"""The affordability guard of the layer form (ASSUMPTIONS O158 (b); fixed 27 October 2026): a move that costs energy is
paid only when its pool holds the whole bill, so no store ever goes below zero. Tests at a pool just below, at and
above the bill, on the helper and on a real proposal; and the ledger's identity still holds.

On the 4 x 6 x 6 slab at lambda = 1.25 the split along the curled axis is the downhill opening (it releases a N_slab =
144) and the contraction back is the paid move (its bill is 144)."""
import numpy as np

from graphity import contraction as C
from graphity import layer_feed as LF

LAM = 1.25


def _opened_and_bill():
    """The flat slab, its split state along axis 0, and the bill of contracting back (positive at lambda > 1)."""
    st = C.build((4, 6, 6))
    split = C.split_layers(st, 0, 0, 8)
    assert split is not None
    f_split = LF.energy(split, LAM, 0.0)
    back = C.contract_layers(split, 0, 0)
    assert back is not None
    bill = LF.energy(back, LAM, 0.0) - f_split
    assert bill > 0
    return st, split, f_split, bill


def test_can_pay_at_the_bill_and_a_hair_either_side():
    bill = 9.0
    assert not LF.can_pay(bill - 1e-13, bill)      # the old guard (pool + 1e-12 >= bill) let this pay
    assert LF.can_pay(bill, bill)
    assert LF.can_pay(bill + 1e-13, bill)
    assert not LF.can_pay(0.0, 1e-300) and LF.can_pay(0.0, 0.0)


def test_a_paid_move_refuses_a_pool_a_hair_short_pays_the_exact_bill_and_never_leaves_a_negative_store():
    st, split, f_split, bill = _opened_and_bill()
    mask = LF.pool_mask(split, 0, 0, True)
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(0)
    proposal = (0, 0, True)
    # just below: one point of the pool holds the whole bill less a hair, the rest nothing
    split.stores[:] = 0.0
    split.stores[idx[0]] = bill - 1e-13
    assert float(split.stores[mask].sum()) < bill
    new, f_new, kind, rel = LF.propose_layer_move(split, f_split, LAM, 0.0, rng, (0,), 8, 0.0, "daughter", proposal)
    assert kind is None and new is split and f_new == f_split
    # at the bill: pays, the store goes exactly to zero, nothing negative anywhere
    split.stores[:] = 0.0
    split.stores[idx[0]] = bill
    new, f_new, kind, rel = LF.propose_layer_move(split, f_split, LAM, 0.0, rng, (0,), 8, 0.0, "daughter", proposal)
    assert kind == "contract" and abs(f_new - f_split - bill) < 1e-9 and rel == 0.0
    assert float(new.stores.min()) >= 0.0 and float(new.stores.sum()) < 1e-9
    # above the bill: pays, the surplus stays, nothing negative
    split.stores[:] = 0.0
    split.stores[idx[0]] = bill + 1.0
    new, f_new, kind, rel = LF.propose_layer_move(split, f_split, LAM, 0.0, rng, (0,), 8, 0.0, "daughter", proposal)
    assert kind == "contract" and float(new.stores.min()) >= 0.0 and abs(float(new.stores.sum()) - 1.0) < 1e-9
    # the bill shared over the pool: whichever side of the bill the float sum falls, no store ends negative
    split.stores[:] = 0.0
    split.stores[mask] = bill / len(idx)
    pool = float(split.stores[mask].sum())
    new, f_new, kind, rel = LF.propose_layer_move(split, f_split, LAM, 0.0, rng, (0,), 8, 0.0, "daughter", proposal)
    if LF.can_pay(pool, bill):
        assert kind == "contract" and float(new.stores.min()) >= 0.0
    else:
        assert kind is None


def test_twins_pay_the_second_twin_only_from_a_pool_that_holds_it():
    """Under "twins" a downhill move draws its release from the pool for the second twin; a pool a hair short refuses,
    the exact pool pays and leaves nothing negative. The downhill move is the flat slab's split along the curled axis."""
    st, split, f_split, bill = _opened_and_bill()
    f = LF.energy(st, LAM, 0.0)
    release = f - f_split
    assert abs(release - bill) < 1e-9 and release > 0
    mask = LF.pool_mask(st, 0, 0, False)
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(1)
    st.stores[:] = 0.0
    st.stores[idx[0]] = release - 1e-13
    new, f_new, kind, rel = LF.propose_layer_move(st, f, LAM, 0.0, rng, (0,), 8, 0.0, "twins", (0, 0, False))
    assert kind is None
    st.stores[idx[0]] = release
    new, f_new, kind, rel = LF.propose_layer_move(st, f, LAM, 0.0, rng, (0,), 8, 0.0, "twins", (0, 0, False))
    assert kind == "split" and abs(rel - release) < 1e-9 and float(new.stores.min()) >= 0.0
    assert float(new.stores.sum()) < 1e-9


def test_the_identity_holds_and_no_final_store_is_negative_under_the_fixed_guard():
    st = C.build((4, 6, 6))
    for seed in range(3):
        for release_to in ("daughter", "twins"):
            final, rows, summary = LF.run_layer_feed(st, LAM, 1.0 / 16, 160, seed, gate=0.0, move_set="model",
                                                     max_width=8, log_every=1, release_to=release_to)
            assert float(final.stores.min()) >= 0.0
            for res, row in zip(LF.conservation_residuals(summary["f0"], rows), rows):
                assert abs(res) < 1e-9 * max(1.0, row[5])
