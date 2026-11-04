"""The T65 end-graph reader (reported, not scored): a flat torus is all open, one region, no pieces."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from graphity.cqg_d import torus                 # noqa: E402
import read_t65_pieces as rp                      # noqa: E402


def test_flat_and_slab(tmp_path):
    adj, part = torus([6, 6, 6])
    np.savez(tmp_path / "flat.npz", adj=adj, part=part, lam=1.25, h0=0.0, e0=0.0)
    r = rp.read_graph(tmp_path / "flat.npz")
    assert r["open"] == 216 and r["curled"] == 0 and r["damaged"] == 0
    assert r["open_regions"] == [216] and r["damaged_pieces"] == [] and r["curled_pieces"] == []
    adj, part = torus([4, 8, 8])
    np.savez(tmp_path / "slab.npz", adj=adj, part=part, lam=1.25, h0=0.0, e0=0.0)
    r = rp.read_graph(tmp_path / "slab.npz")
    assert r["open"] == 0 and r["curled"] == 256 and r["curled_pieces"] == [256]
