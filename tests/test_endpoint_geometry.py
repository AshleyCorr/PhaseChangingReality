"""Known surfaces, non-surfaces and relabelling controls for O122."""
import numpy as np
import pytest
from scripts.read_endpoint_geometry import inspect, topology, validate


def grid(width, length, shift=0, reflect=False):
    edges = [set() for _ in range(width*length)]
    def add(u, v):
        edges[u].add(v)
        edges[v].add(u)
    for x in range(width):
        for y in range(length):
            u = x*length+y
            add(u, x*length+(y+1) % length)
            if x+1 < width:
                add(u, (x+1)*length+y)
            else:
                add(u, ((-y if reflect else y)+shift) % length)
    return np.array([sorted(row) for row in edges], dtype=np.int64)


@pytest.mark.parametrize('width,length,shift,short,long', [
    (6, 6, 0, 6, 6), (8, 12, 0, 8, 12), (4, 16, 2, 6, 16),
    (4, 24, 2, 6, 24), (4, 48, 2, 6, 48)])
def test_known_tori_and_relabellings(width, length, shift, short, long):
    adj = grid(width, length, shift)
    perm = np.random.default_rng(118).permutation(len(adj))
    relabelled = np.empty_like(adj)
    relabelled[perm] = perm[adj]
    for graph in (adj, relabelled):
        result = inspect(graph)
        assert result['kind'] == 'torus'
        assert result['systole'] == short
        assert result['independent'] == long


def test_klein_bottle_is_not_a_torus():
    result = inspect(grid(6, 8, reflect=True))
    assert result['kind'] == 'Klein bottle'
    assert result['chi'] == 0
    assert 'systole' not in result


def test_disconnected_tori_are_not_one_torus():
    adj = grid(6, 8)
    result = inspect(np.concatenate([adj, adj+len(adj)]))
    assert result['kind'] == 'disconnected surface'
    assert result['components'] == [48, 48]


def test_curled_tube_is_not_a_square_surface():
    with pytest.raises(ValueError, match='faces per edge'):
        topology(grid(4, 16))


def test_invalid_endpoint_is_rejected():
    adj = grid(6, 8)
    adj[0, 0] = len(adj)
    assert not validate(adj)
    with pytest.raises(ValueError, match='invalid ensemble'):
        inspect(adj)
