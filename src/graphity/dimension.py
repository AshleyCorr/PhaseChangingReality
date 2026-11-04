"""The local square-count signature of every vertex, and the connected pieces of a set of vertices.

The legacy name local_dimension denotes a square-count signature, not a certified
geometric dimension. In a valid four-regular hard-core graph, d(v)=6-q(v), where
q(v) counts incident squares. It is 2 on a flat square torus, 1 on a 4 x L torus
(even L > 4), and 0 on a 4-cube, but the converses do not hold. In particular,
d=2 everywhere does not establish a surface or torus. Connected components of
selected d values describe where that signature occurs, not a unique moving front.
See ASSUMPTIONS O108 and docs/papers/curled_torus/supplement.md.

Two measured dimensions, added 23 October 2026 for T80 (a draft in PREREGISTRATION.md): spectral_dimension (how
fast a random walker stops coming back) and volume_growth (how fast the number of points within r steps grows).
Unlike d(v), which is read off one point's squares, these are read from the whole graph and are functions of scale.
They are the estimators of [Carlip17] Sec. 2.1. A third, heat_kernel_dimension, is the continuous-time definition in
[T22] Eqs. (15) to (17), whose text calls t "diffusion steps". Their tests use built graphs only (a ring, square-lattice
tori, the tube and the 4-cube as built, the cubic-lattice torus, random 4-regular graphs, bipartite and not); under
T80 nothing is read on the toy's saved states before its gate passes and the owner's predictions are written.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _share_a_neighbour_other_than(adj, a, b, v):
    """Do a and b have a common neighbour that is not v? (That is, does the pair (v-a, v-b)
    close a square v - a - c - b - v.)"""
    for i in range(4):
        c = adj[a, i]
        if c < 0 or c == v:
            continue
        for j in range(4):
            if adj[b, j] == c:
                return True
    return False


@njit(cache=True)
def local_dimension(adj):
    """d[v] = number of the six edge pairs at v that close no square. Sheet 2, tube 1, cube 0."""
    n = adj.shape[0]
    d = np.zeros(n, dtype=np.int64)
    for v in range(n):
        open_pairs = 0
        for i in range(4):
            a = adj[v, i]
            if a < 0:
                continue
            for j in range(i + 1, 4):
                b = adj[v, j]
                if b < 0:
                    continue
                if not _share_a_neighbour_other_than(adj, a, b, v):
                    open_pairs += 1
        d[v] = open_pairs
    return d


@njit(cache=True)
def _share_d(adj, a, b, v):
    """_share_a_neighbour_other_than at any number of links (the slots are read from the array)."""
    deg = adj.shape[1]
    for i in range(deg):
        c = adj[a, i]
        if c < 0 or c == v:
            continue
        for j in range(deg):
            if adj[b, j] == c:
                return True
    return False


@njit(cache=True)
def local_dimension_d(adj):
    """d[v] at any number of links: the edge pairs at v that close no square, counted over all pairs.

    With four links this is local_dimension (tested). With six, on the cubic lattice, the twelve mixed-axis
    pairs each close a square and the three opposite pairs (+x, -x) close one only along a direction curled
    to length 4, so d is 3 on the flat 3-torus, 2 with one direction curled, 1 with two, 0 in the 6-cube
    (VISION Update 22; ASSUMPTIONS O41). A slot holding -1 (an edge removed mid-move) is skipped.
    """
    n, deg = adj.shape
    d = np.zeros(n, dtype=np.int64)
    for v in range(n):
        open_pairs = 0
        for i in range(deg):
            a = adj[v, i]
            if a < 0:
                continue
            for j in range(i + 1, deg):
                b = adj[v, j]
                if b < 0:
                    continue
                if not _share_d(adj, a, b, v):
                    open_pairs += 1
        d[v] = open_pairs
    return d


def piece_labels(adj, members):
    """Label the connected pieces of the subgraph induced on `members` (a boolean mask).

    Returns an int array over vertices: -1 outside `members`, else the piece's index. The pieces are the ones
    `pieces_of` sizes, numbered in order of their lowest vertex. Added 2026-09-25 for T37, which counts separate
    seeds by asking whether a piece touches any vertex already counted.
    """
    members = np.asarray(members, dtype=bool)
    n = adj.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    k = 0
    for start in range(n):
        if not members[start] or labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = k
        while stack:
            v = stack.pop()
            for j in range(adj.shape[1]):
                w = adj[v, j]
                if w >= 0 and members[w] and labels[w] < 0:
                    labels[w] = k
                    stack.append(w)
        k += 1
    return labels


def new_seeds(labels, counted, min_size):
    """T37's seed count at one look. `labels` from piece_labels; `counted` a boolean mask of the vertices in pieces
    already counted. A piece with at least `min_size` vertices and no vertex in `counted` is a new seed. Returns
    (the number of new seeds, the updated mask: every vertex in a piece of at least `min_size`, plus the old mask).
    A piece that splits, or one that merges with another, touches counted vertices and is not counted again."""
    n_new = 0
    counted = counted.copy()
    if labels.max() < 0:
        return 0, counted
    sizes = np.bincount(labels[labels >= 0])
    big = np.flatnonzero(sizes >= min_size)
    for k in big:
        members = labels == k
        if not counted[members].any():
            n_new += 1
    keep = np.isin(labels, big)
    counted |= keep
    return n_new, counted


def pieces_of(adj, members):
    """Connected pieces of the subgraph induced on `members` (a boolean mask over vertices).

    Returns the sizes of the pieces, largest first. Used for T7 observable 3: the vertices that
    carry the selected square-count signature, and how connected that set is.
    It does not identify a unique front or certify sheet geometry.
    """
    members = np.asarray(members, dtype=bool)
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes = []
    for start in range(n):
        if not members[start] or seen[start]:
            continue
        stack, count = [start], 0
        seen[start] = True
        while stack:
            v = stack.pop()
            count += 1
            for k in range(adj.shape[1]):              # any number of links (six-link work, 2026-09-25)
                w = adj[v, k]
                if w >= 0 and members[w] and not seen[w]:
                    seen[w] = True
                    stack.append(w)
        sizes.append(count)
    return sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------------------------------------------
# Measured dimension at each scale (T80, draft). Built graphs only until T80 is registered.
# ---------------------------------------------------------------------------------------------------------------

UNIT = 1e-9     # an eigenvalue of W this close to +1 or -1 counts as exactly +1 or -1 (see spectral_dimension)


def _complete(adj):
    """The (N, k) neighbour array as contiguous int64, after the checks that make it a simple graph. Every function
    below refuses what this refuses, with a ValueError:
      - whole-number entries: an array of another type (floats, say) is refused, not cast, because a cast would
        turn 5.7 into 5 without a word;
      - no empty slot (-1): a graph caught mid-move is not a graph;
      - every entry names a point, 0 <= v < N (the compiled search reads memory unchecked, so this must come first);
      - every link is listed from both ends equally often: v appears in u's row as many times as u in v's;
      - no point lists itself (a self-loop passes the check above);
      - no point lists another twice (a double link listed twice from both ends passes it too).
    Two results below assume a simple graph: the first-step identity of spectral_dimension and the heat kernel's
    K'(0) = -N.
    """
    adj = np.asarray(adj)
    if not np.issubdtype(adj.dtype, np.integer):
        raise ValueError("the neighbour array must hold whole numbers (an integer type), not " + str(adj.dtype))
    adj = np.ascontiguousarray(adj, dtype=np.int64)
    n, k = adj.shape
    if (adj < 0).any():
        raise ValueError("an empty slot (-1): these measurements need a complete graph, not one caught mid-move")
    if (adj >= n).any():
        raise ValueError("a neighbour index past the last point: every entry must be between 0 and N - 1")
    u, v = np.repeat(np.arange(n), k), adj.ravel()
    if not np.array_equal(np.sort(u * n + v), np.sort(v * n + u)):          # each link (u, v) against (v, u)
        raise ValueError("the neighbour array is not symmetric: some u lists v more or less often than v lists u")
    if (adj == np.arange(n)[:, None]).any():
        raise ValueError("a point lists itself (a self-loop): these measurements need a simple graph")
    rows = np.sort(adj, axis=1)
    if (rows[:, 1:] == rows[:, :-1]).any():
        raise ValueError("a point lists another twice (a double link): these measurements need a simple graph")
    return adj


def _walk_eigenvalues(adj):
    """The N eigenvalues of the one-step matrix W = A / k of a k-regular graph, from the (N, k) neighbour array.

    W[u, v] is the chance that one step of a random walk goes from u to v: the number of links between them,
    divided by k. Every point has k links and every link is listed from both ends (checked), so W is symmetric,
    and one symmetric eigendecomposition gives all its eigenvalues, each between -1 and +1.
    """
    adj = _complete(adj)
    n, k = adj.shape
    w = np.zeros((n, n))
    np.add.at(w, (np.repeat(np.arange(n), k), adj.ravel()), 1.0 / k)
    return np.linalg.eigvalsh(w)


def spectral_dimension(adj, t_max):
    """The spectral dimension D_s(t): how fast a random walker stops coming back, at each walk length t.

    A walker starts at a point and takes t steps, each along one of its point's k links, chosen with equal chance.
    P(t) is the chance that it is back at its start after step t, averaged over every start point. On a
    d-dimensional lattice P(t) falls like t^(-d/2), so
        D_s(t) = -2 d ln P / d ln t
    reads d off the fall, one scale at a time ([Carlip17] Sec. 2.1, Eq. (2.7); reading note
    docs/reading/notes/2026-10-23_carlip_dimensional_reduction.md).

    How P(t) is computed: exactly, with no sampling. With W = A / k the one-step matrix, the chance of being back at
    u after t steps is (W^t)[u, u], so the average over starts is
        P(t) = (1/N) Tr(W^t) = (1/N) sum_i mu_i^t,
    where mu_i are the N eigenvalues of W, all from one symmetric eigendecomposition. Time and memory grow as N^3
    and N^2: about 0.5 s at N = 1,152, 5 s at N = 2,304 and 12 s at N = 4,096 on the owner's laptop (measured
    23 October 2026). That covers the saved states T80 names, as far as their configs show (the largest found is
    T53's 4 x 4 x 72, N = 1,152; checked 23 October 2026). There is no sampled (Monte Carlo) route, although
    the draft named one (walks, seed): a sampled P(t) is noisy, and the slope between successive times divides that
    noise by ln((t + 2) / t), which is small, so a sampled slope would need its own fitting window and its own
    tests. It waits until a state is too large for the exact route.

    Even times only. A bipartite graph (every graph of the toy) can only bring a walker back after an even number
    of steps, so P(t) = 0 at odd t. For every graph, only t = 2, 4, ..., t_max are used, and D_s is read between
    successive even times, placed at their geometric mean:
        D_s at t_mid = sqrt(t (t + 2))  is  -2 [ln P(t + 2) - ln P(t)] / [ln(t + 2) - ln t].

    Even a perfect lattice reaches its dimension only slowly. On the square lattice P(t) = [C(t, t/2) / 2^t]^2
    (tested), and D_s is 1.66 between t = 2 and 4, 1.91 at t_mid = 11 and 1.95 at t_mid = 21 (exact). So a
    short-scale reading is to be compared with the perfect lattice's own value at the same t, not with the
    whole number d. And the first step (t = 2 to 4) depends only on the degree k and the mean number of squares per
    point, q = 4S/N, for any simple k-regular graph, D_s(sqrt 8) = -2 log2[(2k^2 - k + 2q) / k^3], because a closed
    walk of four steps either goes once round a square or retraces itself (exact; tested on the flat torus, the
    tube and the 4-cube), so that reading restates the square count and adds nothing to it.

    The floor: what the finite size does. On a finite graph the walker forgets where it started. The eigenvalue
    +1 (once for each connected piece) and -1 (once for each bipartite piece) give mu^t = 1 at every even t, so
    P(t) never falls below
        floor = (number of eigenvalues equal to +1 or -1) / N,
    which is 2/N for one connected bipartite graph and 1/N for one connected graph that is not bipartite. As P(t)
    nears the floor its slope falls to zero, so D_s(t) drops toward 0 once t is comparable to the time a walker
    needs to cross the graph (of order L^2 steps on an L x L torus). That drop is the size of the graph, not the
    dimension of the space at that scale. Both readings are returned and the window is left to the caller, to be
    fixed before any result is read:
      - D_s, from P(t) itself: the one to read. On a torus, P(t) equals the infinite lattice's value exactly
        while t is shorter than the shortest way around (tested), so nothing has been taken out that was not
        there.
      - D_s_above_floor, from P(t) - floor, computed from the other eigenvalues directly (never by subtracting
        two nearly equal numbers). At scales well below the size it removes a share the walker has not yet had
        time to feel, so it reads high; at long t it decays exponentially, so its slope rises without bound,
        which is the size again, pointing the other way. It is a warning light, not a reading.
    On the 32 x 32 torus (exact, tested): D_s stays within 0.05 of 2 from t_mid = 20 to 128, while the floor
    grows from 6.3 % of P(t) at t = 20 to 39.4 % at t = 128 (the two ends of that window), and D_s falls toward 0
    beyond; D_s_above_floor already reads 2.09 to 2.46 for t_mid from 20 to 64. The share floor / P(t) says how
    much of P(t) the floor is, and a cut on that share, written down in advance, is a simple way to fix the window.

    An eigenvalue counts as +1 or -1 when it is within UNIT = 1e-9 of it. Rounding in the eigendecomposition is
    about 1e-15. On a ring of N points the eigenvalue nearest +1 is cos(2 pi / N), about 2 pi^2 / N^2 below it,
    which is more than 1e-9 for any N the exact route can handle (exact, from the ring's spectrum).

    Returns a dict: t (the even times), P, floor, P_above_floor, t_mid, D_s, D_s_above_floor. At long t,
    P_above_floor underflows to exactly zero (on the 4-cube from t = 1,074), and wherever either end of a pair is
    zero, D_s_above_floor is set to nan (without that, the first such pair would read +inf). The plain D_s never
    meets this: P(t) is at least the floor, which is above zero.
    """
    if t_max < 4:
        raise ValueError("t_max must be at least 4: a slope needs two even times")
    mu = _walk_eigenvalues(adj)
    n = len(mu)
    unit = np.abs(mu) > 1 - UNIT
    t = np.arange(2, t_max + 1, 2)
    p = (mu[None, :] ** t[:, None]).sum(axis=1) / n
    p_above = (mu[None, ~unit] ** t[:, None]).sum(axis=1) / n
    t_mid = np.sqrt(t[:-1] * t[1:])
    dlnt = np.log(t[1:] / t[:-1])
    d_s = -2 * np.diff(np.log(p)) / dlnt
    with np.errstate(divide="ignore", invalid="ignore"):
        d_s_above = -2 * np.diff(np.log(p_above)) / dlnt
    d_s_above[~np.isfinite(d_s_above)] = np.nan                       # a pair that reaches zero has no slope
    return dict(t=t, P=p, floor=unit.sum() / n, P_above_floor=p_above,
                t_mid=t_mid, D_s=d_s, D_s_above_floor=d_s_above)


def heat_kernel_dimension(adj, t):
    """The published model's spectral function d_s(t): the same question as spectral_dimension, asked of a walker
    that diffuses in continuous time instead of taking whole steps.

    The definitions are [T22]'s (section "Cartan-Hadamard manifolds in 2D", Eqs. (15) to (17); arXiv:2112.03778,
    text searched 23 October 2026): the heat kernel K(t) = tr exp(t Delta) (15), with the normalized Laplacian
    Delta = -I + A / (2D) (16), where 2D is the number of links per point, and d_s(t) = -2 d ln K / d ln t (17).
    With W = A / k as in spectral_dimension (k = 2D links), Delta = W - I, so from the eigenvalues mu_i of W,
        K(t) = sum_i exp(t (mu_i - 1)),    K'(t) = sum_i (mu_i - 1) exp(t (mu_i - 1)),
        d_s(t) = -2 t K'(t) / K(t),
    computed exactly, with no finite difference. K is [T22]'s trace, not divided by N; d_s does not depend on that.
    Eigenvalues within UNIT of +1 are set to exactly +1 (in spectral_dimension they are only counted in the floor).

    Where it differs from the discrete walk of spectral_dimension (exact, from the formulas):
      - At short t. K(0) = N and, with no link from a point to itself (none in the toy; the input check refuses
        one), K'(0) = -N, so d_s(t) is about 2t as t -> 0: it starts at 0 on every graph. On the square lattice
        d_s(t) = 2t [1 - I_1(t/2) / I_0(t/2)] exactly (I_0, I_1 the modified Bessel functions; tested on the
        48 x 48 torus): it overshoots to 2.44 near t = 3.4 and comes back to 2 from above, as about 2 + 1/t, where
        the discrete walk comes up to 2 from below.
      - In the floor. The eigenvalue -1 of a bipartite graph gives exp(-2t), which dies away, so K(t) / N falls
        to (number of connected pieces) / N: 1/N for one connected graph, bipartite or not, where the discrete
        walk's floor is 2/N on a bipartite graph. Past the time to cross the graph d_s falls to 0.

    What [T22] reports (published; text of arXiv:2112.03778 searched 23 October 2026). Of the ordered phase in its
    Fig. 5 (the 2D ground-state graphs at N = 2000, which it takes to discretize negative-curvature manifolds, so
    not flat lattices): "For small values of the diffusion time t we see a shape increasing to a maximum and then
    decreasing, exactly like in the random phase but on a much smaller scale. As discussed in the previous section,
    these are the discreteness effects below the Planck scale. At the largest diffusion times t the curve ds(t)
    decreases to zero: this is due to the finite-size effects." Its text
    introduces K(t) as "encoding the return probabilities after t diffusion steps on the graph" (just before
    Eq. (15)), so it does not settle whether the numbers of Fig. 5 came from the exponential of Eq. (15) or from
    whole steps, and the two approach the lattice value from opposite sides (above). Ours, unverified and not
    tested: whether any part of that published short-t rise and fall is the estimator's own overshoot, as on the
    perfect lattice above, rather than a property of the graphs.

    t: an array of diffusion times, each above 0. Returns a dict: t, K, d_s.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if t.size == 0 or not (np.isfinite(t) & (t > 0)).all():
        raise ValueError("t must be a non-empty array of finite times above 0")
    delta = _walk_eigenvalues(adj) - 1                                 # eigenvalues of Delta, between -2 and 0
    delta[np.abs(delta) < UNIT] = 0.0
    e = np.exp(t[:, None] * delta[None, :])                            # every exponent is at most 0: no overflow
    k_t = e.sum(axis=1)
    d_s = -2 * t * (e * delta[None, :]).sum(axis=1) / k_t
    return dict(t=t, K=k_t, d_s=d_s)


@njit(cache=True)
def _sphere_sizes(adj, r_max):
    """counts[r] = the number of (start, point) pairs at graph distance exactly r, summed over every start point,
    for r = 0 .. r_max. One breadth-first search from each point, stopped at distance r_max."""
    n, k = adj.shape
    counts = np.zeros(r_max + 1, dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    for start in range(n):
        dist[start] = 0
        queue[0] = start
        head, tail = 0, 1
        while head < tail:
            v = queue[head]
            head += 1
            counts[dist[v]] += 1
            if dist[v] == r_max:
                continue
            for j in range(k):
                w = adj[v, j]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue[tail] = w
                    tail += 1
        for i in range(tail):                 # forget only the points this search reached
            dist[queue[i]] = -1
    return counts


def volume_growth(adj, r_max):
    """The volume-growth dimension D_v(r): how fast the number of points within r steps grows, at each r.

    V(r) is the number of points at graph distance at most r from a start point (the start included, so
    V(0) = 1), averaged over every start point. The distances come from a breadth-first search from every point,
    so V(r) is exact. On a d-dimensional lattice V(r) grows like r^d, so
        D_v(r) = d ln V / d ln r
    reads d off the growth ([Carlip17] Sec. 2.1, "Scaling": the Hausdorff dimension; reading note as for
    spectral_dimension).

    The finite difference is the one spectral_dimension uses: forward, between successive radii, placed at their
    geometric mean,
        D_v at r_mid = sqrt(r (r + 1))  is  [ln V(r + 1) - ln V(r)] / [ln(r + 1) - ln r],   r = 1, ..., r_max - 1.
    (r = 0 is left out, because ln 0 is not defined.)

    What the finite size does. Once r reaches the graph's diameter, V(r) = N and D_v = 0. On an L x L torus the
    ball is the lattice's diamond, V(r) = 2r^2 + 2r + 1, exactly while r <= L/2 - 1; at r = L/2 it starts to
    overlap itself (tested). Even on the infinite square lattice the slope reaches 2 only slowly: the local slope
    of 2r^2 + 2r + 1 is 2 - (2r + 2) / (2r^2 + 2r + 1), about 2 - 1/r (exact, from the formula), so D_v reads
    below the lattice's dimension at small r.

    The first step (r = 1 to 2) restates the square count, as D_s's first step does. On a simple k-regular graph
    with no triangles in which no two points share more than two neighbours (every valid state of the toy, at four
    links or six: it is bipartite and obeys the hard-core rule, ASSUMPTIONS Q2 and Q21), V(1) = k + 1 and
    V(2) = k^2 + 1 - q, with q = 4S/N the mean number of squares per point, so
        D_v(sqrt 2) = log2[(k^2 + 1 - q) / (k + 1)]
    (exact; tested on the flat torus, the tube and the 4-cube as built). The k(k - 1) two-step walks that do not
    turn back all end at distance 2 when there is no triangle, and a point reached by c of them closes
    c(c - 1)/2 squares with the start, which is c - 1 only while c <= 2. So, unlike the D_s identity, which holds
    on any simple k-regular graph, this one needs both conditions.

    Cost: at most N searches of N k steps each; well under a second at N of a few thousand (compiled with numba).

    Returns a dict: r (0 to r_max), V, r_mid, D_v.
    """
    if r_max < 2:
        raise ValueError("r_max must be at least 2: a slope needs two radii above 0")
    adj = _complete(adj)
    n = adj.shape[0]
    r = np.arange(r_max + 1)
    v = np.cumsum(_sphere_sizes(adj, r_max)) / n
    r_mid = np.sqrt(r[1:-1] * r[2:])
    d_v = np.diff(np.log(v[1:])) / np.log(r[2:] / r[1:-1])
    return dict(r=r, V=v, r_mid=r_mid, D_v=d_v)
