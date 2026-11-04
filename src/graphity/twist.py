"""The twisted-diagonal state: the owner's particles as loops of diagonals on the cubic torus, with twist, curl and spin,
and the readers of her rules (docs/design/twisted_diagonals_build_2026-10-30.md; the standing sheet
docs/design/twist_curl_rules_current.md). Her Option B: particles are loops of twisted diagonals built as variables; the
wiring's knots are the scrap. Nothing here is derived; the rules are put in and read back exactly.

The lattice: points of an L x L x L torus, coloured by the parity of x + y + z. A box diagonal is a signed permutation of
(1, 1, 2) (24 steps, sqrt 6, same colour); a long diagonal (1, 1, 2m), m >= 2 (holds curl, no twist); a face diagonal a
signed permutation of (1, 1, 0) (12 steps, sqrt 2, dark twist in halves). Body diagonals (1, 1, 1) change colour (8 steps,
sqrt 3) and carry the photon (her first form of 30 October, evening): not a loop but an open path along one body diagonal,
with half a curl, spin 1, no twist, moving one link per sweep, the angle of its curl turning a quarter turn per link so
that it repeats every four links. Its readers: charge 0, net twist 0, curl +1/2 or -1/2, spin +1 or -1.

A Loop: closed on the torus, one step kind, twist on each diagonal (thirds on box, halves on face, zero on long), an
non-zero integer curl (signed from 30 October, evening: the anti-particle's loop has the opposite curl, so that a
particle's curl can net zero, her neutrino), and a spin in {-1/2, 0, +1/2} that may be non-zero only if some twist is
present (her rule of 30 October: spin needs twist present, not net twist). A loop may also hold an excitation (her
decision of 30 October, evening): the photons it has absorbed, each holding its half curl and its spin 1 on the loop until
the loop re-emits a photon; the absorbed spin is carried as orbital motion (a part of its own, `orb`), not as the loop's
spin, following the published fact that an electron's spin stays 1/2 when it moves up a level. A Particle is a named set
of loops. Readers: charge = sum of net twist over box-diagonal loops only (her rule of 31 October: a charge needs two
sides and is carried only on thirds; a face loop's half twist is not charge); spin = |sum of loop spins|; mass_count = number of looped diagonals (a long
diagonal counts one); parts = (L, n, tau+, T, s, C, orb), with C including the held half curls and orb the held spins;
net_twist over a set of particles, the quantity pair creation conserves; ledger over particles and photons sums T, C, s,
orb, J = s + orb and n.

Twist values are kept as Fractions so that thirds and halves are exact.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations, product

import numpy as np

THIRD, HALF = Fraction(1, 3), Fraction(1, 2)


def _signed_perms(base):
    out = set()
    for p in permutations(base):
        for s in product((1, -1), repeat=3):
            out.add(tuple(int(a * b) for a, b in zip(p, s)))
    return out


BOX_STEPS = _signed_perms((1, 1, 2))
FACE_STEPS = _signed_perms((1, 1, 0))


def step_kind(delta):
    """'box', 'face', 'long' (a (1, 1, 2m) step with m >= 2) or None."""
    d = tuple(int(x) for x in delta)
    if d in BOX_STEPS:
        return "box"
    if d in FACE_STEPS:
        return "face"
    a = sorted(abs(x) for x in d)
    if a[0] == 1 and a[1] == 1 and a[2] >= 4 and a[2] % 2 == 0:
        return "long"
    return None


def colour(point):
    return "black" if sum(int(x) for x in point) % 2 == 0 else "white"


@dataclass
class Loop:
    """points: the loop's points in order (the step from the last back to the first closes it); steps are given as
    displacements so that the torus wrap is explicit. twists: one Fraction per step. curl: non-zero integer. spin: Fraction."""
    L: int
    start: tuple
    steps: list
    twists: list
    curl: int = 1
    spin: Fraction = Fraction(0)
    held: list = field(default_factory=list)       # the excitation: (curl, spin) of each photon absorbed, in order
    kind: str = field(init=False)
    points: list = field(init=False)

    def __post_init__(self):
        kinds = {step_kind(s) for s in self.steps}
        if None in kinds:
            raise ValueError("a step is not a box, face or long diagonal: %s" % [s for s in self.steps if step_kind(s) is None])
        if len(kinds) != 1:
            raise ValueError("a loop uses one kind of diagonal; got %s" % sorted(kinds))
        self.kind = kinds.pop()
        if len(self.twists) != len(self.steps):
            raise ValueError("one twist per diagonal")
        total = np.sum(np.array(self.steps, dtype=int), axis=0)
        if any(int(t) % self.L != 0 for t in total):
            raise ValueError("the loop does not close on the torus: total displacement %s" % (tuple(int(t) for t in total),))
        inc = {"box": THIRD, "face": HALF, "long": None}[self.kind]
        for t in self.twists:
            t = Fraction(t)
            if inc is None and t != 0:
                raise ValueError("a long diagonal holds curl and no twist")
            if inc is not None and (t / inc).denominator != 1:
                raise ValueError("twist %s is not a multiple of %s on a %s diagonal" % (t, inc, self.kind))
        self.twists = [Fraction(t) for t in self.twists]
        if int(self.curl) == 0 or int(self.curl) != self.curl:
            raise ValueError("one curl of energy per loop at least; curl must be a non-zero integer (signed from 30 October)")
        self.spin = Fraction(self.spin)
        if self.spin not in (Fraction(0), HALF, -HALF):
            raise ValueError("a loop's spin is -1/2, 0 or +1/2")
        if self.spin != 0 and all(t == 0 for t in self.twists):
            raise ValueError("spin needs twist present (her rule of 30 October)")
        self.held = [(Fraction(c), Fraction(s)) for c, s in self.held]
        for c, s in self.held:
            if not photon_curl_allowed(c) or s not in (Fraction(1), Fraction(-1)):
                raise ValueError("an excitation holds a photon's curl per link, +-2/n for a whole period n >= 4, and its spin 1; got %s" % ((c, s),))
        p = np.array(self.start, dtype=int)
        pts = [tuple(int(x) % self.L for x in p)]
        for s in self.steps[:-1]:
            p = p + np.array(s, dtype=int)
            pts.append(tuple(int(x) % self.L for x in p))
        self.points = pts

    @property
    def colour(self):
        return colour(self.start)

    @property
    def net_twist(self):
        return sum(self.twists, Fraction(0))

    @property
    def twist_present(self):
        return sum((abs(t) for t in self.twists), Fraction(0))

    @property
    def mass_count(self):
        return len(self.steps)

    @property
    def held_curl(self):
        return sum((c for c, s in self.held), Fraction(0))

    @property
    def orbital(self):
        """The spin the loop has absorbed, carried as orbital motion (her request of 30 October: represent it as its own kind)."""
        return sum((s for c, s in self.held), Fraction(0))

    @property
    def total_curl(self):
        return self.curl + self.held_curl


@dataclass
class Particle:
    name: str
    loops: list

    @property
    def charge(self):
        # her rule of 31 October: a charge needs two sides and is carried only on thirds, so face loops (halves) count 0
        return sum((lp.net_twist for lp in self.loops if loop_kind(lp) == "box"), Fraction(0))

    @property
    def spin(self):
        return abs(sum((lp.spin for lp in self.loops), Fraction(0)))

    @property
    def mass_count(self):
        return sum(lp.mass_count for lp in self.loops)

    @property
    def parts(self):
        return dict(L=len(self.loops), n=self.mass_count,
                    tau_plus=sum((lp.twist_present for lp in self.loops), Fraction(0)),
                    T=self.charge, s=sum((lp.spin for lp in self.loops), Fraction(0)),
                    C=sum((lp.total_curl for lp in self.loops), Fraction(0)),
                    orb=sum((lp.orbital for lp in self.loops), Fraction(0)))

    def check_whole_charge(self):
        if self.charge.denominator != 1:
            raise ValueError("%s: a particle on its own has whole charge; got %s" % (self.name, self.charge))
        return True


def net_twist(particles):
    return sum((p.charge for p in particles), Fraction(0))


# ---------------------------------------------------------------- the sheet's hand-built loops

TRIANGLE = [(1, 1, 2), (1, -2, -1), (-2, 1, -1)]          # three box diagonals that close: a flat triangle (exact)


def triangle(L, start=(0, 0, 0), twists=(-THIRD, -THIRD, -THIRD), spin=HALF, curl=1):
    return Loop(L, start, list(TRIANGLE), list(twists), curl=curl, spin=spin)


def mirror(loop):
    """The same loop with every twist, its curl and its spin reversed: the anti-particle's loop (pairs: +theta with -theta;
    her ledger of 30 October, evening: the positron is -1 twist, -1 loop)."""
    return Loop(loop.L, loop.start, list(loop.steps), [-t for t in loop.twists], curl=-loop.curl, spin=-loop.spin,
                held=[(-c, -s) for c, s in loop.held])


# ---------------------------------------------------------------- the photon, first form (her decision of 30 October, evening)

BODY_STEPS = {tuple(s) for s in product((1, -1), repeat=3)}


def photon_curl_allowed(c):
    """DECISION (hers, 31 October): a photon's curl angle may turn less than a quarter turn per link, so its repeat (its
    wavelength) is any whole number of links n >= 4, and its curl per link is 2/n: 1/2, 2/5, 1/3, 2/7, 1/4, ... (exact: a
    quarter turn per link is the half curl's n = 4; returning exactly to the start needs a whole n). Its energy is its curl
    per link, set by its wavelength ("the one that matches measurement": per-packet energy goes with frequency, the
    photoelectric effect, general knowledge, to verify), not curl times length; the length is the pulse."""
    c = Fraction(c)
    return c != 0 and (2 / abs(c)).denominator == 1 and 2 / abs(c) >= 4


@dataclass
class Photon:
    """Her first form of the photon, put in as she stated it: a curve on the two-colour (body) diagonal, half a curl, spin 1,
    no twist, carried one link per sweep by its spin; the angle its curl sits at turns as it moves and repeats every four
    links; on meeting matter it hands on its spin and its curl (not built: the meeting). An open path of `length` body
    diagonals, all the same step `direction`, from `start`, advanced by `position` links. curl is +1/2 or -1/2 (the sign is
    the sense of the curl; her annihilation ledger makes the two photons opposite), spin +1 or -1 (the two photons' spins
    are opposite in direction). Nothing is derived; the values are the rules. The two-colour diagonal has length sqrt 3 (she
    wrote root 2 and corrected it to root 3, 30 October). **31 October (hers):** the repeat `period` is any whole n >= 4
    and the curl per link is +-2/n (`photon_curl_allowed`); `curl` given as +-1/2 keeps period 4; `energy` is |curl|."""
    L: int
    start: tuple
    direction: tuple
    length: int = 4
    curl: Fraction = HALF
    spin: Fraction = Fraction(1)
    position: int = 0
    period: int = None

    def __post_init__(self):
        if tuple(self.direction) not in BODY_STEPS:
            raise ValueError("a photon runs along a body diagonal (1, 1, 1) in some signed form; got %s" % (self.direction,))
        self.curl, self.spin = Fraction(self.curl), Fraction(self.spin)
        if self.period is None:
            if not photon_curl_allowed(self.curl):
                raise ValueError("a photon's curl per link is +-2/n for a whole period n >= 4 (31 October); got %s" % (self.curl,))
            self.period = int(2 / abs(self.curl))
        else:
            self.period = int(self.period)
            if self.period < 4:
                raise ValueError("a photon's repeat is at least four links (her shortest wavelength); got %d" % self.period)
            sign = 1 if self.curl > 0 else -1
            if abs(self.curl) not in (HALF, Fraction(2, self.period)):
                raise ValueError("a photon of period %d carries curl per link +-2/%d; got %s" % (self.period, self.period, self.curl))
            self.curl = sign * Fraction(2, self.period)
        if self.spin not in (Fraction(1), Fraction(-1)):
            raise ValueError("a photon carries spin 1, in one direction or the other")
        if self.length < 1:
            raise ValueError("a photon has at least one link")

    @property
    def charge(self):
        return Fraction(0)

    net_twist = charge

    @property
    def energy(self):
        """Her decision of 31 October: the energy is the curl per link, 2/period, set by the wavelength and not by the length."""
        return abs(self.curl)

    @property
    def twist_present(self):
        return Fraction(0)

    @property
    def mass_count(self):
        """Zero: a photon's diagonals are two-colour, not matter's one-colour box diagonals, so they are not counted as mass
        (her decision, 30 October, evening); `length` is the path it is drawn on and `period` its repeat (4 for the half curl)."""
        return 0

    @property
    def head(self):
        p = np.array(self.start, dtype=int) + self.position * np.array(self.direction, dtype=int)
        return tuple(int(x) % self.L for x in p)

    @property
    def colour(self):
        return colour(self.head)

    @property
    def angle_degrees(self):
        """The angle the curl sits at: 360/period degrees per link, repeating every `period` links (her picture; a quarter turn
        for the half curl)."""
        return Fraction(360, self.period) * (self.position % self.period)

    @property
    def parts(self):
        return dict(L=0, n=0, tau_plus=Fraction(0), T=Fraction(0), s=self.spin, C=self.curl, orb=Fraction(0))

    def advance(self, sweeps=1):
        """One link per sweep, carried by its spin."""
        return Photon(self.L, self.start, self.direction, self.length, self.curl, self.spin, self.position + int(sweeps), self.period)


def annihilate(particle, antiparticle, direction=(1, 1, 1), length=4, period=4):
    """Her ledger of 30 October, evening: an electron (+1 twist, 1 loop, 1/2 spin) and its anti-particle (-1 twist, -1 loop,
    1/2 spin) make two photons, C (+1/2 loop, spin 1, no twist) and D (-1/2 loop in the opposite direction, spin 1, no
    twist). Built with the photons' spins opposite (+1, -1), the only reading under which the signed spin balances given
    that `mirror` reverses the loop's spin. The photons leave from the particle's first point along `direction` and its
    reverse. Returns (C, D). What is conserved is read by `ledger`. 31 October: the two photons' period (their wavelength)
    is `period`, 4 by default; which period annihilation makes is pinned by nothing in the toy yet (measured, general
    knowledge, to verify: two photons of 511 keV each from a pair at rest, one definite wavelength)."""
    start = particle.loops[0].start
    back = tuple(-x for x in direction)
    c = Fraction(2, period)
    return (Photon(particle.loops[0].L, start, direction, length, c, Fraction(1), period=period),
            Photon(particle.loops[0].L, start, back, length, -c, Fraction(-1), period=period))


def absorb(loop, photon):
    """The meeting with matter (her decision of 30 October, evening: "the electron on a higher energy level"): the loop takes
    the photon's half curl and its spin 1 and holds them as an excitation until it re-emits. The loop's own spin stays as
    it was; the absorbed spin is orbital. Which loop takes it, and whether it may (a free electron cannot, in published
    physics), is not decided here: this is the bookkeeping of one absorption."""
    return Loop(loop.L, loop.start, list(loop.steps), list(loop.twists), curl=loop.curl, spin=loop.spin,
                held=list(loop.held) + [(photon.curl, photon.spin)])


def emit(loop, direction=(1, 1, 1), length=4):
    """The loop re-emits its most recently held photon along `direction`. Returns (loop, photon)."""
    if not loop.held:
        raise ValueError("nothing held: the loop is not excited")
    c, s = loop.held[-1]
    lp = Loop(loop.L, loop.start, list(loop.steps), list(loop.twists), curl=loop.curl, spin=loop.spin, held=list(loop.held[:-1]))
    return lp, Photon(loop.L, loop.start, direction, length, c, s, period=int(2 / abs(c)))


def ledger(objects):
    """The sums her ledger balances, over particles and photons alike: net twist T, curl C (held half curls included),
    the loops' own spin s, orbital orb (held photon spins), J = s + orb, and the mass count n."""
    tw = cu = sp = ob = Fraction(0)
    n = 0
    for o in objects:
        pr = o.parts
        tw, cu, sp, ob, n = tw + pr["T"], cu + pr["C"], sp + pr["s"], ob + pr["orb"], n + pr["n"]
    return dict(T=tw, C=cu, s=sp, orb=ob, J=sp + ob, n=n)


# ---------------------------------------------------------------- the coil's writhe, and the energy of twist (30 October, her decisions)

_PIECES = 8
_gx, _gw = np.polynomial.legendre.leggauss(16)
_GX = np.concatenate([(k + (_gx + 1) / 2) / _PIECES for k in range(_PIECES)])
_GW = np.concatenate([_gw / 2 / _PIECES for _ in range(_PIECES)])


def unwrapped_points(loop):
    """The loop's corners in the plane's own coordinates, unwrapped (cumulative steps from the start); None if the loop
    winds round the torus, since the writhe of a winding loop is not that of a closed curve in space."""
    total = np.sum(np.array(loop.steps, dtype=int), axis=0)
    if any(int(t) != 0 for t in total):
        return None
    pts = [np.array(loop.start, dtype=float)]
    for s in loop.steps[:-1]:
        pts.append(pts[-1] + np.array(s, dtype=float))
    return np.array(pts)


def writhe(loop):
    """The writhe of the loop as a closed polygon in space: the Gauss double integral over non-adjacent segment pairs,
    each segment split into 8 pieces with 16-point Gauss-Legendre (the routine of sandbox/coil_writhe.py). Exact to the
    quadrature; 0 for a planar loop; its sign flips with the loop's mirror image. None for a loop that winds the torus."""
    V = unwrapped_points(loop)
    if V is None:
        return None
    n = len(V)
    T = np.roll(V, -1, axis=0) - V
    R = V[:, None, :] + _GX[None, :, None] * T[:, None, :]
    total = 0.0
    for i in range(n):
        js = [j for j in range(n) if j != i and (i + 1) % n != j and (j + 1) % n != i]
        if not js:
            continue
        cr = np.cross(T[i], T[js])
        d = R[i][:, None, None, :] - R[js][None, :, :, :]
        num = np.einsum("qjpk,jk->qjp", d, cr)
        den = np.linalg.norm(d, axis=3) ** 3
        total += np.einsum("q,p,qjp->", _GW, _GW, num / den)
    return float(total / (4 * np.pi))


def loop_kind(loop):
    """"face" if every step is a face diagonal, else "box" (box and long diagonals share the box constant)."""
    return "face" if all(step_kind(s) == "face" for s in loop.steps) else "box"


def twist_energy(loop, eps=1.0, eps_w=1.0, eps_face=None):
    """E4 plus writhe: eps [sum tau^2 + sum (tau_next - tau)^2] + eps_w Wr^2 (the build document, section 4). A winding
    loop has no writhe and pays eps_w * 0. DECISION (hers, 30 October, "agreed, proceed"): the twist constant may differ
    by kind of diagonal: eps on box (and long) diagonals, eps_face on face diagonals; eps_face=None means one constant for
    both. Exact (sandbox/neutrino_by_kind.py): her dark two-loop neutrino on face diagonals is cheaper than the box one
    iff eps_face < 4/9 eps, whichever way its twist cancels."""
    if eps_face is not None and loop_kind(loop) == "face":
        eps = eps_face
    taus = loop.twists
    n = len(taus)
    elastic = float(sum(t * t for t in taus))
    mismatch = float(sum((taus[(i + 1) % n] - taus[i]) ** 2 for i in range(n)))
    if eps_w == 0.0:
        return eps * (elastic + mismatch)
    wr = writhe(loop)
    return eps * (elastic + mismatch) + eps_w * (0.0 if wr is None else wr * wr)


def particle_energy(particle, eps=1.0, eps_w=1.0, eps_face=None):
    return sum(twist_energy(lp, eps, eps_w, eps_face) for lp in particle.loops)
