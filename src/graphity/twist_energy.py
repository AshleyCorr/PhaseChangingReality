"""The twist rules' energies as a reader (docs/design/twist_energies_2026-11-04.md; built 4 November 2026 with the brief's
defaults, ours, before her answers). Turns the objects of the curl-pair moves (a dark particle, a Klein store) and her
named loops (the electron, the proton) into an energy in MeV under a rule with three parts, so that a store count can
become a mass. Nothing is derived and nothing is fitted; the constants are inputs.

    E(object) = e_loop * n            the count part (hers, Update 139: mass counts looped diagonals; e_loop calibrated
                                      to the electron, 0.511 / 3 MeV)
              + kappa * tau^2 / L     the twist part (ours, a stand-in: the twist energy of a uniformly twisted elastic
                                      rod, general knowledge, to verify; kappa free, energy x length)
              + beta * [store]        the binding paid once when two halves join (ours as a form; beta free)

form="count" switches the twist part off (her count rule alone). Exact under the rod form with beta = 0: a store is
exactly 4 times a dark particle (a full twist against a half on the same length), and per unit length the dark
particle's twist energy is 6.75 times the electron's (her "heaviest per length" without a fit). The reader also gives
the beta that a target store-to-dark ratio would need (arithmetic, not a fit) and the MeV totals of a PairMoves census.
"""
import math
from dataclasses import dataclass

SQ2, SQ6 = math.sqrt(2.0), math.sqrt(6.0)
E_LOOP_MEV = 0.511 / 3.0  # hers: the electron as the unit, three looped diagonals


@dataclass(frozen=True)
class Loop:
    """A loop as the rule sees it: looped diagonals n, |twist| in turns, total length, and whether it is a store."""
    name: str
    n: int
    twist: float
    length: float
    store: bool = False


DARK = Loop("dark particle", 1, 0.5, SQ2)              # one face diagonal, a half twist (hers)
STORE = Loop("Klein store", 1, 1.0, SQ2, store=True)   # two halves joined on one sqrt2 diagonal (hers; n = 1 is choice 3)
ELECTRON = Loop("electron", 3, 1.0, 3 * SQ6)           # three white thirds (hers)
PROTON = Loop("proton", 5508, 1.0, 5508 * SQ6)         # her ring, net twist 1


@dataclass
class TwistEnergy:
    e_loop: float = E_LOOP_MEV
    kappa: float = 0.0
    beta: float = 0.0
    form: str = "rod"

    def __post_init__(self):
        if self.form not in ("rod", "count"):
            raise ValueError(self.form)

    def count_part(self, loop):
        return self.e_loop * loop.n

    def twist_part(self, loop):
        if self.form == "count":
            return 0.0
        return self.kappa * loop.twist * loop.twist / loop.length

    def binding_part(self, loop):
        return self.beta if loop.store else 0.0

    def energy(self, loop):
        return self.count_part(loop) + self.twist_part(loop) + self.binding_part(loop)

    def per_length(self, loop):
        return self.energy(loop) / loop.length

    def ratio_store_dark(self):
        return self.energy(STORE) / self.energy(DARK)

    def beta_for_ratio(self, target):
        """The binding a target store-to-dark ratio would need under this e_loop and kappa. Arithmetic, not a fit."""
        return target * self.energy(DARK) - self.count_part(STORE) - self.twist_part(STORE)

    def sheet(self, census):
        """MeV totals of a PairMoves census: the frozen single curls as dark particles, the stores as stores. Light is
        left in the toy's units (brief, choice 5)."""
        dark_n, store_n = int(census["frozen_curls"]), int(census["stores"])
        e_dark, e_store = self.energy(DARK), self.energy(STORE)
        dark_total, store_total = dark_n * e_dark, store_n * e_store
        pairs = int(census["pairs_made"])
        return dict(dark_particles=dark_n, stores=store_n, e_dark_mev=e_dark, e_store_mev=e_store,
                    dark_total_mev=dark_total, store_total_mev=store_total,
                    store_to_dark_total=(store_total / dark_total) if dark_total else float("inf") if store_total else 0.0,
                    stores_per_pair=(store_n / pairs) if pairs else 0.0,
                    dark_per_pair=(dark_n / pairs) if pairs else 0.0)
