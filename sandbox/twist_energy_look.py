"""Sandbox look (4 November 2026, before the twist-energy brief; rule 19, the exact look first): what the energy rules
that already exist say about the store-to-dark-particle ratio she claims (about a billion, her words of 4 November),
and what her billion would need. Four rules, each labelled; nothing fitted; a look, not a test.

  count   hers, Update 139: mass counts looped diagonals, 0.170 MeV each (the electron as the unit).
  rod     ours, textbook stand-in (general knowledge, to verify): the twist energy of a uniformly twisted elastic rod,
          kappa * tau^2 / L for twist tau (turns) on length L; the record's "a shorter loop holding a half twist is bent
          tighter" has this direction (HYPOTHESIS, refinements of 4 November).
  toy     exact, as built in graphity.curl_pair_moves: a dark particle is one full curl (one unit); a store holds its
          partner's unit plus, under price_from="ledger", the price B + 4(lambda - 1).
  billion her claim of 4 November: E_store / E_dark ~ 1e9; and "the one-in-a-billion is a count of stores against pairs".
"""
import math

SQ2, SQ6 = math.sqrt(2.0), math.sqrt(6.0)
E_LOOP = 0.511 / 3  # MeV per looped diagonal, hers (the electron as the unit)


def rod(tau, length, kappa=1.0):
    return kappa * tau * tau / length


print("== count rule (hers): one looped diagonal = %.3f MeV ==" % E_LOOP)
print("  dark particle (one face diagonal, half twist): %.3f MeV; store on one sqrt2 diagonal: %.3f MeV (two looped: %.3f)"
      % (E_LOOP, E_LOOP, 2 * E_LOOP))
print("  store / dark = 1 (or 2). Per unit length every loop is the same, so 'heaviest per length' needs a twist part.")

print("== rod rule (ours, stand-in), kappa = 1 ==")
dark, store = rod(0.5, SQ2), rod(1.0, SQ2)
elec, prot = rod(1.0, 3 * SQ6), rod(1.0, 5508 * SQ6)
print("  dark %.4f  store %.4f  electron(3 thirds, L=3 sqrt6) %.4f  proton(net 1, 5508 sqrt6) %.6f" % (dark, store, elec, prot))
print("  store / dark = %.1f  (full twist against half, same length: exactly 4)" % (store / dark))
print("  twist energy per unit length: dark %.4f  electron %.4f  -> dark is %.2f times the electron per length"
      % (dark / SQ2, elec / (3 * SQ6), (dark / SQ2) / (elec / (3 * SQ6))))

print("== toy as built (exact): store / dark = 1 + B + 4(lambda - 1) under the ledger price ==")
for B in (0, 4, 12, 21):
    print("  B=%2d: " % B + "  ".join("lam %.2f -> %.1f" % (lam, 1 + B + 4 * (lam - 1)) for lam in (1.05, 1.1, 1.25, 1.5, 2.0)))

print("== her billion ==")
beta = 1e9 * E_LOOP
print("  with the dark particle at the count rule's %.3f MeV, a store would hold %.3g MeV = %.0f TeV" % (E_LOOP, beta, beta / 1e6))
print("  with her other statement (stores are one in a billion of the pairs): total store energy = N_pairs x E_dark,")
print("  the pairs' own weight as dark particles; the two billions cancel exactly, whatever E_dark is.")
ratio_planck = 5.36  # dark to ordinary matter by mass, Planck 2018 (general knowledge, to verify)
m_p = 938.272
print("  Planck's dark-to-ordinary mass ratio %.2f at m_dark = %.3f MeV would need %.0f dark particles per proton"
      % (ratio_planck, E_LOOP, ratio_planck * m_p / E_LOOP))
