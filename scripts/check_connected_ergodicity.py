"""Can a chain that refuses to break the graph still reach every connected state? Usage:

    python scripts/check_connected_ergodicity.py

Prints to the screen and writes nothing; it checks a property of the move set, not the model.

WHY THIS IS NOT OPTIONAL. [Taylor81] proves that any connected graph reaches any other of the
same degree sequence by edge switches with every intermediate graph connected, so "switch, and
reject the move if the graph would come apart" is a valid chain on connected graphs. But that
theorem is about a space constrained only by degree. Ours also carries the hard-core rule, and
[Swap17] reports that adding a constraint can destroy exactly this property. So the guarantee
is not inherited and has to be demonstrated here (ASSUMPTION Q16).

HOW. Take every state at N = 16 and 18 (the exhaustive enumeration of T4), keep the connected
ones, and walk outwards from one of them using only moves that leave the graph in one piece.
If the walk reaches every connected class, the restricted chain is ergodic at these sizes. If
it does not, it is not, and the connected runs in Gate A and T11 need a different move.
"""
from graphity.connectivity import connectivity
from graphity.cqg import CAP, NO_CAP
from graphity.small_graphs import ClassList, count_labelled_states, explore, one_switch_away

for cap, label in ((NO_CAP, "no cap"), (CAP, "cap = 2")):
    print("\n=== %s ===" % label)
    for n in (8, 9):
        total, start = count_labelled_states(n, cap)
        if start is None:
            print("N = %2d: no states at all" % (2 * n))
            continue

        # every class, and which of them are connected
        classes, _ = explore(start, cap)
        connected_classes = [k for k, (adj, _) in enumerate(classes.reps) if connectivity(adj)[0] == 1]
        if not connected_classes:
            print("N = %2d: %d classes, none connected" % (2 * n, len(classes.reps)))
            continue

        # walk outwards using only moves that keep it in one piece
        seed = classes.reps[connected_classes[0]][0]
        reached, _ = explore(seed, cap, connected=True)
        assert all(connectivity(a)[0] == 1 for a, _ in reached.reps), "the restricted walk left the space"

        # match the two lists up by isomorphism
        lookup = ClassList()
        for k in connected_classes:
            lookup.index_of(classes.reps[k][0])
        found = {lookup.index_of(a) for a, _ in reached.reps}

        ok = len(found) == len(connected_classes)

        # did the restriction ever actually refuse anything? If not, the test is vacuous.
        refused = 0
        for k in connected_classes:
            rep = classes.reps[k][0]
            refused += len(one_switch_away(rep, cap)) - len(one_switch_away(rep, cap, connected=True))

        print("N = %2d: %3d classes, %3d connected, restricted walk reaches %3d  -> %s"
              % (2 * n, len(classes.reps), len(connected_classes), len(found),
                 "ERGODIC" if ok else "NOT ERGODIC"))
        print("        moves the restriction refused, over every connected class: %d%s"
              % (refused, "   <-- VACUOUS: it never had anything to refuse" if refused == 0 else ""))
        if not ok:
            print("        unreachable connected classes: %s"
                  % sorted(set(range(len(connected_classes))) - found))

print("""
Reading it. ERGODIC means the restricted walk reached every connected state. But read the
second line before believing it means anything: if the restriction refused NO moves, the test
is vacuous, because there was nothing to reject and the walk was the unrestricted one.

That is what happens here, and the reason is arithmetic. The smallest valid piece has 14
points, so a graph in two pieces needs at least 28, and every valid state at 16 or 18 points is
connected whether you ask for it or not. Exhaustive enumeration cannot reach 28 (20 already did
not finish in about 40 minutes), so **this property cannot be established exhaustively with the
machinery we have.**

What that leaves. [Taylor81] guarantees it for a space constrained only by degree; [Swap17]
warns that adding a constraint can break exactly that guarantee, and ours adds the hard-core
rule. So the honest position is that the restricted chain is *probably* ergodic and unproven on
our space. The practical diagnostic, which every connected run must report, is the share of
proposals refused for disconnection: where it is small the restriction barely bites and the
chain is effectively the unrestricted one; where it is large -- the shattered phase -- the
restriction is doing real work and the result needs the caveat attached.""")
