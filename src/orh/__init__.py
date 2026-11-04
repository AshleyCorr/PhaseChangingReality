"""Observer-relative histories: a unitary toy for Stages B to D of the research handoff
(docs/theory/observer_relative_histories_research_handoff.md) and a separate cosmological population model for
Stage E.

Nothing in this package collapses a state. "Measurement" is an entangling gate that writes a record into another
qubit; probabilities are read from the decoherence functional of a declared family of histories; observers are
registers, not minds. Each module says which published equation it implements and which quantities are the toy's own.

Modules
    statevector   named qubits, gates, partial traces, entropies (the simulator)
    darwinism     closed forms for the controlled-tilt record model (the exact benchmarks for Stage B)
    histories     class operators, the decoherence functional, sum rules, compatible sets, record surprisal (Stage C)
    observers     observer registers that read, share, forget and duplicate records; joint distributions (Stage D)
    cosmic        the observer-moment rate over cosmic time in a fixed FLRW background (Stage E)
"""
__version__ = "0.1.0"
