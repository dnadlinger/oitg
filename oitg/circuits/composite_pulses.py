import numpy as np
from typing import Callable, Iterable
from .gate import Gate, GateGenerator


class UnsupportedGate(ValueError):
    """Raised if a given gate cannot be expanded in the requested form."""
    pass


def to_rxy(gate: Gate):
    """Canonicalise all single-qubit rotations in the xy-plane to ``rxy`` gates.

    :return: A ``rxy`` :class:`.Gate` with positive rotation amount.
    """
    if gate.kind == "rxy":
        return gate
    phase = {"rx": 0.0, "ry": np.pi / 2}.get(gate.kind, None)
    if phase is None:
        raise UnsupportedGate

    amount = gate.parameters[0]
    if amount < 0:
        phase += np.pi
        amount *= -1

    return Gate("rxy", (phase, amount), gate.operands)


def bb1(gate: Gate, symmetric: bool = False):
    """Implement ``gate`` using the BB1 composite pulse.

    BB1, due to S. Wimperis, is a broadband amplitude noise suppression sequence,
    cancelling error terms up to fourth order in the amplitude miscalibration.

    Reference: S. Wimperis, J. Magn. Reson. Ser. A (1994), doi:10.1006/jmra.1994.1159.

    :param symmetric: Whether to implement the gate symmetrically using 5 pulses (2
        half-rotations around the 3-pulse BB1 identity sequence), or asymmetrically
        using 4 pulses. The latter has slightly nicer behaviour under detuning errors.
    """
    gate = to_rxy(gate)
    phase, amount = gate.parameters
    phi = np.arccos(-amount / (4 * np.pi))

    def xy(phase, amount):
        return Gate("rxy", (phase, amount), gate.operands)

    first_amount = 0.5 * amount if symmetric else amount
    yield xy(phase, first_amount)
    yield xy(phase + phi, np.pi)
    yield xy(phase + 3 * phi, 2 * np.pi)
    yield xy(phase + phi, np.pi)
    if symmetric:
        yield xy(phase, first_amount)


def expand_using(method: Callable,
                 gates: GateGenerator,
                 ignore_kinds: Iterable[str] = [],
                 ignore_unsupported_gates: bool = True,
                 insert_barriers: bool = True):
    """Expand all gates in the given sequence using composite pulses.

    :param method: A callable implementing the chosen composite pulse type
        (e.g. :meth:`bb1`).
    :param ignore_kinds: A set of gate kinds not to attempt to expand.
    :param ignore_unsupporrted_gates: If ``True``, silently pass over unsupported gates
        without expanding them.
    :param insert_barriers: Insert a barrier after each composite pulse.
    """
    ignore_kinds = set(ignore_kinds)
    for gate in gates:
        try:
            if gate.kind in ignore_kinds:
                yield gate
                continue
            yield from method(gate)
            if insert_barriers:
                yield Gate("barrier", (), gate.operands)
        except UnsupportedGate:
            if not ignore_unsupported_gates:
                raise
            yield gate
