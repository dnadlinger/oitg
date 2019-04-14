from itertools import chain, product
from typing import Iterable
import numpy as np
from ...gate import Gate, GateSequence, remap_operands


def generate_process_tomography_sequences(target: GateSequence,
                                          num_qubits: int) -> Iterable[GateSequence]:
    # For state preparation, prepare Pauli eigenvalues of both signs. This is nicely
    # symmetric, although it produces more states than necessary (6^n, 4^n would be
    # sufficient, where n is the number of qubits).
    # In practice, even for two-qubit gates this only just more than doubles (9 / 4)
    # the number of measurements, so we keep the symmetry and just take less shots per
    # sequence.
    fiducials = [
        (Gate("ry", (np.pi / 2, ), (0, )), ),  # +x
        (Gate("rx", (-np.pi / 2, ), (0, )), ),  # +y
        (),  # +z
        (Gate("ry", (-np.pi / 2, ), (0, )), ),  # -x
        (Gate("rx", (np.pi / 2, ), (0, )), ),  # -y
        (Gate("rx", (np.pi, ), (0, )), ),  # -z
    ]

    def product_fiducials(locals):
        return [
            tuple(
                chain.from_iterable(
                    remap_operands(seq, {0: i}) for (i, seq) in enumerate(seqs)))
            for seqs in product(locals, repeat=num_qubits)
        ]

    return [
        tuple(chain(prep, target, measure[::-1]))
        for prep in product_fiducials(fiducials)
        for measure in product_fiducials(fiducials[:3])
    ]
