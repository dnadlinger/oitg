from itertools import chain
from typing import Callable, Iterable, Union
from ..gate import GateGenerator, GateSequence
from ..qasm import gate_to_qasm


class SequenceRunnerOptions:
    """
    :param dataset_prefix: Prefix for dataset keys to write executed sequences and their
        results to. If ``None``, results are not written to datasets.
    """

    def __init__(self,
                 num_global_repeats: int = 1,
                 chunk_size: int = 1,
                 num_repeats_per_chunk: int = 1,
                 num_shots_per_repeat: int = 100,
                 randomise_per_chunk: bool = True,
                 dataset_prefix: Union[None, str] = "data.circuits."):
        self.num_global_repeats = num_global_repeats
        self.chunk_size = chunk_size
        self.num_repeats_per_chunk = num_repeats_per_chunk
        self.num_shots_per_repeat = num_shots_per_repeat
        self.randomise_per_chunk = randomise_per_chunk
        self.dataset_prefix = dataset_prefix


class SequenceRunner:
    def run_sequences(self,
                      sequences: Iterable[GateSequence],
                      num_qubits: Union[None, int] = None,
                      progress_callback: Callable = None,
                      progress_callback_interval: float = 5.0):
        raise NotImplementedError


def stringify_gate_sequence(seq: GateGenerator):
    """Return the string used to represent ``seq`` in the result datasets."""
    return ";".join(chain.from_iterable(gate_to_qasm(g) for g in seq))
