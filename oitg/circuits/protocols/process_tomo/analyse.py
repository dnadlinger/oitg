import numpy as np
from typing import Dict, List
from ...gate import *
from ...to_matrix import apply_gate_sequence, gate_sequence_matrix
from .tools import *


def _find_first_index(needle, haystack):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    raise ValueError


def guess_prepare_target_measure_split(all_sequences: List[GateSequence]):
    # FIXME: This is needlessly generic, yet fails if the target is more than one gate
    # or it also appears as part of the prepare/measure fiducials…
    possible_targets = set(all_sequences[0])
    for seq in all_sequences[1:]:
        possible_targets &= set(seq)
    if len(possible_targets) > 1:
        raise ValueError
    target_seq = tuple(possible_targets)

    target_start_idxs = [_find_first_index(target_seq, s) for s in all_sequences]

    return target_seq, [(s[:t], s[t + len(target_seq):])
                        for t, s in zip(target_start_idxs, all_sequences)]


def auto_prepare_data(outcomes: Dict[GateSequence, np.ndarray]):
    seqs = list(outcomes.keys())
    _, fiducial_pairs = guess_prepare_target_measure_split(seqs)
    return prepare_data({f: outcomes[s] for f, s in zip(fiducial_pairs, seqs)})


def prepare_data(outcomes: Dict[Tuple[GateSequence, GateSequence], np.ndarray]):
    fiducial_pairs = list(outcomes.keys())

    prep_sequences = list(set(f[0] for f in fiducial_pairs))
    prep_indices = [prep_sequences.index(f[0]) for f in fiducial_pairs]

    meas_sequences = list(set(f[1] for f in fiducial_pairs))
    meas_indices = [meas_sequences.index(f[1]) for f in fiducial_pairs]

    num_qubits = max(
        max(collect_operands(s), default=0) for f in fiducial_pairs for s in f) + 1
    initial_state = np.zeros(2**num_qubits, dtype=np.complex128)
    initial_state[0] = 1
    prep_states = [apply_gate_sequence(s, initial_state) for s in prep_sequences]

    meas_unitaries = [
        gate_sequence_matrix(s, num_qubits).T.conj() for s in meas_sequences
    ]
    meas_states = [u[:, i] for u in meas_unitaries for i in range(2**num_qubits)]
    observations = np.full((len(prep_states), len(meas_sequences) * 2**num_qubits), -1)
    for prep_idx, meas_idx, fids in zip(prep_indices, meas_indices, fiducial_pairs):
        meas_base_idx = 2**num_qubits * meas_idx
        for i, counts in enumerate(outcomes[fids]):
            observations[prep_idx, meas_base_idx + i] = counts
    if np.sum(observations == -1) != 0:
        raise NotImplementedError(
            "Currently assuming all prepare/measure combinations are present")
    return prep_states, meas_states, observations


def build_choi_predictor(prep_states, meas_states):
    return np.vstack([
        mat2vec(np.kron(projector(prep),
                        projector(meas).T)) for prep in prep_states
        for meas in meas_states
    ])


def invert_choi_predictor(choi_predictor, observations):
    # For consistent normalisation, infer the dimension of the underlying state Hilbert
    # space and the number of measurement bases from the given predictor/observation
    # matrix.
    pure_state_dimension = np.sqrt(np.sqrt(choi_predictor.shape[1]))
    if pure_state_dimension != int(pure_state_dimension):
        raise ValueError("Choi predictor not of right shape for CPTP involution")
    num_measurement_bases, rem = divmod(observations.shape[1], pure_state_dimension)
    if rem:
        raise ValueError("Number of observation matrix columns not consistent with "
                         "dim(pure_state) measurements per basis")

    normalised_observations = observations.astype(np.float64)
    shots_per_basis = np.sum(observations, axis=1) / num_measurement_bases
    for i in range(normalised_observations.shape[0]):
        normalised_observations[i, :] /= shots_per_basis[i]

    solution, residuals, rank, singular_vals = \
        np.linalg.lstsq(choi_predictor, mat2vec(normalised_observations), rcond=None)
    if rank != solution.size:
        raise ValueError("Predictor matrix was rank-deficient; "
                         "check that input/measurement state sets are complete")
    return vec2mat(solution) / pure_state_dimension


def linear_inversion_tomography(prep_states, meas_states, observations):
    predictor = build_choi_predictor(prep_states, meas_states)
    return invert_choi_predictor(predictor, outcomes)


def negative_log_likelihood(choi_predictor, observation_vec, choi):
    # [KBLG18] eq. 3/appendix A.
    probability_vec = np.real(choi_predictor @ mat2vec(choi))

    # Fudge predictions away from 0 to avoid stalling as per [KBLG18] appendix D.
    mask_small = probability_vec < 1e-16
    # if np.any(mask_small):
    #     warnings.warn("{} very small probabilities encountered".format(
    #         np.sum(mask_small)))
    probability_vec[mask_small] = 1e-16
    probability_vec /= np.sum(probability_vec)

    return -observation_vec.T @ np.log(probability_vec)


def negative_log_likelihood_gradient(choi_predictor, observation_vec, choi):
    # [KBLG18] eq. 6/appendix A.
    probability_vec = np.real(choi_predictor @ mat2vec(choi))

    # Fudge predictions away from 0 to avoid stalling as per [KBLG18] appendix D.
    mask_small = probability_vec < 1e-16
    # if np.any(mask_small):
    #     warnings.warn("{} very small probabilities encountered".format(
    #         np.sum(mask_small)))
    probability_vec[mask_small] = 1e-16
    probability_vec /= np.sum(probability_vec)

    # XXX: The text has the equivalent of `-choi_predictor.conj().T @ vec2mat(...)`, but
    # that doesn't work out in terms of dimensions.
    return -vec2mat(choi_predictor.conj().T @ (observation_vec / probability_vec))


def project_into_cp(choi):
    # [KBLG18] eq. 8
    eigvals, eigvecs = np.linalg.eigh((choi + choi.conj().T) / 2)
    eigvals[eigvals < 0.0] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


class TPProjector:
    def __init__(self, pure_state_dimension):
        self.dim = pure_state_dimension

        m = np.zeros((pure_state_dimension**2, pure_state_dimension**4))
        for i in range(pure_state_dimension):
            e = np.zeros(pure_state_dimension)
            e[i] = 1.0
            b = np.kron(np.eye(pure_state_dimension), e.T)
            m += np.kron(b, b)
        self.mdagger_m = m.conj().T @ m
        self.mdagger_id = m.conj().T @ mat2vec(np.eye(pure_state_dimension))

    def project(self, choi):
        # [KBLG18] eq. 12
        return choi + vec2mat(self.mdagger_id -
                              self.mdagger_m @ mat2vec(choi)) / self.dim


def project_into_cptp(choi, tp_projector: TPProjector):
    old_cp_step = np.zeros_like(choi)
    old_tp_step = np.zeros_like(choi)
    old_after_cp = np.zeros_like(choi)
    old_choi = choi

    for i in range(100000):
        before_cp = old_choi + old_cp_step
        after_cp = project_into_cp(before_cp)
        cp_step = before_cp - after_cp

        before_tp = after_cp + old_tp_step
        choi = tp_projector.project(before_tp)
        tp_step = before_tp - choi

        if (np.linalg.norm(old_cp_step - cp_step)**2 +
                np.linalg.norm(old_tp_step - tp_step)**2 +
                np.abs(2 * mat2vec(old_cp_step).conj().T @ mat2vec(choi - old_choi)) +
                np.abs(2 *
                       mat2vec(old_tp_step).conj().T @ mat2vec(after_cp - old_after_cp))
                < 1e-4):
            return choi
        old_cp_step = cp_step
        old_tp_step = tp_step
        old_after_cp = after_cp
        old_choi = choi
    raise ValueError("Did not converge")


def pgdb_mle_tomography(choi_predictor, observations):
    """Projected Gradient Descent with Backtracking."""
    float_pure_state_dimension = np.sqrt(np.sqrt(choi_predictor.shape[1]))
    pure_state_dimension = int(float_pure_state_dimension)
    if pure_state_dimension != float_pure_state_dimension:
        raise ValueError("Choi predictor not of right shape for CPTP involution")

    tp_projector = TPProjector(pure_state_dimension)

    # Note: Different normalisation used here!
    choi = np.eye(pure_state_dimension**2, dtype=np.complex128) / pure_state_dimension
    mu = 1.5  # / pure_state_dimension**2  XXX How should this change with different normalisation?
    gamma = 0.3

    # Normalise so that sum of all probabilities is 1.0, and that a Choi matrix with
    # entries in [-1, 1] gives that.
    number_of_basis_combinations = choi_predictor.shape[0] / pure_state_dimension
    choi_predictor = choi_predictor / number_of_basis_combinations
    observation_vec = mat2vec(observations) / np.sum(observations)

    old_nll = negative_log_likelihood(choi_predictor, observation_vec, choi)
    while True:
        nll_gradient = negative_log_likelihood_gradient(choi_predictor, observation_vec,
                                                        choi)

        choi_step = project_into_cptp(choi - nll_gradient / mu, tp_projector) - choi

        alpha = 1.0
        change = gamma * mat2vec(choi_step).conj().T @ mat2vec(nll_gradient)

        while True:
            nll = negative_log_likelihood(choi_predictor, observation_vec,
                                          choi + alpha * choi_step)
            if nll <= old_nll + change or alpha < 1e-10:
                break
            alpha /= 2
            change /= 2

        choi += alpha * choi_step
        nll = negative_log_likelihood(choi_predictor, observation_vec, choi)
        if old_nll - nll < 1e-10:
            break
        old_nll = nll
    return choi / pure_state_dimension
