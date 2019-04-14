import numpy as np


def mat2vec(matrix):
    return matrix.reshape(-1)


def vec2mat(vector):
    float_dim = np.sqrt(vector.shape[0])
    dim = int(float_dim)
    assert float_dim == dim
    return vector.reshape(dim, dim)


def projector(ket):
    return np.outer(ket, np.conjugate(ket))


def choi2liou(choi):
    float_dim = np.sqrt(choi.shape[0])
    dim = int(float_dim)
    assert dim == float_dim
    return dim * np.reshape(choi, (dim, dim, dim, dim)).swapaxes(0, 3).reshape(
        (dim**2, dim**2))


def liou2choi(liou):
    return choi2liou(liou) / liou.shape[0]


def avg_gate_fidelity(liou, target_unitary):
    target_liou = np.kron(np.conj(target_unitary), target_unitary)
    float_dim = np.sqrt(liou.shape[0])
    dim = int(float_dim)
    assert dim == float_dim
    return (np.real(np.trace(liou @ np.conjugate(target_liou).T)) + dim) / (dim**2 +
                                                                            dim)
