r"""Predefined :class:`GSTSpec`\ s corresponding to commonly used gate sets."""

import numpy as np
from itertools import chain
from ...gate import Gate
from .generate import GSTSpec


def make_1q_xz_pi_2_spec() -> GSTSpec:
    """Return a single-qubit gate set using π/2 x- and z-rotations, corresponding to
    pyGSTi's `std1Q_XZ` model.
    """
    x = Gate("rx", (np.pi / 2, ), (0, ))
    z = Gate("rz", (np.pi / 2, ), (0, ))

    prep = [(), (x, ), (x, z), (x, x), (x, x, x), (x, z, x, x)]
    meas = [seq[::-1] for seq in prep]
    germs = [(x, ), (z, ), (z, x, x), (z, z, x)]

    return GSTSpec(prep, meas, germs, "std1Q_XZ")


def make_1q_xy_pi_2_i_spec() -> GSTSpec:
    x = Gate("rx", (np.pi / 2, ), (0, ))
    y = Gate("ry", (np.pi / 2, ), (0, ))
    i = Gate("i", (np.pi / 2, ), (0, ))

    prep = [(), (x, ), (y, ), (x, x), (x, x, x), (y, y, y)]
    meas = [seq[::-1] for seq in prep]  # Here, prep == meas anyway.
    germs = [(i, ), (x, ), (y, ), (x, y), (x, x, y), (x, y, y), (x, y, i), (x, i, y),
             (x, i, i), (y, i, i), (x, y, y, i), (x, x, y, x, y, y)]

    return GSTSpec(prep, meas, germs, "std1Q_XYI")


def make_2q_xy_pi_2_cphase_spec() -> GSTSpec:
    """Return a two-qubit gate set using a CPHASE (CZ) gate and local π/2 x- and
    y-rotations, corresponding to pyGSTi's `std2Q_XYCPHASE` model.
    """
    xi = (Gate("rx", (np.pi / 2, ), (0, )), )
    ix = (Gate("rx", (np.pi / 2, ), (1, )), )
    yi = (Gate("ry", (np.pi / 2, ), (0, )), )
    iy = (Gate("ry", (np.pi / 2, ), (1, )), )
    cz = (Gate("cz", (), (0, 1)), )

    prep = [(), (ix, ), (iy, ), (ix, ix), (xi, ), (xi, ix), (xi, iy), (xi, ix, ix),
            (yi, ), (yi, ix), (yi, iy), (yi, ix, ix), (xi, xi), (xi, xi, ix),
            (xi, xi, iy), (xi, xi, ix, ix)]

    meas = [(), (ix, ), (iy, ), (ix, ix), (xi, ), (yi, ), (xi, xi), (xi, ix), (xi, iy),
            (yi, ix), (yi, iy)]

    germs = [
        (xi, ), (yi, ), (ix, ), (iy, ), (cz, ), (xi, yi), (ix, iy), (iy, yi), (ix, xi),
        (ix, yi), (iy, xi), (xi, cz), (yi, cz), (ix, cz), (iy, cz), (xi, xi, yi),
        (ix, ix, iy), (ix, iy, cz), (xi, yi, yi), (ix, iy, iy), (iy, xi, xi),
        (iy, xi, yi), (ix, xi, iy), (ix, yi, xi), (ix, yi, iy), (ix, iy, yi),
        (ix, iy, xi), (iy, yi, xi), (xi, cz, cz), (ix, xi, cz), (ix, cz, cz),
        (yi, cz, cz), (iy, xi, cz), (iy, yi, cz), (iy, cz, cz), (ix, yi, cz),
        (cz, ix, xi, xi), (yi, ix, xi, iy), (ix, iy, xi, yi), (ix, ix, ix, iy),
        (xi, yi, yi, yi), (yi, yi, iy, yi), (yi, ix, ix, ix), (xi, yi, ix, ix),
        (cz, ix, cz, iy), (ix, xi, yi, cz), (iy, yi, xi, xi, iy), (xi, xi, iy, yi, iy),
        (iy, ix, xi, ix, xi), (yi, iy, yi, ix, ix), (iy, xi, ix, iy, yi),
        (iy, iy, xi, yi, xi), (ix, yi, ix, ix, cz), (xi, ix, iy, xi, iy, yi),
        (xi, iy, ix, yi, ix, ix), (cz, ix, yi, cz, iy, xi), (xi, xi, yi, xi, yi, yi),
        (ix, ix, iy, ix, iy, iy), (yi, xi, ix, iy, xi, ix), (yi, xi, ix, xi, ix, iy),
        (xi, ix, iy, iy, xi, yi), (ix, iy, iy, ix, xi, xi), (yi, iy, xi, iy, iy, iy),
        (yi, yi, yi, iy, yi, ix), (iy, iy, xi, iy, ix, iy),
        (iy, ix, yi, yi, ix, xi, iy), (yi, xi, iy, xi, ix, xi, yi, iy),
        (ix, ix, yi, xi, iy, xi, iy, yi)
    ]

    def flatten(gs):
        return [tuple(chain.from_iterable(g)) for g in gs]

    return GSTSpec(flatten(prep), flatten(meas), flatten(germs), "std2Q_XYCPHASE")


def make_2q_xy_pi_2_wobble_spec() -> GSTSpec:
    """Return a two-qubit gate set using a wobble gate and local π/2 x- and
    y-rotations.

    Fiducial/germ selection is based on pyGSTi's `std2Q_XYCNOT` model, with CNOT
    replaced by the wobble gate. As suggested in the pyGSTi documentation, this still
    leads to a complete set of germs, but might not be optimal. (The CNOT construction
    was chosen over the CPHASE-based germ set as its germ score was slightly better
    after switching to the wobble gate; fiducials are only single-qubit and hence the
    same for both anyway.)
    """
    xi = (Gate("rx", (np.pi / 2, ), (0, )), )
    ix = (Gate("rx", (np.pi / 2, ), (1, )), )
    yi = (Gate("ry", (np.pi / 2, ), (0, )), )
    iy = (Gate("ry", (np.pi / 2, ), (1, )), )
    w = (Gate("w", (), (0, 1)), )

    prep = [(), (ix, ), (iy, ), (ix, ix), (xi, ), (xi, ix), (xi, iy), (xi, ix, ix),
            (yi, ), (yi, ix), (yi, iy), (yi, ix, ix), (xi, xi), (xi, xi, ix),
            (xi, xi, iy), (xi, xi, ix, ix)]

    meas = [(), (ix, ), (iy, ), (ix, ix), (xi, ), (yi, ), (xi, xi), (xi, ix), (xi, iy),
            (yi, ix), (yi, iy)]

    germs = [
        (xi, ), (yi, ), (ix, ), (iy, ), (w, ), (xi, yi), (ix, iy), (iy, yi), (ix, xi),
        (ix, yi), (iy, xi), (iy, w), (yi, w), (xi, w), (ix, w), (xi, xi, yi),
        (ix, ix, iy), (xi, yi, yi), (ix, iy, iy), (iy, xi, xi), (iy, xi, yi),
        (ix, xi, iy), (ix, yi, xi), (ix, yi, iy), (ix, iy, yi), (ix, iy, xi),
        (iy, yi, xi), (iy, w, xi), (ix, ix, w), (xi, w, w), (iy, yi, w), (yi, w, w),
        (ix, iy, w), (iy, w, w), (w, ix, xi, xi), (yi, ix, xi, iy), (ix, iy, xi, yi),
        (ix, ix, ix, iy), (xi, yi, yi, yi), (yi, yi, iy, yi), (yi, ix, ix, ix),
        (xi, yi, ix, ix), (w, yi, w, xi), (w, ix, w, iy), (ix, xi, w, iy),
        (ix, iy, xi, w), (iy, yi, xi, xi, iy),
        (xi, xi, iy, yi, iy), (iy, ix, xi, ix, xi), (yi, iy, yi, ix, ix),
        (iy, xi, ix, iy, yi), (iy, iy, xi, yi, xi), (yi, w, iy, iy, xi),
        (xi, ix, iy, xi, iy, yi), (xi, iy, ix, yi, ix, ix), (yi, iy, xi, yi, xi, w),
        (xi, xi, yi, xi, yi, yi), (ix, ix, iy, ix, iy, iy), (yi, xi, ix, iy, xi, ix),
        (yi, xi, ix, xi, ix, iy), (xi, ix, iy, iy, xi, yi), (ix, iy, iy, ix, xi, xi),
        (yi, iy, xi, iy, iy, iy), (yi, yi, yi, iy, yi, ix), (iy, iy, xi, iy, ix, iy),
        (iy, ix, yi, yi, ix, xi, iy), (yi, xi, iy, xi, ix, xi, yi, iy),
        (ix, ix, yi, xi, iy, xi, iy, yi)
    ]

    def flatten(gs):
        return [tuple(chain.from_iterable(g)) for g in gs]

    return GSTSpec(flatten(prep), flatten(meas), flatten(germs), "wobble2Q_XYCNOT")


def make_2q_xz_pi_2_wobble_spec() -> GSTSpec:
    """Return a two-qubit gate set using a wobble gate and local π/2 x- and
    z-rotations.

    Fiducial/germ selection is based on pyGSTi's optimisation functions, run
    from scratch for the wobble gate.
    """
    xi = (Gate("rx", (np.pi / 2, ), (0, )), )
    ix = (Gate("rx", (np.pi / 2, ), (1, )), )
    zi = (Gate("rz", (np.pi / 2, ), (0, )), )
    iz = (Gate("rz", (np.pi / 2, ), (1, )), )
    w = (Gate("w", (), (0, 1)), )

    prep = [(), (ix, ), (xi, ), (ix, ix), (ix, iz), (ix, xi), (xi, xi), (xi, zi),
            (ix, iz, xi), (ix, xi, zi), (xi, ix, ix), (xi, ix, xi), (ix, xi, iz, zi),
            (ix, xi, xi, iz), (xi, ix, ix, xi), (xi, ix, ix, zi)]

    meas = [(), (ix, ), (xi, ), (ix, ix), (xi, xi), (iz, ix), (xi, ix), (zi, xi),
            (ix, zi, xi), (iz, ix, xi), (iz, ix, zi, xi)]

    germs = [(ix, ), (iz, ), (xi, ), (zi, ), (w, ), (xi, zi), (ix, zi), (iz, zi),
             (ix, iz, w), (xi, zi, w), (ix, ix, w), (xi, xi, w), (ix, w, zi),
             (iz, xi, w), (xi, w, zi), (iz, w, xi), (ix, w, iz), (xi, w, w),
             (ix, zi, w), (ix, iz, iz), (ix, w, w), (ix, iz, zi), (ix, iz, xi),
             (xi, xi, zi), (ix, zi, iz), (ix, xi, w), (ix, xi, iz), (ix, w, xi),
             (ix, iz, xi, w), (w, ix, xi, xi), (ix, xi, zi, w), (xi, ix, w, iz),
             (iz, zi, w), (ix, xi, w, iz), (iz, ix, xi, zi), (w, w, ix, ix),
             (w, ix, w, iz), (w, xi, zi, xi), (w, w, zi, zi), (zi, xi, xi, w),
             (xi, zi, zi, xi), (ix, ix, ix, iz), (xi, iz, iz, iz), (iz, iz, zi, iz),
             (w, zi, w, xi), (zi, zi, ix, xi), (iz, ix, ix, w), (xi, iz, zi, xi),
             (xi, w, xi, xi), (w, w, xi, w), (iz, ix, ix, zi), (zi, w, w, w),
             (xi, zi, ix, ix), (xi, zi, iz, iz), (iz, w, w, w), (iz, ix, w, ix),
             (zi, ix, iz, w, zi), (iz, w, xi, xi, iz), (xi, xi, zi, w, zi),
             (xi, ix, w, xi, iz), (zi, w, iz, iz, xi), (xi, ix, zi, ix, zi)]

    def flatten(gs):
        return [tuple(chain.from_iterable(g)) for g in gs]

    return GSTSpec(flatten(prep), flatten(meas), flatten(germs), "opt2Q_XZW")
