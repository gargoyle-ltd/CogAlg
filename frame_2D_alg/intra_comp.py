import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# Constants

Y_COEFFS = [
    np.array([-1, -1, 1, 1]),
    np.array([-0.5, -0.5, -0.5, 0, 0.5, 0.5, 0.5, 0]),
]

X_COEFFS = [
    np.array([-1, 1, 1, -1]),
    np.array([-0.5, 0, 0.5, 0.5, 0.5, 0, -0.5, -0.5]),
]


# -----------------------------------------------------------------------------
# Functions

def comp_g(dert__, odd):
    """
        Cross-comp of g or ga in 2x2 kernels
        Parameters
        ----------
        dert__ : array-like, each dert = (i, g, dy, dx, da0, da1, da2, da3)
        Returns
        -------
        gdert__ : masked_array
            Output dert = (g, gg, gdy, gdx, gm, ga, day, dax).
        Examples
        --------

        'specific output'
        Notes
        -----
        Comparand is dert[1]
        """

    g__, cos_da0__, cos_da1__ = dert__[[1, -2, -1]]  # input

    # this mask section would need further test later with actual input from frame_blobs
    if isinstance(g__, ma.masked_array):
        g__.data[g__.mask] = np.nan
        g__.mask = ma.nomask

    g__topleft = g__[:-1, :-1]
    g__topright = g__[:-1, 1:]
    g__bottomleft = g__[1:, :-1]
    g__bottomright = g__[1:, 1:]

    # y-decomposed difference between gs
    dgy__ = ((g__bottomleft + g__bottomright) - (
            g__topleft * cos_da0__  + g__topright * cos_da1__)) * 0.5

    # x-decomposed difference between gs
    dgx__ = ((g__topright + g__bottomright) - (
            g__topleft * cos_da0__ + g__bottomleft * cos_da1__)) * 0.5

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    gm0__ = np.minimum(g__bottomleft, (g__topright * cos_da0__))  # g match = min(g, _g*cos(da))
    gm1__ = np.minimum(g__bottomright, (g__topleft * cos_da1__))
    gm__ = gm0__ + gm1__

    #    ga__=dert__[5], day_=dert__[6], dax=dert__[7]
    gdert = g__, gg__, dgy__, dgx__, gm__, dert__[4], dert__[5], dert__[6]

    return gdert


def comp_r(dert__, fig):
    """
    Cross-comp of input param (dert[0]) over rng set in intra_blob.
    This comparison is selective for blobs with below-average gradient,
    where input intensity doesn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping
    alternating derts as a kernel-central dert at current comparison range,
    which forms increasingly sparse input dert__ for greater range cross-comp,
    while maintaining one-to-one overlap between kernels of compared derts.

    With increasingly sparse input, unilateral rng (distance between central derts)
    can only increase as 2^(n + 1), where n starts at 0:

    rng = 1 : 3x3 kernel, skip orthogonally alternating derts as centrals,
    rng = 2 : 5x5 kernel, skip diagonally alternating derts as centrals,
    rng = 3 : 9x9 kernel, skip orthogonally alternating derts as centrals,
    ...
    That means configuration of preserved (not skipped) derts will always be 3x3.
    Parameters
    ----------
    dert__ : array-like
        Array containing inputs.
    fig : bool
        Set to True if input is g or derived from g
    -------
    output: masked_array
    -------
     dert = i, g, dy, dx, m
    <<< dert = i, g, dy, dx, m
    # results are accumulated in the input dert
    # comparand = dert[0]
    """

    # input is gdert (g,  gg, gdy, gdx, gm, iga, iday, idax)
    # input is dert  (i,  g,  dy,  dx,  m)  or
    # input is rdert (ir, gr, dry, drx, mr)

    ir__, gr__, dry__, drx__, mr__ = dert__[0:5]


    if fig:
        # center derts
        ig_center = gr__[1:-2:2, 1:-2:2]

        # get each direction
        ig__topleft = gr__[:-2:2, :-2:2]
        ig__topcenter = gr__[:-2:2, 1:-1:2]
        ig__topright = gr__[:-2:2, 2::2]
        ig__rightcenter = gr__[1:-1:2, 2::2]
        ig__bottomright = gr__[2::2, 2::2]
        ig__bottomcenter = gr__[2::2, 1:-1:2]
        ig__bottomleft = gr__[2::2, :-2:2]
        ig__leftcenter = gr__[1:-1:2, :-2:2]

        # angle, gradient, match
        a__ = [dry__, drx__] / gr__
        a__center = a__[1:-2:2, 1:-2:2]

        a__topleft = a__[:-2:2, :-2:2]
        a__topcenter = a__[:-2:2, 1:-1:2]
        a__topright = a__[:-2:2, 2::2]
        a__rightcenter = a__[1:-1:2, 2::2]
        a__bottomright = a__[2::2, 2::2]
        a__bottomcenter = a__[2::2, 1:-1:2]
        a__bottomleft = a__[2::2, :-2:2]
        a__leftcenter = a__[1:-1:2, :-2:2]

        # computing angles
        ida__ = np.stack(angle_diff(a__center, a__topleft),
                         angle_diff(a__center, a__topcenter),
                         angle_diff(a__center, a__topright),
                         angle_diff(a__center, a__rightcenter),
                         angle_diff(a__center, a__bottomright),
                         angle_diff(a__center, a__bottomcenter),
                         angle_diff(a__center, a__bottomleft),
                         angle_diff(a__center, a__leftcenter))

        # y-decomposed difference between angles to center
        idy__ = (ida__ * Y_COEFFS[1]).sum(axis=-1)

        # x-decomposed difference between angles to center
        idx__ = (ida__ * X_COEFFS[1]).sum(axis=-1)

        # calculating gradient
        idg__ = np.hypot(np.arctan2(*idy__), np.arctan2(*idx__))

        # calculating match
        # are we using gradient from input or sparsed idg__?
        drm__ =   np.minimum(ig_center, (ig__topleft * ida__[0][1])) \
                + np.minimum(ig_center, (ig__topcenter * ida__[1][1])) \
                + np.minimum(ig_center, (ig__topright * ida__[2][1])) \
                + np.minimum(ig_center, (ig__rightcenter * ida__[2][1])) \
                + np.minimum(ig_center, (ig__bottomright * ida__[2][1])) \
                + np.minimum(ig_center, (ig__bottomcenter * ida__[2][1])) \
                + np.minimum(ig_center, (ig__bottomleft * ida__[2][1])) \
                + np.minimum(ig_center, (ig__leftcenter * ida__[2][1]))


    else:
        ir_center = ir__[1:-2:2, 1:-2:2]

        ir__topleft = ir__[:-2:2, :-2:2]
        ir__topcenter = ir__[:-2:2, 1:-1:2]
        ir__topright = ir__[:-2:2, 2::2]
        ir__rightcenter = ir__[1:-1:2, 2::2]
        ir__bottomright = ir__[2::2, 2::2]
        ir__bottomcenter = ir__[2::2, 1:-1:2]
        ir__bottomleft = ir__[2::2, :-2:2]
        ir__leftcenter = ir__[1:-1:2, :-2:2]

        ir_compares = np.stack((ir__topleft - ir_center),
                               (ir__topcenter - ir_center),
                               (ir__topright - ir_center),
                               (ir__rightcenter - ir_center),
                               (ir__bottomright - ir_center),
                               (ir__bottomcenter - ir_center),
                               (ir__bottomleft - ir_center),
                               (ir__leftcenter - ir_center))

        idy__ = (ir_compares * Y_COEFFS[1]).sum(axis=-1)
        idx__ = (ir_compares * X_COEFFS[1]).sum(axis=-1)

        idg__ = np.hypot(np.arctan2(*idy__), np.arctan2(*idx__))

def comp_a(dert__, fga):
    """
    cross-comp of a or aga in 2x2 kernels
    Parameters
    ----------
    dert__ : array-like
        dert's structure depends on fga
    fga : bool
        If True, dert's structure is interpreted as:
        (g, gg, gdy, gdx, gm, iga, iday, idax)
        Otherwise it is interpreted as:
        (i, g, dy, dx, m)
    Returns
    -------
    adert : masked_array
        adert's structure is (i, g, dy, dx, m, ga, day, dax, da).
    Examples
    --------
    'specific output'
    """

    # input dert = (i,  g,  dy,  dx,  m, ga, day, dax, dat(da0, da1, da2, da3))
    i__, g__, dy__, dx__, m__ = dert__[0:5]

    if fga:  # input is adert
        ga__, day__, dax__ = dert__[5:8]
        a__ = [day__, dax__] / ga__  # similar to calc_a

    else:
        a__ = [dy__, dx__] / g__  # similar to calc_a

    # this mask section would need further test later with actual input from frame_blobs
    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # each shifted a in 2x2 kernel
    a__topleft = a__[:, :-1, :-1]
    a__topright = a__[:, :-1, 1:]
    a__bottomright = a__[:, 1:, 1:]
    a__bottomleft = a__[:, 1:, :-1]

    a__directions = np.stack((a__topleft,
                              a__topright,
                              a__bottomright,
                              a__bottomleft))

    # diagonal angle differences
    sin_da0__, cos_da0__ = angle_diff(a__topleft, a__bottomright)
    sin_da1__, cos_da1__ = angle_diff(a__topright, a__bottomleft)

    # rate of change in y direction for the angles
    day__ = (-sin_da0__ - sin_da1__) + (cos_da0__ + cos_da1__)

    # rate of change in x direction for the angles
    dax__ = (-sin_da0__ + sin_da1__) + (cos_da0__ + cos_da1__)

    # compute gradient magnitudes (how fast angles are changing)
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

    # change adert to tuple as ga__,day__,dax__ would have different dimension compared to inputs
    adert__ = i__, g__, dy__, dx__, m__, ga__, day__, dax__, cos_da0__, cos_da1__

    return adert__


def angle_diff(a2, a1):
    sin_1 = a1[0]
    sin_2 = a2[0]

    cos_1 = a1[1]
    cos_2 = a2[1]

    # by the formulas of sine and cosine of difference of angles
    sin = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos = (sin_1 * cos_1) + (sin_2 * cos_2)

    return ma.array([sin, cos])


# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

