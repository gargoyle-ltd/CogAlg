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
            g__topleft * cos_da0__ + g__topright * cos_da1__)) * 0.5

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


def comp_r(dert__, fig, root_fcr):
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


    if root_fcr:  # if
        # if root fork is comp_r, all params are present in the input:
        i__, g__, dy__, dx__ = dert__[0:4]  # top dimension of numpy stack must be a list

    else:  # root fork is comp_g or comp_pixel
        i__ = dert__[0]
        idy__ = np.zeros((i__.shape[0], i__.shape[1]))
        idx__ = np.zeros((i__.shape[0], i__.shape[1]))
        dy__ = np.zeros((i__.shape[0], i__.shape[1]))
        dx__ = np.zeros((i__.shape[0], i__.shape[1]))

    # i is ig if fig else pixel
    i__center =      i__[1 :-1:2, 1:-1 :2]
    i__topleft =     i__[:-2:2, :-2:2]
    i__top =         i__[:-2:2, 1:-1: 2]
    i__topright =    i__[:-2:2, 2::2]
    i__right =       i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom =      i__[2::2, 1:-1 :2]
    i__bottomleft =  i__[2::2, :-2:2]
    i__left =        i__[1:-1:2, :-2:2]


    if fig:
        if root_fcr:
            m__, idy__, idx__ = dert__[[-4, -2, -1]]
            m__ = m__[1:-1:2, 1:-1:2]

        else:
            m__, idy__, idx__ = [], [], []  # how do we initialize empty np.arrays of len=len(i__)?
            m__ = np.zeros((i__center.shape[0],i__center.shape[1]))

        a__ = [idy__, idx__] / i__  # i is input gradient

        a__center =      a__[:, 1:-1:2, 1:-1:2]
        a__topleft =     a__[:, :-2:2, :-2:2]
        a__top =         a__[:, :-2:2, 1:-1: 2]
        a__topright =    a__[:, :-2:2, 2::2]
        a__right =       a__[:, 1:-1:2, 2::2]
        a__bottomright = a__[:, 2::2, 2::2]
        a__bottom =      a__[:, 2::2, 1:-1:2]
        a__bottomleft =  a__[:, 2::2, :-2:2]
        a__left =        a__[:, 1:-1:2, :-2:2]


        # tuple of angle differences per direction:
        dat__ = np.stack(angle_diff(a__center, a__topleft),
                         angle_diff(a__center, a__top),
                         angle_diff(a__center, a__topright),
                         angle_diff(a__center, a__right),
                         angle_diff(a__center, a__bottomright),
                         angle_diff(a__center, a__bottom),
                         angle_diff(a__center, a__bottomleft),
                         angle_diff(a__center, a__left))

        # roll axis to enable coefficients multiplication
        dat__ = np.rollaxis(dat__, 0, 4)

        # y-decomposed difference between angles to center
        day__ = (dat__ * Y_COEFFS[1]).sum(axis=-1)

        # x-decomposed difference between angles to center
        dax__ = (dat__ * X_COEFFS[1]).sum(axis=-1)

        # calculating gradient
        ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

        # calculating match
        m__ =    np.minimum(i__center, (i__topleft * dat__[1][:, :, 0])) \
               + np.minimum(i__center, (i__top * dat__[1][:, :, 1])) \
               + np.minimum(i__center, (i__topright * dat__[1][:, :, 2])) \
               + np.minimum(i__center, (i__right * dat__[1][:, :, 3])) \
               + np.minimum(i__center, (i__bottomright * dat__[1][:, :, 4])) \
               + np.minimum(i__center, (i__bottom * dat__[1][:, :, 5])) \
               + np.minimum(i__center, (i__bottomleft * dat__[1][:, :, 6])) \
               + np.minimum(i__center, (i__left * dat__[1][:, :, 7]))

        # tuple of cosine differences per direction:
        dt__ = np.stack(((i__center - i__topleft * dat__[1][:, :, 0]),
                         (i__center - i__top * dat__[1][:, :, 1]),
                         (i__center - i__topright * dat__[1][:, :, 2]),
                         (i__center - i__right * dat__[1][:, :, 3]),
                         (i__center - i__bottomright * dat__[1][:, :, 4]),
                         (i__center - i__bottom * dat__[1][:, :, 5]),
                         (i__center - i__bottomleft * dat__[1][:, :, 6]),
                         (i__center - i__left * dat__[1][:, :, 7])))

        dt__ = np.rollaxis(dt__, 0, 3)

        # sparse dy and dx to enable the cumulative summation
        dy__ = dy__[1:-1:2, 1:-1:2]
        dx__ = dx__[1:-1:2, 1:-1:2]

        dy__ += (dt__ * Y_COEFFS[1]).sum(axis=-1)
        dx__ += (dt__ * X_COEFFS[1]).sum(axis=-1)

        g__ = np.hypot(dy__, dx__)


    else:
        '''
        # Sobel operator
        # |--(clockwise)--+  |--(clockwise)--+
        # YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
        #         0       0  ¦          -2       2  ¦
        #         1   2   1  ¦          -1   0   1  ¦
        '''
        YCOEF_ = [-1, -2, -1, 0, 1, 2, 1, 0]
        XCOEF_ = [-1, 0, 1, 2, 1, 0, -1, -2]

        # comparisons of pairs
        dt__ = np.stack(i__topleft - i__bottomright,
                        i__top - i__bottom,
                        i__topright - i__bottomleft,
                        i__right - i__left
                        )

        dy__ = np.zeros((i__center.shape[0], i__center.shape[1]))
        dx__ = np.zeros((i__center.shape[0], i__center.shape[1]))

        for i__rim, YCOEF, XCOEF in zip(dt__, YCOEF_, XCOEF_):

            # decompose differences into dy and dx, same as conventional Gy and Gx:
            dy__ += dt__ * YCOEF[:4]
            dx__ += dt__ * XCOEF[:4]

        g__ = np.hypot(dy__, dx__)

    # return dert__ with accumulated derivatives:
    if fig:
        rdert = i__, g__, dy__, dx__, m__, ga__, idy__, idx__
    else:
        rdert = i__, g__, dy__, dx__

    return rdert


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



    # preallocate size of arrays
    sin_da__ = [None] * 2
    cos_da__ = [None] * 2

    day__ = [None] * 2
    day__[0] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))
    day__[1] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))

    dax__ = [None] * 2
    dax__[0] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))
    dax__[1] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))

    a_rim__ = np.stack((a__topleft,
                        a__topright,
                        a__bottomright,
                        a__bottomleft))

    YCOEF_ = [-1, -1]
    XCOEF_ = [-1, 1]

    for i in range(2):
        # opposing difference
        sin_da__[i], cos_da__[i] = angle_diff(a_rim__[i], a_rim__[i + 2])

        # sine part of day
        day__[0] += sin_da__[i] * YCOEF_[i]
        # cosine part of day
        day__[1] += cos_da__[i]  # no sign based coefficient for cosine part
        # sine part of dax
        dax__[0] += sin_da__[i] * XCOEF_[i]
        # cossine part of dax
        dax__[1] += cos_da__[i]  # no sign based coefficient for cosine part

    # change adert to tuple as ga__,day__,dax__ would have different dimension compared to inputs
    adert__ = ma.stack((i__[:-1,:-1], g__[:-1,:-1], dy__[:-1,:-1], dx__[:-1,:-1], m__[:-1,:-1], ga__, *day__, *dax__,
                        cos_da__[0], cos_da__[1]))

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
