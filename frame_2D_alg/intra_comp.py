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

    def comp_g_draft(dert__):
        """
      Cross-comp of g or ga in 2x2 kernels
    ----------
    input dert = (i, g, dy, dx, ga, day, dax, da0, da1)
    output dert = (g, gg, dgy, dgx, gm, ga, day, dax)
    """
    g__, da0__, da1__ = dert__[1, -2, -1]  # input

    topleft__ = g__[:-1, :-1]
    topright__ = g__[:-1, 1:]
    botleft__ = g__[1:, :-1]
    botright__ = g__[1:, 1:]

    dgy__ = ((botleft__ + botright__) - (topleft__ * da0__[1] + topright__ * da1__[1])) * 0.5
    # diagonal from left
    dgx__ = ((topright__ + botright__) - (topleft__ * da0__[1] + botleft__ * da1__[1])) * 0.5
    # diagonal from right
    gg__ = np.hypot(dgy__, dgx__)
    # gradient of gradient

    gm0__ = min(botleft__, topright__ * da0__[1])  # g match = min(g, _g*cos(da))
    gm1__ = min(botright__, topleft__ * da1__[1])
    gm__ = gm0__ + gm1__

    gdert = ma.stack(g__, gg__, dgy__, dgx__, gm__, ga__=dert__[4], day_=dert__[5], dax=dert__[6])

    '''
    next comp_rg will use g, dgy, dgx
    next comp_agg will use ga, day, dax
    '''
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
    i__, g__, dy__, dx__, m__ = dert__[0:5]

    # get sparsed value
    ri__ = i__[1::2, 1::2]
    rg__ = g__[1::2, 1::2]
    rdy__ = dy__[1::2, 1::2]
    rdx__ = dx__[1::2, 1::2]
    rm__ = m__[1::2, 1::2]

    # get each direction
    ri__topleft = ri__[:-2, :-2]
    ri__top = ri__[:-2, 1:-1]
    ri__topright = ri__[:-2, 2:]
    ri__right = ri__[1:-1, 2:]
    ri__bottomright = ri__[2:, 2:]
    ri__bottom = ri__[2:, 1:-1]
    ri__bottomleft = ri__[2:, :-2]
    ri__left = ri__[1:-1, :-2]

    if fig:  # input is g
        # OUTDATED:

        # compute diagonal differences for g in 3x3 kernel
        drg__ = np.stack((ri__topleft - ri__bottomright,
                          ri__top - ri__bottom,
                          ri__topright - ri__bottomleft,
                          ri__right - ri__left))

        # compute a from the range dy,dx and g
        a__ = [rdy__, rdx__] / rg__

        # angles per direction:
        a__topleft = a__[:, :-2, :-2]
        a__top = a__[:, :-2, 1:-1]
        a__topright = a__[:, :-2, 2:]
        a__right = a__[:, 1:-1, 2:]
        a__bottomright = a__[:, 2:, 2:]
        a__bottom = a__[:, 2:, 1:-1]
        a__bottomleft = a__[:, 2:, :-2]
        a__left = a__[:, 1:-1, :-2]

        # compute opposing diagonals difference for a in 3x3 kernel
        dra__ = np.stack((angle_diff(a__topleft, a__bottomright),
                          angle_diff(a__top, a__bottom),
                          angle_diff(a__topright, a__bottomleft),
                          angle_diff(a__right, a__left)))

        # g difference  = g - g * cos(da) at each opposing diagonals
        dri__ = np.stack((drg__[0] - drg__[0] * dra__[0][1],
                          drg__[1] - drg__[1] * dra__[1][1],
                          drg__[2] - drg__[2] * dra__[2][1],
                          drg__[3] - drg__[3] * dra__[3][1]))

    else:
        a__ = []

        # angles per direction:
        a__topleft = a__[:, :-2, :-2]
        a__top = a__[:, :-2, 1:-1]
        a__topright = a__[:, :-2, 2:]
        a__right = a__[:, 1:-1, 2:]
        a__bottomright = a__[:, 2:, 2:]
        a__bottom = a__[:, 2:, 1:-1]
        a__bottomleft = a__[:, 2:, :-2]
        a__left = a__[:, 1:-1, :-2]

        # compute opposing diagonals difference for a in 3x3 kernel
        dra__ = np.stack((angle_diff(a__topleft, a__bottomright),
                          angle_diff(a__top, a__bottom),
                          angle_diff(a__topright, a__bottomleft),
                          angle_diff(a__right, a__left)))

        # g difference  = g - g * cos(da) at each opposing diagonals
        dri__ = np.stack((drg__[0] - drg__[0] * dra__[0][1],
                          drg__[1] - drg__[1] * dra__[1][1],
                          drg__[2] - drg__[2] * dra__[2][1],
                          drg__[3] - drg__[3] * dra__[3][1]))

    else:

    #  i difference of each opossing diagonals
    dri__ = np.stack((ri__topleft - ri__bottomright,
                      ri__top - ri__bottom,
                      ri__topright - ri__bottomleft,
                      ri__right - ri__left))


    dri__ = np.rollaxis(dri__, 0, 3)
    
    # compute dry and drx
    dry__ = (dri__ * Y_COEFFS[0][0:4]).sum(axis=-1)
    drx__ = (dri__ * X_COEFFS[0][0:4]).sum(axis=-1)
    
    # compute gradient magnitudes
    drg__ = ma.hypot(dry__, drx__)
    
    # pending m computation
    drm = []
    
    # rdert
    rdert = dri__, drg__, dry__, drx__, drm
    
    return rdert






def comp_a(dert__, fga):
    """
    Compute vector representation of gradient angle.
    It is done by normalizing the vector (dy, dx).
    Numpy broad-casting is a viable option when the
    first dimension of input array (dert__) separate
    different CogAlg parameter (like g, dy, dx).

    cross-comp of a or aga, in 2x2 kernels unless root fork is comp_r: odd=TRUE
    if aga:
         dert = g, gg, gdy, gdx, gm, iga, iday, idax
    else:
         dert = i, g, dy, dx, m
    <<< adert = ga, day, dax
    """

    # input dert = (i,  g,  dy,  dx,  m, ga, day, dax, dat(da0, da1, da2, da3))
    i__, g__, dy__, dx__, m__ = dert__[0:4]

    if fga:  # if input is adert

        ga__, day__, dax__ = dert__[5:8]
        a__ = [day__, dax__] / ga__  # similar to calc_a

    else:
        a__ = [dy__, dx__] / g__  # similar to calc_a

    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # each shifted a in 2x2 kernel
    a__topleft = a__[:, :-1, :-1]
    a__topright = a__[:, :-1, 1:]
    a__bottomright = a__[:, 1:, 1:]
    a__bottomleft = a__[:, 1:, :-1]

    # get angle difference of each direction
    da__ = np.stack((angle_diff(a__topleft, a__bottomright),
                     angle_diff(a__bottomleft, a__topright),
                     angle_diff(a__topright, a__bottomleft)))

    day__ = (
            Y_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
            Y_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
    )
    dax__ = (
            X_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
            X_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
    )
    # compute gradient magnitudes (how fast angles are changing)

    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

    if fga:  # if input is adert

        ga__, day__, dax__ = dert__[5:8]
        a__ = [day__, dax__] / ga__  # similar to calc_a

    else:
        a__ = [dy__, dx__] / g__  # similar to calc_a

    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # each shifted a in 2x2 kernel
    a__topleft = a__[:, :-1, :-1]
    a__topright = a__[:, :-1, 1:]
    a__bottomright = a__[:, 1:, 1:]
    a__bottomleft = a__[:, 1:, :-1]

    # get angle difference of each direction
    da__ = np.stack((angle_diff(a__topleft, a__bottomright),
                     angle_diff(a__bottomleft, a__topright),
                     angle_diff(a__topright, a__bottomleft)))

    day__ = (
            Y_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
            Y_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
    )
    dax__ = (
            X_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
            X_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
    )
    # compute gradient magnitudes (how fast angles are changing)

    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

    # change adert to tuple as ga__,day__,dax__ would have different dimension compared to inputs
    adert__ = i__, g__, dy__, dx__, m__, ga__, day__, dax__

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
