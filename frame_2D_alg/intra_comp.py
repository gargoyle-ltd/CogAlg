
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
    cross-comp of g or ga, in 2x2 kernels unless root fork is comp_r: odd=TRUE
    or odd: sparse 3x3, is also effectively 2x2 input, recombined from one-line-distant lines?

     dert = i, g, dy, dx
     adert = ga, day, dax
     odd = bool  # initially FALSE, set to TRUE for comp_a and comp_g called from comp_r fork
    # comparand = dert[1]
    <<< gdert = g, gg, gdy, gdx, gm, ga, day, dax
    """

    # if adert = ga, day, dax
    if odd:
        ga, day, dax = dert__[:3]
        a__ = [day, dax] / ga

    # if dert = i, g, dy, dx
    else:
        g, dy, dx = dert__[1:4]
        a__ = [dy, dx] / g

    a__1 = a__[:, :-1, :-1]                         # top left
    a__2 = a__[:, :-1, 1:]                          # top right
    a__3 = a__[:, 1:, 1:]                           # bottom right
    a__4 = a__[:, 1:, :-1]                          # bottom left
    a_central = ((a__1 + a__2 + a__3 + a__4) / 4)   # central

    # angle difference
    da__ = np.stack((angle_diff(a_central, a__1),
                     angle_diff(a_central, a__2),
                     angle_diff(a_central, a__3),
                     angle_diff(a_central, a__4)))

    day = sum([da__[i][0] * Y_COEFFS[0][i] for i in range(len(da__))])
    dax = sum([da__[i][1] * X_COEFFS[0][i] for i in range(len(da__))])

    # calculating magnitude
    g = np.hypot(day, dax)

    #2x2 cross-comp
    g_1 = g__[:-1, :-1]
    g_2 = g__[:-1, 1:]
    g_3 = g__[1:, 1:]
    g_4 = g__[1:, :-1]


    # match?
    gm = ma.array(np.ones())

    # pack gdert
    gdert = ma.stack(g, gg, gdy, gdx, gm, day, dax)

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

    pass




def comp_a(dert__, fga, fc3):
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

    # computing a__ depending on flag
    a__ = compute_a__(dert__, fga)

    a__1 = a__[:, :-1, :-1]                         # top left
    a__2 = a__[:, :-1, 1:]                          # top right
    a__3 = a__[:, 1:, 1:]                           # bottom right
    a__4 = a__[:, 1:, :-1]                          # bottom left
    a_central = ((a__1 + a__2 + a__3 + a__4) / 4)   # central

    # angle difference
    da__ = np.stack((angle_diff(a_central, a__1),
                     angle_diff(a_central, a__2),
                     angle_diff(a_central, a__3),
                     angle_diff(a_central, a__4)))

    # multiply on coefficients
    day = sum([da__[i][0] * Y_COEFFS[0][i] for i in range(len(da__))])
    dax = sum([da__[i][1] * X_COEFFS[0][i] for i in range(len(da__))])

    # gradient magnitude
    ga__ = np.hypot(np.arctan(day), np.arctan(dax))

    adert = ma.stack((ga__, day, dax))

    return adert


def compute_a__(dert__, fga):
    # dert = i, g, dy, dx, m
    if fga:
        g, dy, dx = dert__[1:4]
        a__ = [dy, dx] / g

    # if dert = g, gg, gdy, gdx, gm, iga, iday, idax
    else:
        ga, day, dax = dert__[:3]
        a__ = [day, dax] / ga

    return a__


def angle_diff(a2, a1):
    sin_1 = a1[0]
    sin_2 = a2[0]

    cos_1 = a1[1]
    cos_2 = a2[1]

    # by the formulas of sine and cosine of difference of angles
    sin = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos = (sin_1 * cos_1) + (sin_2 * cos_2)

    return ma.array([sin, cos])


def sparse_dert(dert__):

    dert = dert__[:, 1::2, 1::2]
    return dert

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------