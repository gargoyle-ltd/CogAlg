import cv2
import argparse
import numpy
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in frame_draft.
    frame_dblobs() is updated version of frame_blobs with only one blob type: dblob, to ease debugging.
    frame_x_blob() forms dblobs only inside negative mblobs, to reduce redundancy

    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives ders ) array ders_
    2Le, line y- 1: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 ) array ders2_ 
    3Le, line y- 1+ rng*2: form_P(ders2) -> 1D pattern P
    4Le, line y- 2+ rng*2: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng*2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng*2+ seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  

    for y = rng *2: line y == P_, line y-1 == hP_, line y-2 == seg_, line y-4 == blob_

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.

    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    Subsequent union of lateral and vertical patterns is by strength only, orthogonal sign is not commeasurable?

    Initial pixel comparison is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
    It would be much faster with matrix computation, but this is minor compared to higher-level processing.
    I implement it sequentially for consistency with accumulation into blobs: irregular and very difficult to map to matrices.

    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, 
    convert elements of lower line into elements of new higher line, then displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
'''


# ************ UTILITY FUNCTIONS ****************************************************************************************
# Includes:
# -rebuild_blobs()
# ***********************************************************************************************************************

def rebuild_blobs(frame):
    " Rebuilt data of blobs into an image "
    blob_image = numpy.array([[[0] * 4] * X] * Y)

    for index, blob in enumerate(frame[9]):  # Iterate through blobs
        for seg in blob[1]:  # Iterate through segments
            y, ave_x = seg[7], seg[3]
            for (P, dx) in reversed(seg[5]):
                if P[0]:
                    x = P[1]
                    for i in range(P[2]):
                        blob_image[y, x, : 3] = [255, 255, 255]
                        x += 1
                else:
                    for dP in P[8]: # dPs inside of negative mP
                        x = dP[1]
                        for i in range(dP[2]):
                            blob_image[y, x, : 3] = [127, 127, 127] if dP[0] else [63, 63, 63]
                            x += 1
                blob_image[y, ave_x + P[1], :3] = [0, 0, 0] if P[0] else [0, 255, 255]  # output middle line of a segment based on ave_x
                y -= 1; ave_x -= dx # ave_x of next line

    return blob_image
    # ---------- rebuild_blobs() end ------------------------------------------------------------------------------------


# ************ UTILITY FUNCTIONS END ************************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp()
# -vertical_comp()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -image_to_blobs
# ***********************************************************************************************************************

def lateral_comp(pixel_):
    " Comparison over x coordinate, within rng of consecutive pixels on each line "

    ders1_ = []  # tuples of complete 1D derivatives: summation range = rng
    rng_ders1_ = deque(maxlen=rng)  # incomplete ders1s, within rng from input pixel: summation range < rng
    rng_ders1_.append((0, 0, 0))
    max_index = rng - 1  # max index of rng_ders1_

    for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
        back_fd, back_fm = 0, 0  # fuzzy derivatives from rng of backward comps per pri_p
        for index, (pri_p, fd, fm) in enumerate(rng_ders1_):
            d = p - pri_p
            m = ave - abs(d)
            fd += d  # bilateral fuzzy d: running sum of differences between pixel and all prior and subsequent pixels within rng
            fm += m  # bilateral fuzzy m: running sum of matches between pixel and all prior and subsequent pixels within rng
            back_fd += d
            back_fm += m  # running sum of d and m between pixel and all prior pixels within rng

            if index < max_index:
                rng_ders1_[index] = (pri_p, fd, fm)
            elif x > rng * 2 - 1:  # after pri_p comp over full bilateral rng
                ders1_.append((pri_p, fd, fm))  # completed bilateral tuple is transferred from rng_ders_ to ders_

        rng_ders1_.appendleft((p, back_fd, back_fm))  # new tuple with initialized d and m, maxlen displaces completed tuple
    # last incomplete rng_ders1_ in line are discarded, vs. ders1_ += reversed(rng_ders1_)
    ders1_.append( ( 0, 0, 0 ) )
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp(ders1_, ders2__, _xP_, _yP_, xframe, yframe):
    " Comparison to bilateral rng of vertically consecutive pixels, forming ders2: pixel + lateral and vertical derivatives"

    xP = 0, rng, 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, x0, L, I, D, Dy, V, Vy, ders2_
    yP = 0, rng, 0, 0, 0, 0, 0, 0, []
    xP_ = deque()  # line y - 1 + rng*2
    yP_ = deque()
    xbuff_ = deque()  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
    ybuff_ = deque()
    new_ders2__ = deque()  # 2D array: line of ders2_s buffered for next-line comp
    max_index = rng - 1  # max ders2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, d, m), ders2_ in zip(ders1_, ders2__):  # pixel comp to rng _pixels in ders2_, summing dy and my
        index = 0
        back_dy, back_my = 0, 0
        for (_p, _d, fdy, _m, fmy) in ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable

            dy = p - _p
            my = ave - abs(dy)
            fdy += dy  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmy += my  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dy += dy
            back_my += my  # running sum of d and m between pixel _p and all higher pixels within rng

            if index < max_index:
                ders2_[index] = (_p, _d, fdy, _m, fmy)
            elif y > min_coord + ini_y:
                ders = _p, _d, fdy, _m, fmy
                xP, xP_, xbuff_, _xP_, xframe = form_P(ders, x, xP, xP_, xbuff_, _xP_, xframe, 0)
                yP, yP_, ybuff_, _yP_, yframe = form_P(ders, x, yP, yP_, ybuff_, _yP_, yframe, 1)
            index += 1

        ders2_.appendleft((p, d, back_dy, m, back_my))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_ders2__.append(ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to ders2__, for next-line vertical comp
        x += 1

    return new_ders2__, xP_, yP_, xframe, yframe
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def form_P(ders, x, P, P_, buff_, hP_, frame, vert=0):
    " Initializes, accumulates, and terminates 1D pattern"

    p, d, dy, m, my = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    if vert:
        s = 1 if my > 0 else 0  # core = 0 is negative: no selection?
    else:
        s = 1 if m > 0 else 0
    pri_s, x0, L, I, D, Dy, M, My, ders_ = P

    if not (s == pri_s or x == rng) or x == X - rng:  # P is terminated
        if not pri_s:  # dPs formed inside of negative mP
            dP_ = [];
            dP = -1, x0, 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, Dy, M, My, ders_
            ders_.append((0, 0, 0, 0, 0))
            for i in range(L + 1):
                ip, id, idy, im, imy = ders_[i]
                if vert:
                    sd = 1 if idy > 0 else 0
                else:
                    sd = 1 if id > 0 else 0
                pri_sd, x0d, Ld, Id, Dd, Dyd, Md, Myd, sders_ = dP
                if (pri_sd != sd and not i == 0) or i == L:
                    dP_.append(dP)
                    x0d, Ld, Id, Dd, Dyd, Md, Myd, sders_ = x0 + i, 0, 0, 0, 0, 0, 0, []
                Ld += 1
                Id += ip
                Dd += id
                Dyd += idy
                Md += im
                Myd += imy
                sders_.append((ip, id, idy, im, imy))
                dP = sd, x0d, Ld, Id, Dd, Dyd, Md, Myd, sders_

            P = pri_s, x0, L, I, D, Dy, M, My, dP_

        if y == rng * 2 + ini_y:  # 1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
            P_.append([P, 0, [], x - 1])  # P, roots, _fork_, x
        else:
            P_, buff_, hP_, frame = scan_P_(x - 1, P, P_, buff_, hP_, frame)  # scans higher-line Ps for contiguity
            # x-1 for prior p
        x0, L, I, D, Dy, M, My, ders_ = x, 0, 0, 0, 0, 0, 0, []  # new P initialization

    L += 1  # length of a pattern, continued or initialized input and derivatives are accumulated:
    I += p  # summed input
    D += d  # lateral D
    Dy += dy  # vertical D
    M += m  # lateral M
    My += my  # vertical M
    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, x0, L, I, D, Dy, M, My, ders_
    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    ini_x = 0  # to start while loop, next ini_x = _x + 1

    while ini_x <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame = form_segment(hP_.popleft(), frame)
        else:
            break  # higher line ends, all hPs are converted to segments

        roots = hP[1]
        if P[0] == hP[6][0]:  # if s == _s: core sign match, + selective inclusion if contiguity eval?
            roots += 1;
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        _x = hP[5][-1][0][1] + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame = form_blob(hP, frame)  # segment is terminated and packed into its blob

        ini_x = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, roots = hP[:2]
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if y == rng * 2 + 1 + ini_y:  # scan_P_ of the 1st line converts each hP into initialized blob segment:
        hP[0] = list(_P[2:8])
        hP += 0, [(_P, 0)], [_P[0], 0, 0, 0, 0, 0, 0, _P[1], hP[3], y - rng - 1, [hP], 1, 0]  # initialize blob, min_y = current y
        hP[3] = ave_x
    else:
        if len(hP[2]) == 1 and hP[2][0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, Dx, Py_, blob) at hP[2][0]:
            s, x0, L, I, D, Dy, M, My, ders_ = _P
            Ls, Is, Ds, Dys, Ms, Mys = hP[2][0][0]
            hP[2][0][0] = [Ls + L, Is + I, Ds + D, Dys + Dy, Ms + M, Mys + My]  # seg parameters
            hP[2][0][1] = roots
            dx = ave_x - hP[2][0][3]
            hP[2][0][3] = ave_x
            hP[2][0][4] += dx  # xD for seg normalization and orientation, or += |dx| for curved yL?
            hP[2][0][5].append((_P, dx))  # Py_: vertical buffer of Ps merged into seg
            blob = hP[2][0][6]
            blob[7] = min(_P[1], blob[7])  # min_x
            blob[8] = max(hP[3], blob[8])  # max_x
            hP = hP[2][0]  # replace segment with including fork's segment

        elif not hP[2]:  # seg is initialized with initialized blob
            hP[0] = list(_P[2:8])  # seg parameters
            hP += 0, [(_P, 0)], [_P[0], 0, 0, 0, 0, 0, 0, _P[1], hP[3], y - rng - 1, [hP], 1, 0]
            # last var is blob: s, L, I, D, Dy, M, My, min_x, max_x, min_y, root_, remaining_roots, xD
            hP[3] = ave_x

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP[0] = list(_P[2:8])
            hP += 0, [(_P, 0)], hP[2][0][6]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[10].append(hP)  # segment is buffered into root_
            blob[7] = min(_P[1], blob[7])  # min_x
            blob[8] = max(hP[3], blob[8])  # max_x
            hP[3] = ave_x

            if len(hP[2]) > 1:  # merge blobs of all forks
                if hP[2][0][1] == 1:
                    frame = form_blob(hP[2][0], frame, 1)  # merge seg of 1st fork into its blob

                for fork in hP[2][1:len(hP[2])]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame = form_blob(fork, frame, 1)

                    if not fork[6] is blob:
                        blob[1] += fork[6][1]
                        blob[2] += fork[6][2]
                        blob[3] += fork[6][3]
                        blob[4] += fork[6][4]
                        blob[5] += fork[6][5]
                        blob[6] += fork[6][6]
                        blob[7] = min(fork[6][7], blob[7])
                        blob[8] = max(fork[6][8], blob[8])
                        blob[9] = min(fork[6][9], blob[9])
                        blob[11] += fork[6][11]
                        blob[12] += fork[6][12]
                        for seg in fork[6][10]:
                            if not seg is fork:
                                seg[6] = blob  # blobs in other forks are references to blob in the first fork
                                blob[10].append(seg)  # buffer of merged root segments
                        fork[6] = blob
                        blob[10].append(fork)
                    blob[11] -= 1
    return hP, frame
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "

    [L, I, D, Dy, M, My], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[1] += L
    blob[2] += I
    blob[3] += D
    blob[4] += Dy
    blob[5] += M
    blob[6] += My
    blob[11] += roots - 1  # reference to term_seg is already in blob[9]
    blob[12] += xD  # ave_x angle, to evaluate blob for re-orientation
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: elevation of term_seg over current hP

    if not blob[11]:  # if remaining_roots == 0: blob is terminated and packed in frame
        s, L, I, D, Dy, M, My, min_x, max_x, min_y, root_, remaining_roots, xD = blob
        frame[0] += L  # frame P are to compute averages, redundant for same-scope alt_frames
        frame[1] += I
        frame[2] += D
        frame[3] += Dy
        frame[4] += M
        frame[5] += My
        frame[6] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[7] += max_x - min_x + 1  # blob width
        frame[8] += term_seg[7] - min_y + 1  # blob height
        frame[9].append(((s, L, I, D, Dy, M, My, min_x, max_x, min_y, term_seg[7]), root_, xD))

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


def image_to_blobs(image):
    " Main body of the operation, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    _xP_ = deque()  # higher-line same- d-, m-, dy-, my- sign 1D patterns
    _yP_ = deque()
    xframe = [0, 0, 0, 0, 0, 0, 0, 0, 0, []]  # L, I, D, Dy, M, My, xD, b_width, b_height, blob_
    yframe = [0, 0, 0, 0, 0, 0, 0, 0, 0, []]
    global y
    y = ini_y  # initial line
    ders2__ = []  # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    pixel_ = image[ini_y, :]  # first line of pixels at y == 0
    ders1_ = lateral_comp(pixel_)  # after partial comp, while x < rng?

    for (p, d, m) in ders1_:
        ders2 = p, d, 0, m, 0  # dy, my initialized at 0
        ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
        ders2_.append(ders2)  # only one tuple in first-line ders2_
        ders2__.append(ders2_)

    for y in range(ini_y + 1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, _xP_, _yP_, xframe, yframe = vertical_comp(ders1_, ders2__, _xP_, _yP_, xframe, yframe)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # merge segs of last line into their blobs:
    y = Y
    hP_ = _xP_
    while hP_:
        hP, xframe = form_segment(hP_.popleft(), xframe)
        xframe = form_blob(hP, xframe)
    hP_ = _yP_
    while hP_:
        hP, yframe = form_segment(hP_.popleft(), yframe)
        yframe = form_blob(hP, yframe)
    return (xframe, yframe)  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of pixels compared to each pixel in four directions
ave = 15  # |d| value that coincides with average match: mP filter
ave_rate = 0.25  # not used; match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 0  # not used

# Load inputs --------------------------------------------------------------------
# image = misc.face(gray=True)  # read image as 2d-array of pixels (gray scale):
# image = image.astype(int)
# or:
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_eye.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

Y, X = image.shape  # image height and width

# Main ---------------------------------------------------------------------------
start_time = time()
frame_of_blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
cv2.imwrite('./images/blobs_horizontal.jpg', rebuild_blobs(frame_of_blobs[0]))
cv2.imwrite('./images/blobs_vertical.jpg', rebuild_blobs(frame_of_blobs[1]))

# Check for redundant segments  --------------------------------------------------
print 'Searching for redundant segments...\n'
for blob in frame_of_blobs[0][9]:
    for i, seg in enumerate(blob):
        for j, seg2 in enumerate(blob):
            if i != j and seg is seg2: print 'Redundant segment detected!\n'

# ************ PROGRAM BODY END ******************************************************************************************