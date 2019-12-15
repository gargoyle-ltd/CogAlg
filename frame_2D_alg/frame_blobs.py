from time import time
from collections import deque, defaultdict
import numpy as np
import numpy.ma as ma
import cv2

'''
    2D version of first-level core algorithm will have frame_blobs, intra_blob (recursive search within blobs), and comp_P.
    frame_blobs() forms parameterized blobs: contiguous areas of positive or negative deviation of gradient per pixel.    
    
    comp_pixel (lateral, vertical, diagonal) forms derts ) dert__: tuples of pixel + derivatives, over the whole frame. 
    Then pixel-level and external parameters are accumulated in row segment Ps, vertical blob segments, and blobs,
    adding a level of encoding per row y, defined relative to y of current input row, with top-down scan:

    1Le, line y-1: form_P( dert_) -> 1D pattern P: contiguous row segment, a slice of blob
    2Le, line y-2: scan_P_(P, hP) -> hP, up_fork_, down_fork_cnt: vertical connections per blob segment
    3Le, line y-3: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    4Le, line y-4+ seg depth: form_blob(seg, blob): merge connected segments in fork_ incomplete blobs, recursively

    Higher-line elements include additional parameters, derived while they were lower-line elements.
    Processing is mostly sequential because blobs are irregular, not suited for matrix operations.
    Resulting blob structure (fixed set of parameters per blob): 
    
    - root_fork = frame,  # replaced by blob-level fork in sub_blobs
    - Dert = dict(I, G, Dy, Dx, S, Ly), # summed pixel dert params (I, G, Dy, Dx), surface area S, vertical depth Ly
    - sign = s,  # sign of gradient deviation
    - box  = [y0, yn, x0, xn], 
    - map, # inverted mask
    - dert__,  # 2D array of pixel-level derts: (p, g, dy, dx) tuples
    - segment_,  # contains intermediate structures: blob segments ( Ps: row segments
    ( intra_blob extends Dert, adds crit, rng, fork_)
    
    prefix '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
    postfix '_' denotes array name, vs. same-name elements of that array
'''
# Constants: MAX_G = 256  # 721.2489168102785 without normalization.
# Adjustable parameters:

kwidth = 3  # input-centered, low resolution kernel: frame | blob shrink by 2 pixels per row,
# kwidth = 2  # co-centered, grid shift, 1-pixel row shrink, no deriv overlap, 1/4 chance of boundary pixel in kernel?
# kwidth = 2 quadrant: g = ((dx + dy) * .705 + d_diag) / 2, signed-> gPs? no i res-, ders co-location, + orthogonal quadrant for full rep?
ave = 50

# ----------------------------------------------------------------------------------------------------------------------------------------
# Functions

def image_to_blobs(image):  # root function, postfix '_' denotes array vs element, prefix '_' denotes higher- vs lower- line variable

    dert__ = comp_pixel(image)  # comparison of central pixel to rim pixels in a square kernel
    frame = dict(rng=1,
                 dert__=dert__,
                 mask=None,
                 I=0, G=0, Dy=0, Dx=0, blob_=[])

    seg_ = deque()  # buffer of running segments
    height, width = image.shape

    for y in range(height - kwidth + 1):  # first and last row are discarded
        P_ = form_P_(dert__[:, y].T)      # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)     # vertical clustering
        seg_ = form_seg_(y, P_, frame)

    while seg_:
        form_blob(seg_.popleft(), frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of blobs


def comp_pixel(image):  # 3x3 or 2x2 pixel cross-correlation within image

    if kwidth == 2:  # cross-compare four adjacent pixels diagonally:

        dy__ = (image[1:, 1:] - image[:-1, 1:]) + (image[1:, :-1] - image[:-1, :-1]) * 0.5
        dx__ = (image[1:, 1:] - image[1:, :-1]) + (image[:-1, 1:] - image[:-1, :-1]) * 0.5
        # sum pixel values:
        p__ = (image[:-1, :-1] + image[:-1, 1:] + image[1:, :-1] + image[1:, 1:]) * 0.25

    else:  # kwidth == 3, compare central pixel to 8 rim pixels, current default option

        ycoef = np.array([-0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0])  # this is equivalent to Sobel operator, but
        xcoef = np.array([-0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1])  # coefs scale diagonal vs. orthogonal pixels

        d___ = np.array(list(  # subtract centered image from translated image:
            map(lambda trans_slices: image[trans_slices] - image[1:-1, 1:-1],
                [
                    (slice(None, -2), slice(None, -2)),
                    (slice(None, -2), slice(1, -1)),
                    (slice(None, -2), slice(2, None)),
                    (slice(1, -1), slice(2, None)),
                    (slice(2, None), slice(2, None)),
                    (slice(2, None), slice(1, -1)),
                    (slice(2, None), slice(None, -2)),
                    (slice(1, -1), slice(None, -2)),
                ]
            )
        )).swapaxes(0, 2).swapaxes(0, 1)

        # Decompose differences into dy and dx, same as Gy and Gx in conventional edge detection operators:

        dy__ = (d___ * ycoef).sum(axis=2)
        dx__ = (d___ * xcoef).sum(axis=2)

        p__ = image[1:-1, 1:-1]

    g__ = np.hypot(dy__, dx__) * 0.354801226089485  # compute gradients per kernel, converted to 0-255 range

    return ma.around(ma.stack((p__, g__, dy__, dx__), axis=0))

''' 
Parameterized connectivity clustering functions below:

- form_P sums dert params within P and increments its L: horizontal length.
- scan_P_ searches for horizontal (x) overlap between Ps of consecutive (in y) rows.
- form_seg combines these overlapping Ps into non-forking blob segment: vertical stack of 1-above to 1-below Ps
- form_blob and terminate_segment combine terminated forking segments into blob
- terminate_blob eliminates redundant representations of the same blob by multiple forking segments, 
  then combines terminated blob into whole-frame representation.
  
dert is a tuple of derivatives per pixel, initially (p, dy, dx, g), will be extended in intra_blob
Dert is params of a composite structure (P, seg, blob): summed dert params + dimensions: vertical Ly and area S
'''

def form_P_(dert_):  # horizontal clustering and summation of dert params into P params, per row of a frame
    # P is contiguous segment in horizontal slice of a blob

    P_ = deque()  # row of Ps
    I, G, Dy, Dx, L, x0 = *dert_[0], 1, 0  # P params = first dert + init params
    G -= ave
    _s = G > 0  # sign

    for x, (i, g, dy, dx) in enumerate(dert_[1:], start=1):
        vg = g - ave
        s = vg > 0
        if s != _s:
            # terminate and pack P:
            P = dict(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, dert_=dert_[x0:x0+L], sign=_s)
            P_.append(P)
            # initialize new P:
            I, G, Dy, Dx, L, x0 = 0, 0, 0, 0, 0, x
        # accumulate P params:
        I += i
        G += vg  # M += m only within negative vg blobs
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = dict(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, dert_=dert_[x0:x0 + L], sign=_s)
    P_.append(P)  # terminate last P in a row
    return P_


def scan_P_(P_, seg_, frame):  # merge P into same-sign blob segments that contain higher-row _P with x-overlap
    """
    Each P in P_ scans higher-row _Ps (packed in seg_) left-to-right, until P.x0 >= _P.xn: no more x-overlap.
    Then scanning stops and P is packed into its up_fork segs or initializes a new seg.
    This x-overlap evaluation is also done for each _P, removing those that won't overlap next P.
    Segment that contains removed _P is packed in blob if its down_fork_cnt==0: no lower-row connections.
    """
    next_P_ = deque()  # to recycle P + up_fork_ that finished scanning _P, will be converted into next_seg_

    if P_ and seg_: # if both input row and higher row have any Ps / _Ps left
        P = P_.popleft()      # load left-most (lowest-x) input-row P
        seg = seg_.popleft()  # higher-row segments,
        _P = seg['Py_'][-1]   # last element of each segment is higher-row P
        up_fork_ = []         # list of same-sign x-overlapping _Ps per P

        while True:
            x0 = P['x0']      # first x in P
            xn = x0 + P['L']  # first x in next P
            _x0 = _P['x0']    # first x in _P
            _xn = _x0 + _P['L']  # first x in next _P

            if (P['sign'] == seg['sign']
                    and _x0 < xn and x0 < _xn):  # test for sign match and x overlap between loaded P and _P
                seg['down_fork_cnt'] += 1
                up_fork_.append(seg)  # P-connected higher-row segments are buffered into up_fork_ per P

            if xn < _xn:  # _P overlaps next P in P_
                next_P_.append((P, up_fork_))  # recycle _P for the next run of scan_P_
                up_fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if seg['down_fork_cnt'] != 1:  # terminate segment
                        form_blob(seg, frame)
                    break
            else:  # no next-P overlap
                if seg['down_fork_cnt'] != 1:  # terminate segment
                    form_blob(seg, frame)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg['Py_'][-1]
                else:  # no seg left: terminate loop
                    next_P_.append((P, up_fork_))
                    break

    while P_:  # terminate Ps and segs that continue at row's end
        next_P_.append((P_.popleft(), []))  # no up_fork
    while seg_:
        form_blob(seg_.popleft(), frame)  # down_fork_cnt always == 0

    return next_P_  # each element is P + up_fork_ refs


def form_seg_(y, P_, frame):
    """ Convert or merge every P into blob segment, merge blobs."""
    next_seg_ = deque()

    while P_:
        P, up_fork_ = P_.popleft()
        s = P.pop('sign')
        I, G, Dy, Dx, L, x0, dert_ = P.values()
        xn = x0 + L   # next-P x0
        if not up_fork_:
            # initialize blob segments for each input-row P that has no connections in higher row:
            blob = dict(Dert=dict(I=0, G=0, Dy=0, Dx=0, S=0, Ly=0), box=[y, x0, xn], seg_=[], sign=s, open_segments=1)
            next_seg = dict(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_fork_cnt=0, sign=s)
            blob['seg_'].append(next_seg)
        else:
            if len(up_fork_) == 1 and up_fork_[0]['down_fork_cnt'] == 1:
                # P has one up_fork and that up_fork has one root: merge P into up_fork segment:
                next_seg = up_fork_[0]
                accum_Dert(next_seg, I=I, G=G, Dy=Dy, Dx=Dx, S=L, Ly=1)
                next_seg['Py_'].append(P)  # Py_: vertical buffer of Ps
                next_seg['down_fork_cnt'] = 0  # reset down_fork_cnt
                blob = next_seg['blob']

            else:  # if > 1 up_forks, or 1 up_fork that has > 1 down_fork_cnt:
                blob = up_fork_[0]['blob']
                # initialize next_seg with up_fork blob:
                next_seg = dict(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_fork_cnt=0, sign=s)
                blob['seg_'].append(next_seg)  # segment is buffered into blob

                if len(up_fork_) > 1:  # merge blobs of all up_forks
                    if up_fork_[0]['down_fork_cnt'] == 1:  # up_fork is not terminated
                        form_blob(up_fork_[0], frame)  # merge seg of 1st up_fork into its blob

                    for up_fork in up_fork_[1:len(up_fork_)]:  # merge blobs of other up_forks into blob of 1st up_fork
                        if up_fork['down_fork_cnt'] == 1:
                            form_blob(up_fork, frame)

                        if not up_fork['blob'] is blob:
                            Dert, box, seg_, s, open_segs = up_fork['blob'].values()  # merged blob
                            I, G, Dy, Dx, S, Ly = Dert.values()
                            accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
                            blob['open_segments'] += open_segs
                            blob['box'][0] = min(blob['box'][0], box[0])  # extend box y0
                            blob['box'][1] = min(blob['box'][1], box[1])  # extend box x0
                            blob['box'][2] = max(blob['box'][2], box[2])  # extend box xn
                            for seg in seg_:
                                if not seg is up_fork:
                                    seg['blob'] = blob  # blobs in other up_forks are references to blob in the first up_fork.
                                    blob['seg_'].append(seg)  # buffer of merged root segments.
                            up_fork['blob'] = blob
                            blob['seg_'].append(up_fork)
                        blob['open_segments'] -= 1  # overlap with merged blob.

        blob['box'][1] = min(blob['box'][1], x0)  # extend box x0
        blob['box'][2] = max(blob['box'][2], xn)  # extend box xn
        next_seg_.append(next_seg)

    return next_seg_


def form_blob(seg, frame):  # terminated segment is merged into continued or initialized blob (all connected segments)

    blob = terminate_segment(seg)
    if blob['open_segments'] == 0:  # if number of incomplete segments == 0: blob is terminated and packed in frame
        terminate_blob(blob, seg, frame)


def terminate_segment(seg):
    I, G, Dy, Dx, S, Ly, y0, Py_, blob, down_fork_cnt, sign = seg.values()
    accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)

    blob['open_segments'] += down_fork_cnt - 1  # number of incomplete segments
    return blob


def terminate_blob(blob, last_seg, frame):
    Dert, [y0, x0, xn], seg_, s, open_segs = blob.values()
    yn = last_seg['y0'] + last_seg['Ly']

    mask = np.ones((yn - y0, xn - x0), dtype=bool)  # map of blob in coord box
    for seg in seg_:
        seg.pop('sign')
        seg.pop('down_fork_cnt')
        for y, P in enumerate(seg['Py_'], start=seg['y0'] - y0):
            x_start = P['x0'] - x0
            x_stop = x_start + P['L']
            mask[y, x_start:x_stop] = False
    dert__ = frame['dert__'][:, y0:yn, x0:xn]
    dert__.mask[:] = mask  # default mask is all 0s

    blob.pop('open_segments')
    blob.update(box=(y0, yn, x0, xn),  # boundary box
                map=~mask,  # to compute overlap in comp_blob
                crit=1,     # clustering criterion is g
                rng=1,      # if 3x3 kernel
                dert__=dert__,   # dert__ + box replace slices=(Ellipsis, slice(y0, yn), slice(x0, xn))
                root_fork=frame,
                fork_=defaultdict(dict),  # or []? contains forks ( sub-blobs
                )
    frame.update(I=frame['I'] + blob['Dert']['I'],
                 G=frame['G'] + blob['Dert']['G'],
                 Dy=frame['Dy'] + blob['Dert']['Dy'],
                 Dx=frame['Dx'] + blob['Dert']['Dx'])

    frame['blob_'].append(blob)


# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


# -----------------------------------------------------------------------------
# Main

if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    image = cv2.imread(arguments['image'], 0).astype(int)

    start_time = time()
    frame_of_blobs = image_to_blobs(image)

    # from intra_blob import cluster_eval, intra_fork, cluster, aveF, aveC, aveB, etc.?
    '''
    frame_of_deep_blobs = {  # initialize frame_of_deep_blobs
        'blob_': [],
        'params': defaultdict(int, {
            'I': frame_of_blobs['I'],
            'G': frame_of_blobs['G'],
            'Dy': frame_of_blobs['Dy'],
            'Dx': frame_of_blobs['Dx'],
            # deeper params are initialized when they are fetched
        }),
    }
    for blob in frame_of_blobs['blob_']:  # evaluate recursive sub-clustering in each blob, via cluster_eval -> intra_fork
    
        if blob['Dert']['G'] > aveB:  # +G blob directly calls intra_fork(comp_g), no immediate sub-clustering
            intra_fork(blob, aveF, aveC, aveB, ave, rng * 2 + 1, 1, fig=0, fa=0)  # nI = 1: g
        
        elif -blob['Dert']['G'] > aveB: # -G blob, sub-clustering by -vg for rng+ eval
            cluster_eval(blob, aveF, aveC, aveB, ave, rng + 1, 2, fig=0, fa=0)  # cluster by -g for rng+, idiomatic crit=2: not index 

        frame_of_deep_blobs['blob_'].append(blob)
        frame_of_deep_blobs['params'][1:] += blob['params'][1:]  # incorrect, for selected blob params only?
    '''

    # DEBUG -------------------------------------------------------------------
    from utils import map_frame

    cv2.imwrite("./images/blobs.bmp", map_frame(frame_of_blobs))
    # END DEBUG ---------------------------------------------------------------

    end_time = time() - start_time
    print(end_time)