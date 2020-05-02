from CogAlg.frame_2D_alg.comp_pixel import comp_pixel
from CogAlg.frame_2D_alg.utils import *
from CogAlg.frame_2D_alg.frame_blobs import *

IMAGE_PATH = "./images/raccoon.jpg"
image = imread(IMAGE_PATH)

#frame = image_to_blobs(image)

def draw_blobs(frame, dert__select):
    box_list = []

    # loop across blobs
    for i, blob in enumerate(frame['blob__']):
        print('Blob number {}'.format(i))
        img_blobs = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2]))

        # if there is unmask dert
        if False in blob['dert__'][0].mask:

            # get dert value from blob
            dert__ = blob['dert__'][dert__select].data
            # get the index of mask
            mask_index = np.where(blob['dert__'][0].mask == True)
            nonmask_index = np.where(blob['dert__'][0].mask == False)

            # set masked area as 0
            dert__[mask_index] = 0
            dert__[nonmask_index] = 500

            # draw blobs into image
            img_blobs[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3]] = dert__
            # box_list.append(blob['box'])

            iblobs = img_blobs.astype('uint8')

            cv2.imwrite("images/box_draw1/iblobs_draft_{}.png".format(i), iblobs)

            img1 = cv2.imread("images/box_draw1/iblobs_draft_{}.png".format(i))


            img1 = cv2.rectangle(img1, (blob['box'][2], blob['box'][0]), (blob['box'][3], blob['box'][1]),
                                 color=(0, 0, 750), thickness=1)

            cv2.imwrite("images/box_draw/iblobs_draft_{0}_{1}.png".format(i, blob['box']), img1)


#draw_blobs(frame, dert__select=1)

def form_P_(dert__):  # horizontal clustering and summation of dert params into P params, per row of a frame
    # P is a segment of same-sign derts in horizontal slice of a blob

    P_ = deque()  # row of Ps
    I, G, Dy, Dx, L, x0 = *dert__[0], 1, 0  # initialize P params with 1st dert params
    G = int(G) - ave
    _s = G > 0  # sign
    for x, (p, g, dy, dx) in enumerate(dert__[1:], start=1):
        vg = int(g) - ave  # deviation of g
        s = vg > 0
        if s != _s:
            P = dict(L=L, x0=x0, dert__=dert__[x0:x0 + L], sign=_s)
            P_.append(P)
            L, x0 = 0, x

        L += 1
        _s = s  # prior sign

    P = dict(L=L, x0=x0, dert__=dert__[x0:x0 + L], sign=_s)
    P_.append(P)
    return P_


def scan_P_(P_, stack_, frame):

    next_P_ = deque()

    if P_ and stack_:

        P = P_.popleft()
        stack = stack_.popleft()
        _P = stack['Py_'][-1]
        up_fork_ = []

        while True:

            x0 = P['x0']
            xn = x0 + P['L']
            _x0 = _P['x0']
            _xn = _x0 + _P['L']

            if (P['sign'] == stack['sign']
                    and _x0 < xn and x0 < _xn):
                stack['down_fork_cnt'] += 1
                up_fork_.append(stack)

            if xn < _xn:
                next_P_.append((P, up_fork_))
                up_fork_ = []
                if P_:
                    P = P_.popleft()
                else:
                    if stack['down_fork_cnt'] != 1:
                        form_blob(stack, frame)
                    break
            else:
                if stack['down_fork_cnt'] != 1:
                    form_blob(stack, frame)

                if stack_:
                    stack = stack_.popleft()
                    _P = stack['Py_'][-1]
                else:
                    next_P_.append((P, up_fork_))
                    break

    while P_:
        next_P_.append((P_.popleft(), []))
    while stack_:
        form_blob(stack_.popleft(), frame)

    return next_P_


def form_stack_(y, P_, frame):

    next_stack_ = deque()

    while P_:
        P, up_fork_ = P_.popleft()
        s = P.pop('sign')
        L, x0, dert__ = P.values()
        xn = x0 + L
        if not up_fork_:

            blob = dict(Dert=dict(S=0, Ly=0), box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = dict(S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_fork_cnt=0, sign=s)
            blob['stack_'].append(new_stack)
        else:
            if len(up_fork_) == 1 and up_fork_[0]['down_fork_cnt'] == 1:

                new_stack = up_fork_[0]
                accum_Dert(new_stack, S=L, Ly=1)
                new_stack['Py_'].append(P)
                new_stack['down_fork_cnt'] = 0
                blob = new_stack['blob']

            else:
                blob = up_fork_[0]['blob']
                new_stack = dict(S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_fork_cnt=0, sign=s)
                blob['stack_'].append(new_stack)

                if len(up_fork_) > 1:
                    if up_fork_[0]['down_fork_cnt'] == 1:
                        form_blob(up_fork_[0], frame)

                    for up_fork in up_fork_[1:len(up_fork_)]:
                        if up_fork['down_fork_cnt'] == 1:
                            form_blob(up_fork, frame)

                        if not up_fork['blob'] is blob:
                            Dert, box, stack_, s, open_stacks = up_fork['blob'].values()  # merged blob
                            S, Ly = Dert.values()
                            accum_Dert(blob['Dert'], S=S, Ly=Ly)
                            blob['open_stacks'] += open_stacks
                            blob['box'][0] = min(blob['box'][0], box[0])  # extend box y0
                            blob['box'][1] = min(blob['box'][1], box[1])  # extend box x0
                            blob['box'][2] = max(blob['box'][2], box[2])  # extend box xn
                            for stack in stack_:
                                if not stack is up_fork:
                                    stack[
                                        'blob'] = blob
                                    blob['stack_'].append(stack)
                            up_fork['blob'] = blob
                            blob['stack_'].append(up_fork)
                        blob['open_stacks'] -= 1

        blob['box'][1] = min(blob['box'][1], x0)
        blob['box'][2] = max(blob['box'][2], xn)
        next_stack_.append(new_stack)

    return next_stack_


def form_blob(stack, frame):
    S, Ly, y0, Py_, blob, down_fork_cnt, sign = stack.values()
    accum_Dert(blob['Dert'], S=S, Ly=Ly)

    blob['open_stacks'] += down_fork_cnt - 1

    if blob['open_stacks'] == 0:
        last_stack = stack

        Dert, [y0, x0, xn], stack_, s, open_stacks = blob.values()
        yn = last_stack['y0'] + last_stack['Ly']

        mask = np.ones((yn - y0, xn - x0), dtype=bool)
        for stack in stack_:
            stack.pop('sign')
            stack.pop('down_fork_cnt')
            for y, P in enumerate(stack['Py_'], start=stack['y0'] - y0):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y, x_start:x_stop] = False
        dert__ = frame['dert__'][:, y0:yn, x0:xn].copy()
        dert__.mask[:] = mask  # default mask is all 0s

        blob.pop('open_stacks')
        blob.update(root=frame,
                    box=(y0, yn, x0, xn),  # boundary box
                    dert__=dert__,  # includes mask, no need for map
                    fork=defaultdict(dict),  # will contain fork params, layer_
                    blobs_in=0
                    )
        for i, blob2 in enumerate(frame['blob__']):
            if blob['sign'] == blob2['sign']:
                if blob['box'][0] <= blob2['box'][0] and \
                        blob['box'][1] >= blob2['box'][1] and \
                        blob['box'][2] <= blob2['box'][2] and \
                        blob['box'][3] >= blob2['box'][3]:
                    blob['blobs_in'] += 1

                elif blob2['box'][0] <= blob['box'][0] and \
                        blob2['box'][1] >= blob['box'][1] and \
                        blob2['box'][2] <= blob['box'][2] and \
                        blob2['box'][3] >= blob['box'][3]:
                    blob2['blobs_in'] += 1

        frame['blob__'].append(blob)



def image_to_blobs(image):

    dert__ = comp_pixel(image)

    frame = dict(rng=1, dert__=dert__, mask=None, blob__=[])
    stack_ = deque()
    height, width = dert__.shape[1:]

    for y in range(height):
        print(f'Processing line {y}...')
        P_ = form_P_(dert__[:, y].T)
        P_ = scan_P_(P_, stack_, frame)
        stack_ = form_stack_(y, P_, frame)

    while stack_:
        form_blob(stack_.popleft(), frame)

    return frame


frame = image_to_blobs(image)

for i in frame['blob__']:
    print(i['box'], i['blobs_in'])

