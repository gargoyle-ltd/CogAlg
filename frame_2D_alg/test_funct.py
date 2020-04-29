import numpy as np
import numpy.ma as ma
import cv2
from CogAlg.frame_2D_alg.frame_blobs import ave, image_to_blobs

from CogAlg.frame_2D_alg.utils import pairwise, flatten

IMAGE_PATH = "../images/raccoon.jpg"

# Outputs:
OUTPUT_PATH = "../visualization/images/"

a = ma.array([[0,0,0, 2, 2, 4],
             [0,0,0, 2, 2, 8]])

b = np.array([[3, 3, 3],
              [3, 3, 3]])

c = [a, b]

a1 = [1, 2, 4, 5, 6, 7, 8, 9]

print(ma.getmask(a))
print(a)

for x in range(len(a)):
    print(x)
    for y in range(len(a[x])):
        print(a[x][y], '|', x, '|', y)
        if ma.is_masked(a[x][y]):
             print('True', x)


image = cv2.imread('images/raccoon.jpg', 0).astype(int)

#frame = image_to_blobs(image)
'''for blob__ in frame['blob__']:

    print('_________________________________________________________')

    print(blob__['dert__'][0].mask)
    print(blob__['dert__'][0])
    for y in range(len(blob__['dert__'][0])):
        for x in range(len(blob__['dert__'][0][y])):

            if not blob__['dert__'].mask[0][y][x]:
                print(True, 'Mask', blob__['dert__'][0][y][x])
                print('x0 = {0}, y0 = {1}'.format(x, y))
                print(len(blob__['dert__'].mask), len(blob__['dert__']))
                break
        break'''

_mask = False

if ~_mask:
    print(True)

