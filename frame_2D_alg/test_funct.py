import numpy as np
import numpy.ma as ma
import cv2
from CogAlg.frame_2D_alg.frame_blobs import ave, image_to_blobs
from CogAlg.frame_2D_alg.comp_pixel import comp_pixel

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

'''print(ma.getmask(a))
print(a)

for x in range(len(a)):
    print(x)
    for y in range(len(a[x])):
        print(a[x][y], '|', x, '|', y)
        if ma.is_masked(a[x][y]):
             print('True', x)'''


coordinates = []
image = cv2.imread('images/raccoon.jpg', 0).astype(int)

frame = image_to_blobs(image)
for blob__ in frame['blob__']:

    print('_________________________________________________________')

    # print(blob__['dert__'][0].mask)
    # print(blob__['dert__'][0])
    # print(blob__['dert__'][3])
    if blob__['dert__'][0].mask.all():
        print(blob__['dert__'][0].mask)
        print('------------------------')

        print(blob__['dert__'][0])
        print('------------------------')

        print(blob__['dert__'][3])
        print('------------------------')
        coordinates.append(blob__['box'])


check_image = comp_pixel(image)
# (y0, yn, x0, xn)
print('On the picture')
for i in coordinates:
    print(check_image[0][i[0]:i[1], i[2]:i[3]])


'''    for y in range(len(blob__['dert__'][0])):
        for x in range(len(blob__['dert__'][0][y])):

            if not blob__['dert__'].mask[0][y][x]:
                print(True, 'Mask', blob__['dert__'][0][y][x])
                print('x0 = {0}, y0 = {1}'.format(x, y))
                print(len(blob__['dert__'].mask), len(blob__['dert__']))
                break
        break'''

_mask = False

'''mask1 = np.array([[False, False, False],
         [False, False, False],
         [False, False, False]])

mask2 = np.array([[True, True, True],
         [True, True, True],
         [True, True, True]])

mask3 = np.array([[]])

a = mask1.all() or mask3.all()
b = mask1.all() or mask2.all()
c = mask1.all() and mask3.all()

print(a, b, c)
print(np.hypot(0, 0)>0)'''

