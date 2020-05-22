import cv2
import numpy as np

from CogAlg.frame_2D_alg.frame_blobs import *
from CogAlg.frame_2D_alg.intra_comp import *

IMAGE_PATH = "./images/raccoon.jpg"
OUTPUT_PATH = "./images/mask_blob_full"
image = imread(IMAGE_PATH)
frame = image_to_blobs(image)
aveB = 10000


def mask_draw(dert_, coordinates, image, counter, flag_r, file, list_of_g):
    if flag_r:
        for val in range(len(dert_.mask[:1])):
            mask = np.zeros((dert_[val].shape[0], dert_[val].shape[1]), dtype=int)
            new_img = image
            new_img[coordinates[0]: coordinates[1], coordinates[2]: coordinates[3]] = 0

            for i in range(len(new_img[coordinates[0]: mask.shape[0]])):
                for j in range(len(new_img[i][coordinates[2]:mask.shape[1]])):
                    if not dert_[val].mask[i][j]:
                        new_img[coordinates[0] + i * 2][coordinates[2] + j * 2] = 250

            img_mask = new_img.astype(np.uint8)
            cv2.rectangle(img_mask,
                          (coordinates[2], coordinates[0]),
                          (coordinates[3], coordinates[1]),
                          (250, 250, 250),
                          2)


    else:
        for val in range(len(dert_.mask[:1])):
            mask = np.zeros((dert_[val].shape[0], dert_[val].shape[1]), dtype=int)
            new_img = image
            # (y0, yn, x0, xn)
            new_img[coordinates[0]: coordinates[1], coordinates[2]: coordinates[3]] = 0
            for i in range(len(new_img[coordinates[0]: mask.shape[0]])):
                for j in range(len(new_img[i][coordinates[2]: mask.shape[1]])):
                    if not dert_[val].mask[i][j]:
                        new_img[coordinates[0] + i][coordinates[2] + j] = 250

            img_mask = new_img.astype(np.uint8)
            cv2.rectangle(img_mask,
                          (coordinates[2], coordinates[0]),
                          (coordinates[3], coordinates[1]),
                          (250, 250, 250),
                          2)
            cv2.imwrite("images/mask_blob_full/{0}_{1}.jpg".format(counter, val), img_mask)
        # print(list_of_g[0], list_of_g[1], list_of_g[2], list_of_g[3])
        # sum = functools.reduce(lambda x1, x2: x1.astype('int') + x2.astype('int'), list_of_g)



counter = 0
document = open('g_percentage.txt','w')

g1 = frame['dert__'][0][:-1, :-1]
g2 = [:-1, 1:].
g3 = [1:, 1:].c
g4 = [1:, :-1].

#for blob in frame['blob__']:
#    counter += 1
#    im_to_draw = imread(IMAGE_PATH)
#
#    if blob['sign']:
#        if blob['Dert']['G'] > aveB and blob['Dert']['S'] > 20:
#            print(counter)
#
#            dert__= comp_g(blob['dert__'], flag=False)
#            mask_draw(dert__, blob['box'], im_to_draw, counter, flag_r=False, file=document, list_of_g=[])
#
#
#    elif -blob['Dert']['G'] > aveB and blob['Dert']['S'] > 30:
#        print(counter, 'comp_r')
#
#        dert__ = comp_r(blob['dert__'], 0, False)
#        mask_draw(dert__, blob['box'], im_to_draw, counter, flag_r=True, file=document, list_of_g=[])

