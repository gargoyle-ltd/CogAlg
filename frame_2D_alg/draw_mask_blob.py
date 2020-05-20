from CogAlg.frame_2D_alg.frame_blobs import *
from CogAlg.frame_2D_alg.intra_comp import *
import numpy as np
import cv2

IMAGE_PATH = "./images/raccoon.jpg"
OUTPUT_PATH = "./images/mask_blob_full"
image = imread(IMAGE_PATH)
frame = image_to_blobs(image)
aveB = 10000

def mask_draw(dert_, coordinates, image, counter, flag_r):

    if flag_r:
        for val in range(len(dert_.mask)):
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
            cv2.imwrite("images/mask_blob_full/{0}_{1}.jpg".format(counter, val), img_mask)


    else:
        for val in range(len(dert_.mask)):
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


counter = 0
for blob in frame['blob__']:
    counter += 1
    im_to_draw = imread(IMAGE_PATH)

    if blob['sign']:
        if blob['Dert']['G'] > aveB and blob['Dert']['S'] > 20:
            print(counter)

            dert__ = comp_g(blob['dert__'], flag=False)
            mask_draw(dert__, blob['box'], im_to_draw, counter, flag_r=False)


    elif -blob['Dert']['G'] > aveB and blob['Dert']['S'] > 30:
        print(counter, 'comp_r')

        dert__ = comp_r(blob['dert__'], 0, False)
        mask_draw(dert__, blob['box'], im_to_draw, counter, flag_r=True)


