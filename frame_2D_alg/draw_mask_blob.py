from CogAlg.frame_2D_alg.frame_blobs import *
from CogAlg.frame_2D_alg.intra_comp import *
import numpy as np
import cv2

IMAGE_PATH = "./images/raccoon_eye.jpg"
OUTPUT_PATH = "./images/mask_blob"
image = imread(IMAGE_PATH)
frame = image_to_blobs(image)
aveB = 10000

def mask_draw(dert_, coordinates, image, counter):

    for val in range(len(dert_.mask)):
        mask = np.zeros((dert_[val].shape[0], dert_[val].shape[1]), dtype=int)

        for row in range(len(dert_[val])):
            for col in range(len(dert_[val][row])):
                if not dert_.mask[val][row][col]:
                    mask[row][col] = 1


        new_img = image
        for i in range(len(new_img[coordinates[0]: mask.shape[0]])):
            for j in range(len(new_img[i][coordinates[2]:mask.shape[1]])):
                if mask[i][j] != 0:
                    new_img[coordinates[0] + i][coordinates[2] + j] = 0
                else:
                    new_img[coordinates[0] + i][coordinates[2] + j] = 250

        img_mask = new_img.astype(np.uint8)
        cv2.imwrite("images/mask_blob/{0}_{1}.jpg".format(counter, val), img_mask)

counter = 0
for blob in frame['blob__']:
    counter += 1

    if blob['sign']:
        if blob['Dert']['G'] > aveB and blob['Dert']['S'] > 20:
            dert__ = comp_g(blob['dert__'])
            mask_draw(dert__, blob['box'], image, counter)


    elif -blob['Dert']['G'] > aveB and blob['Dert']['S'] > 30:
        dert__ = comp_r(blob['dert__'], 0, blob['root']['rng'])
        mask_draw(dert__, blob['box'], image, counter)


