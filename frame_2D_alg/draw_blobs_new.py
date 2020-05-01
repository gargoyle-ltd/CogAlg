from CogAlg.frame_2D_alg.comp_pixel import comp_pixel
from CogAlg.frame_2D_alg.utils import *
from CogAlg.frame_2D_alg.frame_blobs import *

IMAGE_PATH = "./images/raccoon.jpg"
image = imread(IMAGE_PATH)

frame = image_to_blobs(image)

def draw_blobs(frame, dert__select):
    img_blobs = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2]))

    box_list = []

    # loop across blobs
    for i, blob in enumerate(frame['blob__']):

        # if there is unmask dert
        if False in blob['dert__'][0].mask:

            # get dert value from blob
            dert__ = blob['dert__'][dert__select].data
            # get the index of mask
            mask_index = np.where(blob['dert__'][0].mask == True)
            # set masked area as 0
            dert__[mask_index] = 0



            # draw blobs into image
            img_blobs[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3]] += dert__
            box_list.append(blob['box'])


            # uncomment to enable draw animation
    #            plt.figure(dert__select); plt.clf()
    #            plt.imshow(img_blobs.astype('uint8'))
    #            plt.title('Blob Number ' + str(i))
    #            plt.pause(0.001)

    return img_blobs.astype('uint8'), box_list

iblobs, box_list1 = draw_blobs(frame, dert__select=0)
gblobs, box_list2 = draw_blobs(frame, dert__select=1)




# save to disk
cv2.imwrite("images/iblobs_draft5.png", iblobs)
cv2.imwrite("images/gblobs_draft5.png", gblobs)

img1 = cv2.imread("images/iblobs_draft5.png")
img2 = cv2.imread("images/gblobs_draft5.png")

for i in range(len(box_list1)):
    img1 = cv2.rectangle(img1, (box_list1[i][0], box_list1[i][2]), (box_list1[i][1], box_list1[i][3]), color=(0, 0, 350),
                  thickness=1)
    img2 = cv2.rectangle(img2, (box_list2[i][0], box_list2[i][2]), (box_list2[i][1], box_list2[i][3]), color=(0, 0, 350),
                  thickness=1)

cv2.imwrite("images/iblobs_draft6.png", img1)
cv2.imwrite("images/gblobs_draft6.png", img2)