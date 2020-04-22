

import CogAlg.frame_2D_alg.frame_blobs
from CogAlg.frame_2D_alg.intra_comp import *
from CogAlg.frame_2D_alg.utils import imread, imwrite
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "./images/raccoon.jpg"
# Outputs:
OUTPUT_PATH = "./images/intra_comp/"


# -----------------------------------------------------------------------------
# Functions

def draw_m(img_out, m_):
    for y in range(m_.shape[0]):  # loop rows, skip last row
        for x in range(m_.shape[1]):  # loop columns, skip last column
            img_out[y, x] = m_[y, x]

    return img_out.astype('uint8')


def draw_ma(img_out, m_):
    for y in range(m_.shape[0]):
        for x in range(m_.shape[1]):
            img_out[y, x] = m_[y, x]

    img_out = img_out * 180 / np.pi  # convert to degrees
    img_out = (img_out / 180) * 255  # scale 0 to 180 degree into 0 to 255

    return img_out.astype('uint8')


def draw_mr(img_out, m_, rng):
    for y in range(m_.shape[0]):
        for x in range(m_.shape[1]):

            # project central dert to surrounding rim derts
            img_out[(y * rng) + 1:(y * rng) + 1 + rng, (x * rng) + 1:(x * rng) + 1 + rng] = m_[y, x]

    return img_out.astype('uint8')


def draw_mar(img_out, m_, rng):
    for y in range(m_.shape[0]):
        for x in range(m_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y * rng) + 1:(y * rng) + 3, (x * rng) + 1:(x * rng) + 3] = m_[y, x]

    img_out = img_out * 180 / np.pi  # convert to degrees
    img_out = (img_out / 180) * 255  # scale 0 to 180 degree into 0 to 255

    return img_out.astype('uint8')


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    print('Reading image...')
    image = imread(IMAGE_PATH)

    dert_ = CogAlg.frame_2D_alg.frame_blobs.comp_pixel(image)

    print('Processing first layer comps...')

    ga_dert_ = comp_a(dert_, fga=0)  # if +G
    gr_dert_ = comp_r(dert_, fig=0, root_fcr=0)  # if -G

    print('Processing second layer comps...')
    # comp_a ->
    gaga_dert_ = comp_a(ga_dert_, fga=1)  # if +Ga
    gg_dert_ = comp_g(ga_dert_)  # if -Ga

    # comp_r ->
    gagr_dert_ = comp_a(gr_dert_, fga=0)  # if +Gr
    grr_dert_ = comp_r(gr_dert_, fig=0, root_fcr=1)  # if -Gr


    print('Processing third layer comps...')
    # comp_aga ->
    ga_gaga_dert_ = comp_a(gaga_dert_, fga=1)  # if +Gaga
    g_ga_dert_ = comp_g(gaga_dert_)  # if -Gaga

    # comp_g ->
    ga_gg_dert_ = comp_a(gg_dert_, fga=0)  # if +Gg
    g_rg_dert_ = comp_r(gg_dert_, fig=1, root_fcr=0)  # if -Gg

    # comp_agr ->
    ga_gagr_dert_ = comp_a(gagr_dert_, fga=1)  # if +Gagr
    g_gr_dert_ = comp_g(gagr_dert_)  # if -Gagr

    # comp_rr ->
    ga_grr_dert_ = comp_a(grr_dert_, fga=0)  # if +Grr
    g_rrr_dert_ = comp_r(grr_dert_, fig=0, root_fcr=1)  # if -Grr：

    print('Drawing forks...')
    ini_ = np.zeros((image.shape[0], image.shape[1]))  # initialize image y, x

    # 0th layer
    g_ = draw_m(ini_, dert_[4])

    # 1st layer
    ga_ = draw_ma(ini_.copy(), ga_dert_[7])
    gr_ = draw_mr(ini_.copy(), gr_dert_[1], rng=2)

    # 2nd layer
    gaga_ = draw_ma(ini_.copy(), gaga_dert_[7])
    gg_ = draw_m(ini_.copy(), gg_dert_[9])
    gagr_ = draw_mar(ini_.copy(), gagr_dert_[7], rng=2)
    grr_ = draw_mr(ini_.copy(), grr_dert_[4], rng=4)

    # 3rd layer
    ga_gaga_ = draw_ma(ini_.copy(), ga_gaga_dert_[7])
    g_ga_ = draw_m(ini_.copy(), g_ga_dert_[9])
    ga_gg_ = draw_ma(ini_.copy(), ga_gg_dert_[7])
    g_rg_ = draw_mr(ini_.copy(), g_rg_dert_[4], rng=2)
    ga_gagr_ = draw_mar(ini_.copy(), ga_gagr_dert_[7], rng=2)
    g_gr_ = draw_mr(ini_.copy(), g_gr_dert_[9], rng=2)
    ga_grr_ = draw_mar(ini_.copy(), ga_grr_dert_[7], rng=4)
    g_rrr_ = draw_mr(ini_.copy(), g_rrr_dert_[4], rng=8)

    # save to disk
    cv2.imwrite(OUTPUT_PATH + '0_g.png', g_)
    cv2.imwrite(OUTPUT_PATH + '1_ga.png', ga_)
    cv2.imwrite(OUTPUT_PATH + '2_gr.png', gr_)
    cv2.imwrite(OUTPUT_PATH + '3_gaga.png', gaga_)
    cv2.imwrite(OUTPUT_PATH + '4_gg.png', gg_)
    cv2.imwrite(OUTPUT_PATH + '5_gagr.png', gagr_)
    cv2.imwrite(OUTPUT_PATH + '6_grr.png', grr_)
    cv2.imwrite(OUTPUT_PATH + '7_ga_gaga.png', ga_gaga_)
    cv2.imwrite(OUTPUT_PATH + '8_g_ga.png', g_ga_)
    cv2.imwrite(OUTPUT_PATH + '9_ga_gg.png', ga_gg_)
    cv2.imwrite(OUTPUT_PATH + '10_g_rg.png', g_rg_)
    cv2.imwrite(OUTPUT_PATH + '11_ga_gagr.png', ga_gagr_)
    cv2.imwrite(OUTPUT_PATH + '12_g_gr.png', g_gr_)
    cv2.imwrite(OUTPUT_PATH + '13_ga_grr.png', ga_grr_)
    cv2.imwrite(OUTPUT_PATH + '14_g_rrr.png', g_rrr_)

    print('Terminating...')


def add_colour(img_comp, size_y, size_x):
    img_colour = np.zeros((3, size_y, size_x))
    img_colour[2] = img_comp
    img_colour[2][img_colour[2] < 255] = 0
    img_colour[2][img_colour[2] > 0] = 205
    img_colour[1] = img_comp
    img_colour[1][img_colour[1] == 255] = 0
    img_colour[1][img_colour[1] > 0] = 255
    img_colour = np.rollaxis(img_colour, 0, 3).astype('uint8')

    return img_colour

