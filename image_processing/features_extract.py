import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm='L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def color_space_tweak(car_img, nocar_img, index):
    # convert color space
    car_img_rgb = np.copy(car_img)
    car_img_hls = cv2.cvtColor(car_img, cv2.COLOR_RGB2HLS)
    car_img_luv = cv2.cvtColor(car_img, cv2.COLOR_RGB2LUV)
    car_img_yuv = cv2.cvtColor(car_img, cv2.COLOR_RGB2YUV)
    nocar_img_rgb = np.copy(nocar_img)
    nocar_img_hls = cv2.cvtColor(nocar_img, cv2.COLOR_RGB2HLS)
    nocar_img_luv = cv2.cvtColor(nocar_img, cv2.COLOR_RGB2LUV)
    nocar_img_yuv = cv2.cvtColor(nocar_img, cv2.COLOR_RGB2YUV)

    # HOG parameter
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    one = 'orient={}, pix_cell={}, cell_block={}'.format(orient, pix_per_cell, cell_per_block)
    _, hog_image_rgb1 = get_hog_features(car_img_rgb[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_rgb2 = get_hog_features(car_img_rgb[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_rgb3 = get_hog_features(car_img_rgb[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_rgb4 = get_hog_features(nocar_img_rgb[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_rgb5 = get_hog_features(nocar_img_rgb[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_rgb6 = get_hog_features(nocar_img_rgb[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)

    _, hog_image_hls1 = get_hog_features(car_img_hls[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_hls2 = get_hog_features(car_img_hls[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_hls3 = get_hog_features(car_img_hls[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_hls4 = get_hog_features(nocar_img_hls[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_hls5 = get_hog_features(nocar_img_hls[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_hls6 = get_hog_features(nocar_img_hls[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)

    _, hog_image_luv1 = get_hog_features(car_img_luv[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_luv2 = get_hog_features(car_img_luv[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_luv3 = get_hog_features(car_img_luv[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_luv4 = get_hog_features(nocar_img_luv[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_luv5 = get_hog_features(nocar_img_luv[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_luv6 = get_hog_features(nocar_img_luv[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)

    _, hog_image_yuv1 = get_hog_features(car_img_yuv[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_yuv2 = get_hog_features(car_img_yuv[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_yuv3 = get_hog_features(car_img_yuv[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_yuv4 = get_hog_features(nocar_img_yuv[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_yuv5 = get_hog_features(nocar_img_yuv[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    _, hog_image_yuv6 = get_hog_features(nocar_img_yuv[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block, vis=True, feature_vec=True)
    plt.figure(figsize=(24, 32))
    # RGB Channel
    plt.subplot(8, 6, 1)
    plt.title('RGB Channel 1')
    plt.imshow(car_img_rgb[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 2)
    plt.title('RGB Channel 2')
    plt.imshow(car_img_rgb[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 3)
    plt.title('RGB Channel 3')
    plt.imshow(car_img_rgb[:, :, 2], cmap='gray')
    plt.subplot(8, 6, 4)
    plt.title('RGB Channel 1')
    plt.imshow(nocar_img_rgb[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 5)
    plt.title('RGB Channel 2')
    plt.imshow(nocar_img_rgb[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 6)
    plt.title('RGB Channel 3')
    plt.imshow(nocar_img_rgb[:, :, 2], cmap='gray')

    plt.subplot(8, 6, 7)
    plt.title(one)
    plt.imshow(hog_image_rgb1, cmap='gray')
    plt.subplot(8, 6, 8)
    plt.title(one)
    plt.imshow(hog_image_rgb2, cmap='gray')
    plt.subplot(8, 6, 9)
    plt.title(one)
    plt.imshow(hog_image_rgb3, cmap='gray')
    plt.subplot(8, 6, 10)
    plt.title(one)
    plt.imshow(hog_image_rgb4, cmap='gray')
    plt.subplot(8, 6, 11)
    plt.title(one)
    plt.imshow(hog_image_rgb5, cmap='gray')
    plt.subplot(8, 6, 12)
    plt.title(one)
    plt.imshow(hog_image_rgb6, cmap='gray')

    # HLS Channel
    plt.subplot(8, 6, 13)
    plt.title('HLS Channel 1')
    plt.imshow(car_img_hls[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 14)
    plt.title('HLS Channel 2')
    plt.imshow(car_img_hls[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 15)
    plt.title('HLS Channel 3')
    plt.imshow(car_img_hls[:, :, 2], cmap='gray')
    plt.subplot(8, 6, 16)
    plt.title('HLS Channel 1')
    plt.imshow(nocar_img_hls[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 17)
    plt.title('HLS Channel 2')
    plt.imshow(nocar_img_hls[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 18)
    plt.title('HLS Channel 3')
    plt.imshow(nocar_img_hls[:, :, 2], cmap='gray')

    plt.subplot(8, 6, 19)
    plt.title(one)
    plt.imshow(hog_image_hls1, cmap='gray')
    plt.subplot(8, 6, 20)
    plt.title(one)
    plt.imshow(hog_image_hls2, cmap='gray')
    plt.subplot(8, 6, 21)
    plt.title(one)
    plt.imshow(hog_image_hls3, cmap='gray')
    plt.subplot(8, 6, 22)
    plt.title(one)
    plt.imshow(hog_image_hls4, cmap='gray')
    plt.subplot(8, 6, 23)
    plt.title(one)
    plt.imshow(hog_image_hls5, cmap='gray')
    plt.subplot(8, 6, 24)
    plt.title(one)
    plt.imshow(hog_image_hls6, cmap='gray')

    # LUV Channel
    plt.subplot(8, 6, 25)
    plt.title('LUV Channel 1')
    plt.imshow(car_img_luv[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 26)
    plt.title('LUV Channel 2')
    plt.imshow(car_img_luv[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 27)
    plt.title('LUV Channel 3')
    plt.imshow(car_img_luv[:, :, 2], cmap='gray')
    plt.subplot(8, 6, 28)
    plt.title('LUV Channel 1')
    plt.imshow(nocar_img_luv[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 29)
    plt.title('LUV Channel 2')
    plt.imshow(nocar_img_luv[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 30)
    plt.title('LUV Channel 3')
    plt.imshow(nocar_img_luv[:, :, 2], cmap='gray')

    plt.subplot(8, 6, 31)
    plt.title(one)
    plt.imshow(hog_image_luv1, cmap='gray')
    plt.subplot(8, 6, 32)
    plt.title(one)
    plt.imshow(hog_image_luv2, cmap='gray')
    plt.subplot(8, 6, 33)
    plt.title(one)
    plt.imshow(hog_image_luv3, cmap='gray')
    plt.subplot(8, 6, 34)
    plt.title(one)
    plt.imshow(hog_image_luv4, cmap='gray')
    plt.subplot(8, 6, 35)
    plt.title(one)
    plt.imshow(hog_image_luv5, cmap='gray')
    plt.subplot(8, 6, 36)
    plt.title(one)
    plt.imshow(hog_image_luv6, cmap='gray')

    # YUV Channel
    plt.subplot(8, 6, 37)
    plt.title('YUV Channel 1')
    plt.imshow(car_img_yuv[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 38)
    plt.title('YUV Channel 2')
    plt.imshow(car_img_yuv[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 39)
    plt.title('YUV Channel 3')
    plt.imshow(car_img_yuv[:, :, 2], cmap='gray')
    plt.subplot(8, 6, 40)
    plt.title('YUV Channel 1')
    plt.imshow(nocar_img_yuv[:, :, 0], cmap='gray')
    plt.subplot(8, 6, 41)
    plt.title('YUV Channel 2')
    plt.imshow(nocar_img_yuv[:, :, 1], cmap='gray')
    plt.subplot(8, 6, 42)
    plt.title('YUV Channel 3')
    plt.imshow(nocar_img_yuv[:, :, 2], cmap='gray')

    plt.subplot(8, 6, 43)
    plt.title(one)
    plt.imshow(hog_image_yuv1, cmap='gray')
    plt.subplot(8, 6, 44)
    plt.title(one)
    plt.imshow(hog_image_yuv2, cmap='gray')
    plt.subplot(8, 6, 45)
    plt.title(one)
    plt.imshow(hog_image_yuv3, cmap='gray')
    plt.subplot(8, 6, 46)
    plt.title(one)
    plt.imshow(hog_image_yuv4, cmap='gray')
    plt.subplot(8, 6, 47)
    plt.title(one)
    plt.imshow(hog_image_yuv5, cmap='gray')
    plt.subplot(8, 6, 48)
    plt.title(one)
    plt.imshow(hog_image_yuv6, cmap='gray')

    file_name = 'color_space_compare_{}.png'.format(index)
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')

    # Save HLS h Channel
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('CAR HLS H Channel')
    plt.imshow(car_img_hls[:, :, 0], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('NOCar HLS H Channel')
    plt.imshow(nocar_img_hls[:, :, 0], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(hog_image_hls1, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(hog_image_hls3, cmap='gray')
    file_name = 'HLS_H_Channel_{}.png'.format(index)
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')

    # Save HLS L Channle
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('CAR HLS L Channel')
    plt.imshow(car_img_hls[:, :, 1], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('NOCar HLS L Channel')
    plt.imshow(nocar_img_hls[:, :, 1], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(hog_image_hls2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(hog_image_hls5, cmap='gray')
    file_name = 'HLS_L_Channel_{}.png'.format(index)
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')

    # Save LUV U channel
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('CAR LUV U Channel')
    plt.imshow(car_img_luv[:, :, 1], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('NOCar LUV U Channel')
    plt.imshow(nocar_img_luv[:, :, 1], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(hog_image_luv2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(hog_image_luv5, cmap='gray')
    file_name = 'LUV_U_Channel_{}.png'.format(index)
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')


def hog_parameter_tweak(car_img, nocar_img):
    # HOG parameter
    orient = [7, 9, 11]          # HOG orientations
    pix_per_cell = [8, 8, 8]   # HOG pixels per cell
    cell_per_block = 2          # HOG cells per block

    one = 'orient={}, pix_cell={}, cell_block={}'.format(orient[0], pix_per_cell[0], cell_per_block)
    hog_features1, hog_image1 = get_hog_features(car_img[:, :, 0], orient=orient[0], pix_per_cell=pix_per_cell[0],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features2, hog_image2 = get_hog_features(car_img[:, :, 1], orient=orient[0], pix_per_cell=pix_per_cell[0],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features3, hog_image3 = get_hog_features(car_img[:, :, 2], orient=orient[0], pix_per_cell=pix_per_cell[0],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features01, hog_image01 = get_hog_features(nocar_img[:, :, 0], orient=orient[0], pix_per_cell=pix_per_cell[0],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features02, hog_image02 = get_hog_features(nocar_img[:, :, 1], orient=orient[0], pix_per_cell=pix_per_cell[0],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features03, hog_image03 = get_hog_features(nocar_img[:, :, 2], orient=orient[0], pix_per_cell=pix_per_cell[0],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)

    two = 'orient={}, pix_cell={}, cell_block={}'.format(orient[1], pix_per_cell[1], cell_per_block)
    hog_features4, hog_image4 = get_hog_features(car_img[:, :, 0], orient=orient[1], pix_per_cell=pix_per_cell[1],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features5, hog_image5 = get_hog_features(car_img[:, :, 1], orient=orient[1], pix_per_cell=pix_per_cell[1],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features6, hog_image6 = get_hog_features(car_img[:, :, 2], orient=orient[1], pix_per_cell=pix_per_cell[1],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features04, hog_image04 = get_hog_features(nocar_img[:, :, 0], orient=orient[1], pix_per_cell=pix_per_cell[1],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features05, hog_image05 = get_hog_features(nocar_img[:, :, 1], orient=orient[1], pix_per_cell=pix_per_cell[1],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features06, hog_image06 = get_hog_features(nocar_img[:, :, 2], orient=orient[1], pix_per_cell=pix_per_cell[1],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)

    three = 'orient={}, pix_cell={}, cell_block={}'.format(orient[2], pix_per_cell[2], cell_per_block)
    hog_features7, hog_image7 = get_hog_features(car_img[:, :, 0], orient=orient[2], pix_per_cell=pix_per_cell[2],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features8, hog_image8 = get_hog_features(car_img[:, :, 1], orient=orient[2], pix_per_cell=pix_per_cell[2],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features9, hog_image9 = get_hog_features(car_img[:, :, 2], orient=orient[2], pix_per_cell=pix_per_cell[2],
                                                 cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features07, hog_image07 = get_hog_features(nocar_img[:, :, 0], orient=orient[2], pix_per_cell=pix_per_cell[2],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features08, hog_image08 = get_hog_features(nocar_img[:, :, 1], orient=orient[2], pix_per_cell=pix_per_cell[2],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)
    hog_features09, hog_image09 = get_hog_features(nocar_img[:, :, 2], orient=orient[2], pix_per_cell=pix_per_cell[2],
                                                   cell_per_block=cell_per_block, vis=True, feature_vec=True)

    # Image Show
    plt.figure(figsize=(24, 16))
    plt.subplot(4, 6, 1)
    plt.title('HLS H Channel')
    plt.imshow(car_img[:, :, 0], cmap='gray')
    plt.subplot(4, 6, 2)
    plt.title('HLS L Channel')
    plt.imshow(car_img[:, :, 1], cmap='gray')
    plt.subplot(4, 6, 3)
    plt.title('YUV UChannel')
    plt.imshow(car_img[:, :, 2], cmap='gray')
    plt.subplot(4, 6, 4)
    plt.title('HLS H Channel')
    plt.imshow(nocar_img[:, :, 0], cmap='gray')
    plt.subplot(4, 6, 5)
    plt.title('HLS L Channel')
    plt.imshow(nocar_img[:, :, 1], cmap='gray')
    plt.subplot(4, 6, 6)
    plt.title('YUV UChannel')
    plt.imshow(nocar_img[:, :, 2], cmap='gray')

    plt.subplot(4, 6, 7)
    plt.title(one)
    plt.imshow(hog_image1, cmap='gray')
    plt.subplot(4, 6, 8)
    plt.title(one)
    plt.imshow(hog_image2, cmap='gray')
    plt.subplot(4, 6, 9)
    plt.title(one)
    plt.imshow(hog_image3, cmap='gray')
    plt.subplot(4, 6, 10)
    plt.title(one)
    plt.imshow(hog_image01, cmap='gray')
    plt.subplot(4, 6, 11)
    plt.title(one)
    plt.imshow(hog_image02, cmap='gray')
    plt.subplot(4, 6, 12)
    plt.title(one)
    plt.imshow(hog_image03, cmap='gray')

    plt.subplot(4, 6, 13)
    plt.title(two)
    plt.imshow(hog_image4, cmap='gray')
    plt.subplot(4, 6, 14)
    plt.title(two)
    plt.imshow(hog_image5, cmap='gray')
    plt.subplot(4, 6, 15)
    plt.title(two)
    plt.imshow(hog_image6, cmap='gray')
    plt.subplot(4, 6, 16)
    plt.title(two)
    plt.imshow(hog_image04, cmap='gray')
    plt.subplot(4, 6, 17)
    plt.title(two)
    plt.imshow(hog_image05, cmap='gray')
    plt.subplot(4, 6, 18)
    plt.title(two)
    plt.imshow(hog_image06, cmap='gray')

    plt.subplot(4, 6, 19)
    plt.title(three)
    plt.imshow(hog_image7, cmap='gray')
    plt.subplot(4, 6, 20)
    plt.title(three)
    plt.imshow(hog_image8, cmap='gray')
    plt.subplot(4, 6, 21)
    plt.title(three)
    plt.imshow(hog_image9, cmap='gray')
    plt.subplot(4, 6, 22)
    plt.title(three)
    plt.imshow(hog_image07, cmap='gray')
    plt.subplot(4, 6, 23)
    plt.title(three)
    plt.imshow(hog_image08, cmap='gray')
    plt.subplot(4, 6, 24)
    plt.title(three)
    plt.imshow(hog_image09, cmap='gray')

    # Image Save
    file_name = 'car_orient_{}'.format(index)  # orient, pix_per_cell
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')


if __name__ == "__main__":
    # PATH definition
    FUNCTIONS_PATH = os.getcwd()
    ROOT_PATH = os.path.split(FUNCTIONS_PATH)[0]
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    VEHICLES = os.path.join(DATA_PATH, 'vehicles')
    NON_VEHICLES = os.path.join(DATA_PATH, 'non-vehicles')
    IMAGES_OUTPUT = os.path.join(ROOT_PATH, 'output_images')

    # Read in cars and notcars
    cars = glob(VEHICLES + '/*/*')
    notcars = glob(NON_VEHICLES + '/*/*')

    # TODO: change the index to choose the test image
    index = 0
    test_car_file = cars[index]
    test_nocar_file = notcars[index]

    car_img = cv2.imread(test_car_file)
    nocar_img = cv2.imread(test_nocar_file)
    car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
    nocar_img = cv2.cvtColor(nocar_img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(car_img)
    plt.subplot(1, 2, 2)
    plt.imshow(nocar_img)
    file_name = 'test_image_{}.png'.format(index)
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')

    # mode = "test_color_space" if input("Choose which mode do you want to test?\n 1) test color space mode input 1\n "
    #                                    "2) test hog parameter mode input 2") == '1' else "test_hog_parameter"
    mode = 'test_hog_parameter'
    if mode == 'test_color_space':
        color_space_tweak(car_img, nocar_img, index)
    elif mode == 'test_hog_parameter':
        car_img_hls = cv2.cvtColor(car_img, cv2.COLOR_RGB2HLS)
        car_img_yuv = cv2.cvtColor(car_img, cv2.COLOR_RGB2YUV)
        car_img[:, :, :2] = car_img_hls[:, :, :2]
        car_img[:, :, 2] = car_img_yuv[:, :, 1]
        nocar_img_hls = cv2.cvtColor(nocar_img, cv2.COLOR_RGB2HLS)
        nocar_img_yuv = cv2.cvtColor(nocar_img, cv2.COLOR_RGB2YUV)
        nocar_img[:, :, :2] = nocar_img_hls[:, :, :2]
        nocar_img[:, :, 2] = nocar_img_yuv[:, :, 1]

        hog_parameter_tweak(car_img, nocar_img)

    plt.show()
    print()


    # TODO: Tweak these parameters and see how the results change.
    # bin_spatial parameter
    spatial_size = (16, 16)  # Spatial binning dimensions
    spatial_feat = True  # Spatial features on or off
    # color_hist
    hist_bins = 16  # Number of histogram bins
    hist_feat = True  # Histogram features on or off
    # sliding window parameter
    y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
