import os
import pickle
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

from Udacity_self_driving_car_challenge_5.svm_training import testing
from Udacity_self_driving_car_challenge_5.image_processing.sliding_window import draw_boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def apply_mask(heatmap, bboxes):
    height, width = heatmap.shape
    mask = np.zeros_like(heatmap)

    for bbox in bboxes:
        x = int((bbox[1][0] + bbox[0][0]) / 2)
        y = int((bbox[1][1] + bbox[0][1]) / 2)
        w_half = int((bbox[1][0] - bbox[0][0]) /2)
        h_half = int((bbox[1][1] - bbox[0][1]) /2)
        w_add = 50
        h_add = 50

        vetices = np.array([[[np.max([x-w_half-w_add, 0]), np.max([y-h_half-h_add, 0])],
                             [np.min([x+w_half+w_add, width]), np.max([y-h_half-h_add, 0])],
                             [np.min([x + w_half + w_add, width]), np.min([y + h_half + h_add, height])],
                             [np.max([x-w_half-w_add, 0]), np.min([y+h_half+h_add, height])]]])
        cv2.fillPoly(mask, vetices, 255)
    return mask


def combine_mask_threshold(heatmap, mask, threshold):
    masked_heatmap = cv2.bitwise_and(heatmap, mask)
    # Zero out pixels below the threshold
    masked_heatmap[masked_heatmap <= threshold] = 0
    # Return thresholded map
    return masked_heatmap


def label_bboxes(labels):
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bboxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
    return bboxes


def draw_labeled_bboxes(img, labels, color):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    return img


def detection_v2(img, hot_windows, color=(0, 0, 255), vis=False, save=False):
    # 1) heat map
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)
    # 2) threshold 2
    heatmap_threshold = np.copy(heatmap)
    heatmap_threshold = apply_threshold(heatmap_threshold, threshold=3)
    # 3) scipy label
    labels = label(heatmap_threshold)
    # 4) label bboxes
    bboxes = label_bboxes(labels)
    # 5) mask
    mask = apply_mask(heatmap, bboxes)
    # 5) threshold 1
    heatmap = combine_mask_threshold(heatmap, mask=mask, threshold=1)
    # 6) scipy label
    labels = label(heatmap)
    # 7) draw labeles bboxes
    img_out = np.copy(img)
    img_out = draw_labeled_bboxes(img_out, labels, color)

    if save:
        # Image Show
        plt.figure(figsize=(8, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(heatmap, cmap='hot')

        plt.subplot(3, 1, 2)
        plt.imshow(labels[0], cmap='gray')

        plt.subplot(3, 1, 3)
        plt.imshow(img_out)
        file_name = 'Vehicles_detection_v2.png'
        file_path = os.path.join(IMAGES_OUTPUT, file_name)
        plt.savefig(file_path, bbox_inches='tight')
    if vis:
        window_img = draw_boxes(img, hot_windows, color=color, thick=6)
        return img_out, window_img, heatmap
    else:
        return img_out


def detection_v1(img, hot_windows, color=(0, 0, 255), save=False):
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 2)

    labels = label(heatmap)
    print(labels[1], 'cars found')

    img_out = draw_labeled_bboxes(img, labels, color)

    if save:
        # Image Show
        plt.figure(figsize=(8, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(heatmap, cmap='hot')

        plt.subplot(3, 1, 2)
        plt.imshow(labels[0], cmap='gray')

        plt.subplot(3, 1, 3)
        plt.imshow(img_out)
        file_name = 'Vehicles_detection_v1.png'
        file_path = os.path.join(IMAGES_OUTPUT, file_name)
        plt.savefig(file_path, bbox_inches='tight')

    return img_out


def test_image():
    # load a pe-trained svc model from a serialized (pickle) file
    dist_pickle = pickle.load(open(SVC_PICKLE_FILE, "rb"))

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    x_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    image_path = os.path.join(IMAGES_TEST, 'test1.jpg')
    # testing

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hot_windows = testing(img, svc, x_scaler, spatial_size=spatial_size, hist_bins=hist_bins,
                          orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=False)
    # window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)

    img1 = np.copy(img)
    detection_v1(img1, hot_windows, save=True)
    img2 = np.copy(img)
    detection_v2(img2, hot_windows, save=True)

    plt.show()


def test_images():
    # load a pe-trained svc model from a serialized (pickle) file
    dist_pickle = pickle.load(open(SVC_PICKLE_FILE, "rb"))

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    x_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    images_path = glob(IMAGES_TEST + '/*')

    # testing
    print('Start testing images')
    plt.figure(figsize=(12, 20))
    for index, path in enumerate(images_path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hot_windows = testing(img, svc, x_scaler, spatial_size=spatial_size, hist_bins=hist_bins,
                              orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=False)
        img1 = np.copy(img)
        img_out1 = detection_v1(img1, hot_windows)
        img2 = np.copy(img)
        img_out2 = detection_v2(img2, hot_windows)

        # window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)

        # Image Show

        plt.subplot(6, 2, index * 2 + 1)
        plt.title(os.path.split(path)[-1])
        plt.imshow(img_out1)
        plt.subplot(6, 2, index * 2 + 2)
        plt.imshow(img_out2)

        # Image save
        # file_name = 'output_{}.png'.format(os.path.split(path)[-1].split('.')[0])
        # file_path = os.path.join(IMAGES_OUTPUT, file_name)
        # plt.imsave(file_path, out_img)

    file_name = 'Vehicles_detectionv2.png'
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # PATH definition
    ROOT_PATH = os.path.split(os.getcwd())[0]
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    SVC_PICKLE_FILE = os.path.join(DATA_PATH, 'svc_pickle.p')
    IMAGES_TEST = os.path.join(ROOT_PATH, 'test_images')
    IMAGES_OUTPUT = os.path.join(ROOT_PATH, 'output_images')

    # test one Image
    # test_image()

    # test all the images in test file
    test_images()
