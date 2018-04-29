import os
import time
import pickle
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from image_processing.vehicles_detect_model import custom_extract_features, custom_search_windows
from image_processing.sliding_window import slide_window, draw_boxes

# RGB 0.9673
# HSV 0.9868
# SV 0.9856
# HLS 0.98
# LS 0.9823
# HLS HL, YUV U0.9885


# --------------------- 前處理 特徵擷取 --------------------- #
def pre_processing():
    # car_features = extract_features(cars, color_space=color_space,
    #                                 spatial_size=spatial_size, hist_bins=hist_bins,
    #                                 orient=orient, pix_per_cell=pix_per_cell,
    #                                 cell_per_block=cell_per_block,
    #                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                                 hist_feat=hist_feat, hog_feat=hog_feat)
    # notcar_features = extract_features(notcars, color_space=color_space,
    #                                    spatial_size=spatial_size, hist_bins=hist_bins,
    #                                    orient=orient, pix_per_cell=pix_per_cell,
    #                                    cell_per_block=cell_per_block,
    #                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                                    hist_feat=hist_feat, hog_feat=hog_feat)
    car_features = custom_extract_features(cars, spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = custom_extract_features(notcars, spatial_size=spatial_size, hist_bins=hist_bins,
                                              orient=orient, pix_per_cell=pix_per_cell,
                                              cell_per_block=cell_per_block, spatial_feat=spatial_feat,
                                              hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # --------------------- 分成將數據分為 ---------------------- #
    # --------------- training and testing set --------------- #
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state, shuffle=True)

    # ---------------------- 前處理 正規化 ---------------------- #
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    return X_train, X_test, y_train, y_test, X_scaler


# ------------------------ SVM 訓練 ----------------------- #
def training(X_train, X_test, y_train, y_test):
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy)
    return svc, accuracy


# ------------------------ SVM 測試 ----------------------- #
def testing(image, svc, X_scaler, y_start_stop=[400, 656],
            spatial_size=(32, 32), hist_bins=32,
            orient=9, pix_per_cell=8, cell_per_block=2,
            spatial_feat=True, hist_feat=True, hog_feat=True, vis=True):
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(96, 96), xy_overlap=(0.5, 0.5)))

    windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(128, 128), xy_overlap=(0.7, 0.7)))

    windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                xy_window=(200, 200), xy_overlap=(0.75, 0.75)))

    start = time.time()
    hot_windows = custom_search_windows(image, windows, svc, X_scaler,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
    print('test process running time {:.3}s'.format(time.time() - start))
    if vis is True:
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        return window_img
    else:
        return hot_windows


if __name__ == "__main__":
    # PATH definition
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    MODEL_PATH = os.path.join(ROOT_PATH, 'model')
    VEHICLES = os.path.join(DATA_PATH, 'vehicles')
    NON_VEHICLES = os.path.join(DATA_PATH, 'non-vehicles')
    SVC_PICKLE_FILE = os.path.join(MODEL_PATH, 'svc_pickle.p')
    IMAGES_TEST = os.path.join(ROOT_PATH, 'test_images')
    IMAGES_OUTPUT = os.path.join(ROOT_PATH, 'output_images')

    # Read in cars and notcars
    cars = glob(VEHICLES + '/*/*')
    notcars = glob(NON_VEHICLES + '/*/*')

    # TODO: 測試更改這些參數
    # Color space and color channel
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hog_channel = [0, 1, 2]  # Can be list [0, 1, 2]

    # bin_spatial parameter
    spatial_size = (16, 16)  # Spatial binning dimensions
    spatial_feat = True  # Spatial features on or off

    # color_hist
    hist_bins = 16  # Number of histogram bins
    hist_feat = True  # Histogram features on or off

    # HOG parameter
    orient = 9  # HOG orientations             9
    pix_per_cell = 8  # HOG pixels per cell     8
    cell_per_block = 2  # HOG cells per block
    hog_feat = True  # HOG features on or off

    # sliding window parameter
    y_start_stop = [400, 656]  # Min and max in y to search in slide_window()

    MODE = 'training'
    MODE = 'testing'

    if MODE == 'training':
        x_train, x_test, y_train, y_test, x_scaler = pre_processing()
        svc, accuracy = training(x_train, x_test, y_train, y_test)
        if not os.path.exists(SVC_PICKLE_FILE):
            print('Pickle File {} is not exists, create one now.'.format(SVC_PICKLE_FILE))
            dist_pickle = {}
            dist_pickle["svc"] = svc
            dist_pickle["scaler"] = x_scaler
            dist_pickle["orient"] = orient
            dist_pickle["pix_per_cell"] = pix_per_cell
            dist_pickle["cell_per_block"] = cell_per_block
            dist_pickle["spatial_size"] = spatial_size
            dist_pickle["hist_bins"] = hist_bins
            dist_pickle["accuracy"] = accuracy
            pickle.dump(dist_pickle, open(SVC_PICKLE_FILE, "wb"))

        # testing
        img = cv2.imread('./test_images/test1.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_img1 = testing(img, svc, x_scaler, y_start_stop,
                           spatial_size, hist_bins,
                           orient, pix_per_cell, cell_per_block,
                           spatial_feat, hist_feat, hist_feat)
        plt.figure()
        plt.title('out_img1')
        plt.imshow(out_img1)

        ystart = 400
        ystop = 656
        scale = 1.5
        # out_img2 = find_cars(img, ystart, ystop, scale, svc, x_scaler, color_space, orient,
        #                     pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel)
        # plt.figure()
        # plt.title('out_img2')
        # plt.imshow(out_img2)
        plt.show()

        # Check is this model need to save
        dist_pickle = pickle.load(open(SVC_PICKLE_FILE, "rb"))
        pre_accuracy = dist_pickle["accuracy"]
        print("Accuracy {}, Pre-accuracy {}".format(accuracy, pre_accuracy))

        ans = input("是否要儲存這次的model(y/n): ")
        if ans == 'y':
            print('The parameters have more higher performance, Save {} File.'.format(SVC_PICKLE_FILE))
            dist_pickle = {}
            dist_pickle["svc"] = svc
            dist_pickle["scaler"] = x_scaler
            dist_pickle["orient"] = orient
            dist_pickle["pix_per_cell"] = pix_per_cell
            dist_pickle["cell_per_block"] = cell_per_block
            dist_pickle["spatial_size"] = spatial_size
            dist_pickle["hist_bins"] = hist_bins
            dist_pickle["accuracy"] = accuracy
            pickle.dump(dist_pickle, open(SVC_PICKLE_FILE, "wb"))

    else:
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
        accuracy = dist_pickle["accuracy"]
        images_path = glob(IMAGES_TEST + '/*')
        print("Accuracy {}".format(accuracy))
        # testing
        print('Start testing images')
        plt.figure(figsize=(12, 10))
        for index, path in enumerate(images_path, 1):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_img = testing(img, svc, x_scaler, y_start_stop,
                              spatial_size, hist_bins,
                              orient, pix_per_cell, cell_per_block,
                              spatial_feat, hist_feat, hog_feat)

            # Image Show
            plt.subplot(3, 2, index)
            plt.title(os.path.split(path)[-1])
            plt.imshow(out_img)

            # Image save
            file_name = 'output_svm_{}.png'.format(os.path.split(path)[-1].split('.')[0])
            file_path = os.path.join(IMAGES_OUTPUT, file_name)
            plt.imsave(file_path, out_img)

        file_name = 'output_svm_test_all.png'
        file_path = os.path.join(IMAGES_OUTPUT, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
