import os
import random
import pickle
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt


def found_chessboard():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) (x,y,z)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)   # x,y,coordinates

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for img_path in images_path:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('image', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints


def camera_cal(objpoints, imgpoints):
    # random chose image to test
    random_chose = random.randint(0, len(images_path)-1)
    img_test = images_path[random_chose]

    # Test undistortion on an image
    img = cv2.imread(img_test)
    img_size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('Undistorted Image')
    plt.imshow(dst)
    plt.show()
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    if not os.path.exists(WIDE_DIST_FILE):
        print('Pickle File {} is not exists, create one now.'.format(WIDE_DIST_FILE))
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(WIDE_DIST_FILE, "wb"))

    return mtx, dist


def read_camera_cal_file(file):
    with open(file, 'rb') as f:
        dump = pickle.load(f)
    return dump['mtx'], dump['dist']


def test():
    if not os.path.exists(WIDE_DIST_FILE):
        objpoints, imgpoints = found_chessboard()
        mtx, dist = camera_cal(objpoints, imgpoints)
    else:
        print('Get parameter from pickle file')
        mtx, dist = read_camera_cal_file(WIDE_DIST_FILE)


def undistort_test_images():
    if not os.path.exists(WIDE_DIST_FILE):
        objpoints, imgpoints = found_chessboard()
        mtx, dist = camera_cal(objpoints, imgpoints)
    else:
        print('Get parameter from pickle file')
        mtx, dist = read_camera_cal_file(WIDE_DIST_FILE)
    vstack = []
    for path in images_path:
        img = cv2.imread(path)
        img_out = cv2.undistort(img, mtx, dist, None, None)
        img = cv2.resize(img, (320, 180))
        img_out = cv2.resize(img_out, (320, 180))
        vstack.append(np.hstack((img, img_out)))
    vstack = np.vstack(vstack)
    cv2.imwrite('../output_images/undistort_compare.png', vstack)


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    IMAGE_PATH = os.path.join(ROOT_PATH, 'camera_cal/')
    WIDE_DIST_FILE = os.path.join(ROOT_PATH, 'wide_dist_pickle.p')
    images_path = glob(IMAGE_PATH + '*.jpg')

    test()
    # undistort_test_images()
