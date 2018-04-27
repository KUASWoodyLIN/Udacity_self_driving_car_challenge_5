import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Udacity_self_driving_car_challenge_4.image_processing.calibration import camera_cal, found_chessboard, read_camera_cal_file


if __name__ == '__main__':
    # Load cameraMatrix and distCoeffs parameter
    if not os.path.exists('wide_dist_pickle.p'):
        objpoints, imgpoints = found_chessboard()
        mtx, dist = camera_cal(objpoints, imgpoints)
    else:
        print('Get parameter from pickle file')
        mtx, dist = read_camera_cal_file('wide_dist_pickle.p')

    test_image = '../test_images/straight_lines1.jpg'
    # STEP 1. Try on straight lines image
    img = plt.imread(test_image)
    img = cv2.undistort(img, mtx, dist, None, None)

    offset = img.shape[1] / 2
    # Longer one
    src = np.float32([(596, 447), (683, 447), (1120, 720), (193, 720)])

    # src = np.float32([(578, 460), (704, 460), (1120, 720), (193, 720)])
    dst = np.float32([(offset-300, 0), (offset+300, 0), (offset+300, 720), (offset-300, 720)])

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    perspective_M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    img_out = cv2.warpPerspective(img, perspective_M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Show the straight lines image
    plt.figure('figure1', figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.plot([578, 704, 1122, 193, 578], [460, 460, 720, 720, 460])
    plt.subplot(1, 2, 2)
    plt.imshow(img_out)
    plt.plot([offset-300, offset+300, offset+300, offset-300, offset-300], [0, 0, 720, 720, 0])
    plt.savefig('../output_images/get_wrap_parameter2.png', bbox_inches='tight')

    # STEP 2. Test on Turning image
    test_image = '../test_images/test5.jpg'
    img = plt.imread(test_image)
    img_out = cv2.warpPerspective(img, perspective_M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_out)
    plt.savefig('../output_images/wrap_parameter_test2.png', bbox_inches='tight')


    plt.show()
