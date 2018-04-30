import os
import random
import pickle
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt

from image_processing.calibration import camera_cal, found_chessboard, read_camera_cal_file
from image_processing.edge_detection import combing_color_thresh
from image_processing.find_lines import histogram_search, histogram_search2
from image_processing.line_fit_fix import Line
from image_processing.multiple_detections import detection_v2



ROOT_PATH = os.getcwd()
IMAGE_TEST_DIR = os.path.join(ROOT_PATH, 'test_images')
IMAGE_OUTPUT_DIR = os.path.join(ROOT_PATH, 'output_images')
VIDEO_OUTPUT_DIR = os.path.join(ROOT_PATH, 'output_video')
IMAGE_PROCESSING_PATH = os.path.join(ROOT_PATH, 'image_processing')
WIDE_DIST_FILE = os.path.join(IMAGE_PROCESSING_PATH, 'wide_dist_pickle.p')
IMAGES_PATH = glob(IMAGE_TEST_DIR + '/*.jpg')
MODEL_PATH = os.path.join(ROOT_PATH, 'model')
SVC_PICKLE_FILE = os.path.join(MODEL_PATH, 'svc_pickle.p')


# Load cameraMatrix and distCoeffs parameter
if not os.path.exists(WIDE_DIST_FILE):
    objpoints, imgpoints = found_chessboard()
    mtx, dist = camera_cal(objpoints, imgpoints)
else:
    print('Get parameter from pickle file')
    mtx, dist = read_camera_cal_file(WIDE_DIST_FILE)

# Get Perspective Transform Parameter
offset = 1280 / 2
src = np.float32([(596, 447), (683, 447), (1120, 720), (193, 720)])     # Longer line
# src = np.float32([(578, 460), (704, 460), (1120, 720), (193, 720)])     # shorter line
dst = np.float32([(offset-300, 0), (offset+300, 0), (offset+300, 720), (offset-300, 720)])
perspective_M = cv2.getPerspectiveTransform(src, dst)
inver_perspective_M = cv2.getPerspectiveTransform(dst, src)

left_line = Line()
right_line = Line()
count_h1 = 0
count_h2 = 0


def process_image_svm(image, show_birdview=False):
    global count_h1, count_h2
    # Apply a distortion correction to raw images.
    image = cv2.undistort(image, mtx, dist, None, None)

    # Use color transforms, gradients to find the object edge and change into binary image
    image_binary = combing_color_thresh(image)

    # Transform image to bird view
    image_bird_view = cv2.warpPerspective(image_binary, perspective_M, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # find the road lines, curvature and distance between car_center and road_center
    color_warp, curv, center, left_or_right, left_line.new_fit, right_line.new_fit, left_line.allx, right_line.allx = histogram_search(image_bird_view)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warp_back = cv2.warpPerspective(color_warp, inver_perspective_M, (image.shape[1], image.shape[0]))

    # svm vehicles detections
    img_detect = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hot_windows = testing_svm(img_detect, svc, x_scaler, spatial_size=spatial_size, hist_bins=hist_bins,
                              orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=False)

    # multiple bboxes detections
    image= detection_v2(image, hot_windows, color=(255, 0, 0))

    # Combine the result with the original image
    img_out = cv2.addWeighted(image, 1, warp_back, 0.3, 0)

    # Add description on images
    text1 = "Radius of Curature = {:.2f}(m)".format(curv)
    text2 = "Vehicle is {:.3f}m {} of center".format(abs(center), left_or_right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, text1, (50, 50), font, 1.5, color=(255, 255, 255), thickness=3)
    cv2.putText(img_out, text2, (50, 100), font, 1.5, color=(255, 255, 255), thickness=3)

    if show_birdview:
        show_image_bird_view = cv2.resize(image_bird_view, (360, 360))
        show_image_bird_view = cv2.cvtColor(show_image_bird_view, cv2.COLOR_GRAY2RGB)
        show_color_warp = cv2.resize(color_warp, (360, 360))
        show_color_warp = cv2.addWeighted(show_image_bird_view, 1, show_color_warp, 0.5, 0)
        return img_out, show_image_bird_view, show_color_warp

    return img_out


def process_video_svm(image, show_birdview=False, show_heatmaps=False):
    global count_h1, count_h2
    # Apply a distortion correction to raw images.
    image = cv2.undistort(image, mtx, dist, None, None)

    # Use color transforms, gradients to find the object edge and change into binary image
    image_binary = combing_color_thresh(image)

    # Transform image to bird view
    image_bird_view = cv2.warpPerspective(image_binary, perspective_M, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # find the road lines, curvature and distance between car_center and road_center
    if not left_line.detected or not right_line.detected:
        color_warp, curv, center, left_or_right, left_line.new_fit, right_line.new_fit, \
        left_line.allx, right_line.allx = histogram_search(image_bird_view)
        count_h1 += 1
    else:
        color_warp, curv, center, left_or_right, left_line.new_fit, right_line.new_fit, \
        left_line.allx, right_line.allx = histogram_search2(image_bird_view, left_line.best_fit, right_line.best_fit)
        count_h2 += 1

    # Check the lines health
    left_line.fit_fix()
    right_line.fit_fix()

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warp_back = cv2.warpPerspective(color_warp, inver_perspective_M, (image.shape[1], image.shape[0]))

    # svm vehicles detections
    img_detect = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hot_windows = testing_svm(img_detect, svc, x_scaler, spatial_size=spatial_size, hist_bins=hist_bins,
                              orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=False)

    # multiple bboxes detections
    image, window_img, heatmap = detection_v2(image, hot_windows, color=(255, 0, 0), vis=True)

    # Combine the result with the original image
    img_out = cv2.addWeighted(image, 1, warp_back, 0.3, 0)

    # Add description on images
    text1 = "Radius of Curature = {:.2f}(m)".format(curv)
    text2 = "Vehicle is {:.3f}m {} of center".format(abs(center), left_or_right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, text1, (50, 50), font, 1.5, color=(255, 255, 255), thickness=3)
    cv2.putText(img_out, text2, (50, 100), font, 1.5, color=(255, 255, 255), thickness=3)

    if show_birdview and show_heatmaps:
        show_image_bird_view = cv2.resize(image_bird_view, (360, 360))
        show_image_bird_view = cv2.cvtColor(show_image_bird_view, cv2.COLOR_GRAY2RGB)
        show_color_warp = cv2.resize(color_warp, (360, 360))
        show_color_warp = cv2.addWeighted(show_image_bird_view, 1, show_color_warp, 0.5, 0)
        show_window_img = cv2.resize(window_img, (360, 360))

        heatmap = cv2.resize(heatmap, (360, 360))
        show_heat_map = np.zeros_like(show_image_bird_view)
        show_heat_map[:, :, 2] = (heatmap * 25).astype(np.uint8)
        return img_out, show_image_bird_view, show_color_warp, show_window_img, show_heat_map
    elif show_birdview:
        show_image_bird_view = cv2.resize(image_bird_view, (360, 360))
        show_image_bird_view = cv2.cvtColor(show_image_bird_view, cv2.COLOR_GRAY2RGB)
        show_color_warp = cv2.resize(color_warp, (360, 360))
        show_color_warp = cv2.addWeighted(show_image_bird_view, 1, show_color_warp, 0.5, 0)
        return img_out, show_image_bird_view, show_color_warp
    elif show_heatmaps:
        show_window_img = cv2.resize(window_img, (360, 360))
        show_heat_map = cv2.resize(heatmap, (360, 360))
        show_heat_map = cv2.cvtColor(show_heat_map, cv2.COLOR_GRAY2RGB)
        return img_out, show_window_img, show_heat_map

    return img_out


def process_image_yolov2(image, show_birdview=False):
    global count_h1, count_h2
    # Apply a distortion correction to raw images.
    image = cv2.undistort(image, mtx, dist, None, None)

    # Use color transforms, gradients to find the object edge and change into binary image
    image_binary = combing_color_thresh(image)

    # Transform image to bird view
    image_bird_view = cv2.warpPerspective(image_binary, perspective_M, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # find the road lines, curvature and distance between car_center and road_center
    color_warp, curv, center, left_or_right, left_line.new_fit, right_line.new_fit, left_line.allx, right_line.allx = histogram_search(image_bird_view)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warp_back = cv2.warpPerspective(color_warp, inver_perspective_M, (image.shape[1], image.shape[0]))

    # yolo_v2 vehicles detections
    image = testing_yolov2(image)

    # Combine the result with the original image
    img_out = cv2.addWeighted(image, 1, warp_back, 0.3, 0)

    # Add description on images
    text1 = "Radius of Curature = {:.2f}(m)".format(curv)
    text2 = "Vehicle is {:.3f}m {} of center".format(abs(center), left_or_right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, text1, (50, 50), font, 1.5, color=(255, 255, 255), thickness=3)
    cv2.putText(img_out, text2, (50, 100), font, 1.5, color=(255, 255, 255), thickness=3)

    if show_birdview:
        show_image_bird_view = cv2.resize(image_bird_view, (360, 360))
        show_image_bird_view = cv2.cvtColor(show_image_bird_view, cv2.COLOR_GRAY2RGB)
        show_color_warp = cv2.resize(color_warp, (360, 360))
        show_color_warp = cv2.addWeighted(show_image_bird_view, 1, show_color_warp, 0.5, 0)
        return img_out, show_image_bird_view, show_color_warp

    return img_out


def process_video_yolov2(image, show_birdview=False):
    global count_h1, count_h2
    # Apply a distortion correction to raw images.
    image = cv2.undistort(image, mtx, dist, None, None)

    # Use color transforms, gradients to find the object edge and change into binary image
    image_binary = combing_color_thresh(image)

    # Transform image to bird view
    image_bird_view = cv2.warpPerspective(image_binary, perspective_M, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # find the road lines, curvature and distance between car_center and road_center
    if not left_line.detected or not right_line.detected:
        color_warp, curv, center, left_or_right, left_line.new_fit, right_line.new_fit, \
        left_line.allx, right_line.allx = histogram_search(image_bird_view)
        count_h1 += 1
    else:
        color_warp, curv, center, left_or_right, left_line.new_fit, right_line.new_fit, \
        left_line.allx, right_line.allx = histogram_search2(image_bird_view, left_line.best_fit, right_line.best_fit)
        count_h2 += 1

    # Check the lines health
    left_line.fit_fix()
    right_line.fit_fix()

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warp_back = cv2.warpPerspective(color_warp, inver_perspective_M, (image.shape[1], image.shape[0]))

    # svm vehicles detections
    image = testing_yolov2(image)

    # Combine the result with the original image
    img_out = cv2.addWeighted(image, 1, warp_back, 0.3, 0)

    # Add description on images
    text1 = "Radius of Curature = {:.2f}(m)".format(curv)
    text2 = "Vehicle is {:.3f}m {} of center".format(abs(center), left_or_right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, text1, (50, 50), font, 1.5, color=(255, 255, 255), thickness=3)
    cv2.putText(img_out, text2, (50, 100), font, 1.5, color=(255, 255, 255), thickness=3)

    if show_birdview:
        show_image_bird_view = cv2.resize(image_bird_view, (360, 360))
        show_image_bird_view = cv2.cvtColor(show_image_bird_view, cv2.COLOR_GRAY2RGB)
        show_color_warp = cv2.resize(color_warp, (360, 360))
        show_color_warp = cv2.addWeighted(show_image_bird_view, 1, show_color_warp, 0.5, 0)
        return img_out, show_image_bird_view, show_color_warp

    return img_out


def test_image_svm():
    # random chose image to test
    random_chose = random.randint(0, len(IMAGES_PATH)-1)
    img_test = IMAGES_PATH[random_chose]
    print(img_test)
    img = cv2.imread('./test_images/test1.jpg')

    img_out = process_image_svm(img)

    # Converter BGR -> RGB for plt show
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.title('Output Image')
    plt.imshow(img_out, cmap='gray')
    plt.show()


def test_images_svm():
    for path in IMAGES_PATH:
        img = cv2.imread(path)
        img_out, show_image_bird_view, show_color_warp = process_image_svm(img, show_birdview=True)
        add_image = np.vstack((show_image_bird_view, show_color_warp))
        img_out = np.hstack((img_out, add_image))
        img_out_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.split(path)[-1].split('.')[0] + 'svm_.png')
        cv2.imwrite(img_out_path, img_out)


def test_video_svm():
    video_file = 'project_video.mp4'
    video_output_file = os.path.join(VIDEO_OUTPUT_DIR, video_file.split('.')[0] + '_svm.avi')

    # Video save
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_file, fourcc, 20, (2000, 720))

    # Video read
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_out, show_image_bird_view, show_color_warp, show_windows_img, show_heat_map = \
                process_video_svm(frame, show_birdview=True, show_heatmaps=True)
            add_image1 = np.vstack((show_image_bird_view, show_color_warp))
            add_image2 = np.vstack((show_windows_img, show_heat_map))
            img_out = np.hstack((img_out, add_image1, add_image2))
            out.write(img_out)
            cv2.imshow('frame', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print("h1: {}\th2: {}".format(count_h1, count_h2))
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def test_image_yolov2():
    # random chose image to test
    random_chose = random.randint(0, len(IMAGES_PATH)-1)
    img_test = IMAGES_PATH[random_chose]
    print(img_test)
    img = cv2.imread('./test_images/test1.jpg')

    img_out = process_image_yolov2(img)

    # Converter BGR -> RGB for plt show
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.title('Output Image')
    plt.imshow(img_out, cmap='gray')
    plt.show()


def test_images_yolov2():
    for path in IMAGES_PATH:
        img = cv2.imread(path)
        img_out, show_image_bird_view, show_color_warp = process_image_yolov2(img, show_birdview=True)
        add_image = np.vstack((show_image_bird_view, show_color_warp))
        img_out = np.hstack((img_out, add_image))
        img_out_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.split(path)[-1].split('.')[0] + '_yolov2.png')
        cv2.imwrite(img_out_path, img_out)


def test_video_yolov2():
    video_file = 'project_video.mp4'
    video_output_file = os.path.join(VIDEO_OUTPUT_DIR, video_file.split('.')[0] + '_yolov2.avi')

    # Video save
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_file, fourcc, 20, (1640, 720))

    # Video read
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_out, show_image_bird_view, show_color_warp, = process_video_yolov2(frame, show_birdview=True)
            add_image = np.vstack((show_image_bird_view, show_color_warp))
            img_out = np.hstack((img_out, add_image))
            out.write(img_out)
            cv2.imshow('frame', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print("h1: {}\th2: {}".format(count_h1, count_h2))
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    choose = input("Please choose which detection model\n1) Yolo v2\n2) SVM\nPlease input 1 or 2: ")
    if choose == '1':
        MODE = 'yolov2'
    elif choose == '2':
        MODE = 'svm'
    else:
        MODE = 'yolov2'
    # Run svm detection
    if MODE == 'svm':
        from svm_training import testing_svm
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

        # run on test image
        # test_image_svm()

        # run on test images
        # test_images_svm()

        # run on test video
        test_video_svm()
    elif MODE == 'yolov2':
        from yolo_v2_training import testing_yolov2
        # run on test image
        # test_image_yolov2()

        # run on test images
        # test_images_yolov2()

        # run on test video
        test_video_yolov2()
