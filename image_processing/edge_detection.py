import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(0, 255)):
    """
    Input: img, thresh
        img = input gray Image
        kernel = kernel size
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    if orient == 'x':
        x = 1
        y = 0
    else:
        x = 0
        y = 1
    # 1) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=kernel)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 3) Scale to 8-bits (0-255) then convert to type = np.unit8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output


# Magnitude of the Gradient
def mag_thresh(img, kernel=3, thresh=(0, 255)):
    """
    Input: img, thresh
        img = input gray Image
        kernel = kernel size
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # 2) Calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 3) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 4) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


# Direction of the Gradient
def dir_thresh(img, kernel=3, thresh=(0, np.pi/2)):
    """
    Input: img, thresh
        img = input gray Image
        kernel = kernel size
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    graddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(graddir)
    binary_output[(graddir >= thresh[0]) & (graddir <= thresh[1])] = 1
    return binary_output


def hls_detect(img, thresh=(0, 255)):
    """
    Input: img, thresh
        img = input RGB Image
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel < thresh[1])] = 1
    return binary_output


def combing_sobel_schannel_thresh(img, kernel=3):
    """
    Is function combing Sobelx, Sobley, Magnitude and Direction Operator,
    Input: img, kernel
        img = input RGB Image
        kernel = kernel size
    Output: binary_output
        binary_output = output binary image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Threshold x gradient
    sobalx_binary = abs_sobel_thresh(img_gray, orient='x', kernel=kernel, thresh=(30, 100))

    # Threshold color channel
    s_channel_binary = hls_detect(img, thresh=(155, 255))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobalx_binary)
    combined_binary[(sobalx_binary == 1) | (s_channel_binary == 1)] = 1

    # plt.figure(figsize=(12, 12))
    # plt.subplot(3, 1, 1)
    # plt.title('Sobal X')
    # plt.imshow(sobalx_binary, cmap='gray')
    # plt.subplot(3, 1, 2)
    # plt.title('HLS S Channel')
    # plt.imshow(s_channel_binary, cmap='gray')
    # plt.subplot(3, 1, 3)
    # plt.title('Combine')
    # plt.imshow(combined_binary, cmap='gray')

    return combined_binary


def combing_smd_thresh(img, kernel=3):
    """
    Is function combing Sobelx, Sobley, Magnitude and Direction Operator,
    Input: img, kernel
        img = input RGB Image
        kernel = kernel size
    Output: binary_output
        binary_output = output binary image
    """
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', kernel=kernel, thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', kernel=kernel, thresh=(30, 100))
    mag_binary = mag_thresh(img, kernel=kernel, thresh=(20, 100))
    dir_binary = dir_thresh(img, kernel=kernel, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def combing_color_thresh(img):
    # hsv yellow and white
    yellow_low = np.array([0, 80, 100])
    yellow_high = np.array([50, 255, 255])
    white_low = np.array([18, 0, 200])          # 18, 0, 180
    white_high = np.array([255, 80, 255])       # 255, 80, 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, yellow_low, yellow_high)
    mask_white = cv2.inRange(hsv, white_low, white_high)

    # RGB RED channel
    thresh = (215, 255)
    binary = np.zeros_like(img[:, :, 2])    # RED channel
    binary[(img[:, :, 2] > thresh[0]) & (img[:, :, 2] <= thresh[1])] = 255

    # combined mask
    combined_mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
    combined_mask = cv2.bitwise_or(combined_mask_yw, binary)

    # color_binary = np.dstack((np.zeros_like(combined_mask), binary, combined_mask_yw))
    # fig = plt.figure(figsize=(10, 9))
    # gs2 = gridspec.GridSpec(3, 2)
    # plt.subplot(gs2[0, :])
    # plt.title('Original image')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.subplot(gs2[1, 0])
    # plt.title('HSV detect yellow color')
    # plt.imshow(mask_yellow, cmap='gray')
    # plt.subplot(gs2[1, 1])
    # plt.title('HSV detect white color')
    # plt.imshow(mask_white, cmap='gray')
    # plt.subplot(gs2[2, 0])
    # plt.title('RGB R channel detect')
    # plt.imshow(binary, cmap='gray')
    # plt.subplot(gs2[2, 1])
    # plt.title('Combined HSV and R channel detect')
    # plt.imshow(color_binary)
    #
    # plt.savefig('../output_images/edge_detection2.png', bbox_inches='tight')

    return combined_mask


if __name__ == '__main__':

    # test_image = '../test_images/straight_lines1.jpg'
    test_image = '../test_images/test3.jpg'
    # test_image = '../test_images/project1.png'
    # Read image
    img = cv2.imread(test_image)

    # bird-eye view
    offset = 1280 / 2
    src = np.float32([(596, 447), (683, 447), (1120, 720), (193, 720)])  # Longer one
    dst = np.float32([(offset - 300, 0), (offset + 300, 0), (offset + 300, 720), (offset - 300, 720)])
    perspective_M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, perspective_M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Image transform
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel Operator
    img_sobelx_out = abs_sobel_thresh(img_gray, orient='x', kernel=5, thresh=(40, 100))
    img_sobely_out = abs_sobel_thresh(img_gray, orient='y', kernel=7, thresh=(30, 100))
    # Magnitude Operator
    img_mag_out = mag_thresh(img_gray, kernel=7, thresh=(20, 100))
    # Direction Operator
    img_dir_out = dir_thresh(img_gray, kernel=7, thresh=(0.7, 1.3))
    # Combined Sobelx, Sobely, Magnitude, Direction the operator
    img_comb_out_1 = combing_smd_thresh(img_gray, kernel=7)

    # # Show original image
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Customizing Figure Layouts
    fig = plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(3, 2, right=0.48, wspace=0.1)

    ax1 = fig.add_subplot(gs1[0, 0])
    plt.title('Sobel X')
    plt.imshow(img_sobelx_out, cmap='gray')

    ax2 = fig.add_subplot(gs1[0, 1])
    plt.title('Sobel Y')
    plt.imshow(img_sobely_out, cmap='gray')

    ax3 = fig.add_subplot(gs1[1, 0])
    plt.title('Magnitude')
    plt.imshow(img_mag_out, cmap='gray')

    ax4 = fig.add_subplot(gs1[1, 1])
    plt.title('Direction')
    plt.imshow(img_dir_out, cmap='gray')

    ax5 = fig.add_subplot(gs1[2, :])
    plt.title('Combined')
    plt.imshow(img_comb_out_1, cmap='gray')

    # HLS S Channel detection
    img_s_output = hls_detect(img, thresh=(155, 255))
    # Combined Sobelx, HLS S Channel
    img_comb_out_2 = combing_sobel_schannel_thresh(img, kernel=7)
    # HSV yellow and white thresh
    img_comb_out_3 = combing_color_thresh(img)

    plt.figure(figsize=(8, 12))
    plt.subplot(4, 1, 1)
    plt.title('Sobel X')
    plt.imshow(img_sobelx_out, cmap='gray')
    plt.subplot(4, 1, 2)
    plt.title('HLS S Channel')
    plt.imshow(img_s_output, cmap='gray')
    plt.subplot(4, 1, 3)
    plt.title('Combined Sobelx and HLS S Channel')
    plt.imshow(img_comb_out_2, cmap='gray')
    plt.subplot(4, 1, 4)
    plt.title('Combined HSV_yellow and HSV_white')
    plt.imshow(img_comb_out_3, cmap='gray')

    plt.show()
