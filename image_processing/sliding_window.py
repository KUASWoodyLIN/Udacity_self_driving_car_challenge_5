import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


if __name__ == '__main__':
    # PATH definition
    FUNCTIONS_PATH = os.getcwd()
    ROOT_PATH = os.path.split(FUNCTIONS_PATH)[0]
    IMAGES_TEST = os.path.join(ROOT_PATH, 'test_images')
    IMAGES_OUTPUT = os.path.join(ROOT_PATH, 'output_images')
    y_start_stop = [400, 656]  # Min and max in y to search in slide_window()

    plt.figure(figsize=(12, 8))
    image_path = os.path.join(IMAGES_TEST, 'test1.jpg')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(128, 128), xy_overlap=(0.7, 0.7))

    windows4 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(200, 200), xy_overlap=(0.75, 0.75))

    image_out1 = draw_boxes(image, windows1)
    image_out2 = draw_boxes(image, windows2)
    image_out3 = draw_boxes(image, windows3)
    image_out4 = draw_boxes(image, windows4)

    # Image Show
    plt.subplot(2, 2, 1)
    plt.title('sliding window 64x64')
    plt.imshow(image_out1)

    plt.subplot(2, 2, 2)
    plt.title('sliding window 96x96')
    plt.imshow(image_out2)

    plt.subplot(2, 2, 3)
    plt.title('sliding window 128x128')
    plt.imshow(image_out3)

    plt.subplot(2, 2, 4)
    plt.title('sliding window 200x200')
    plt.imshow(image_out4)

    file_name = 'sliding_window_output.png'
    file_path = os.path.join(IMAGES_OUTPUT, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()

