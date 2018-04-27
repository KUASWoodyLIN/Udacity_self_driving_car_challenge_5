import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


if __name__ == '__main__':
    test_image = 'straight_lines1.jpg'
    test_image = '../test_images/test4.jpg'

    img = plt.imread(test_image)
    height, width, _ = img.shape
    vertices = np.array([[[100, height], [440, 310], [525, 310], [width - 70, height]]], dtype=np.int32)
    mask_test = region_of_interest(img, vertices)

    plt.imshow(mask_test)
    plt.show()
