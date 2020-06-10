import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
cv2.__version__

# Get the name of all files in images dir to add to list
path, dirs, files = next(os.walk('images/'))
images: list = []
# To stitch all images together
stitcher: cv2.Stitcher = cv2.Stitcher_create()

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(10, 5))
fig.tight_layout()


def main():
    # Add all images in dir to cv2 and add to list
    for file in files:
        print('images/' + file)
        image = cv2.imread('images/' + file)
        images.append(image)

    # Passes image list to stitcher to stitch together
    (status, stitched) = stitcher.stitch(images)

    # Continues if stitching is successful
    if status == 0:
        # Adds stitched image to plot
        axs[0].imshow(stitched)
        axs[0].set_title('Stitched')

        cropped = crop_img(stitched)

        # Adds cropped image to plot
        axs[1].imshow(cropped)
        axs[1].set_title('Cropped')

        plt.show()

        cv2.imwrite('stitched.jpg', stitched)
        cv2.imwrite('cropped.jpg', cropped)
        cv2.waitKey(0)
    else:
        print('Stitching failed (' + str(status) + ')')


def crop_img(image):
    # Convert image to grayscale for contour finding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Only value not in threshold is pure black, which is the outer area of the image
    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Finds largest contour, AKA the entire image outline
    max_area = 0
    best = None
    for c in contours:
        cur = cv2.contourArea(c)
        if cur > max_area:
            max_area = cur
            best = c

    # Creates a 4 point polygon
    approx = cv2.approxPolyDP(best, 0.01 * cv2.arcLength(c, True), True)

    # Finds the inner points, removing all black outer area
    x1 = np.maximum(approx[0][0][0], approx[1][0][0])
    y1 = np.minimum(approx[1][0][1], approx[2][0][1])
    x2 = np.minimum(approx[2][0][0], approx[3][0][0])
    y2 = np.maximum(approx[0][0][1], approx[3][0][1])

    # Crops image with points found above
    cropped = image[y2: y1, x1: x2]

    return cropped


if __name__ == '__main__':
    main()
