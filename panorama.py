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


def main():
    # Add all images in dir to cv2 and add to list
    for file in files:
        print('images/' + file)
        image = cv2.imread('images/' + file)
        images.append(image)

    row, col = 1, len(images)
    fig, axs = plt.subplots(row, col, figsize=(10, 5))
    fig.tight_layout()
    fig.suptitle('The images that will be stitched together. Close window to continue.', fontsize=16)
    i = 0

    #Displays the images just read in
    for image in images:
        axs[i].imshow(image)
        i += 1
    plt.show()

    row, col = 1, 2
    fig, axs = plt.subplots(row, col, figsize=(10, 5))
    fig.tight_layout()

    # Passes image list to stitcher to stitch together
    (status, stitched) = stitcher.stitch(images)

    # Continues if stitching is successful
    if status == 0:
        # Adds stitched image to plot
        axs[0].imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Stitched')

        cropped = crop_img(stitched)

        # Adds cropped image to plot
        axs[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Cropped')

        plt.show()

        #Writes a stitched and cropped version of the panoramic image
        print("\nSaved: stitched.jpg, cropped.jpg")
        cv2.imwrite('stitched.jpg', stitched)
        cv2.imwrite('cropped.jpg', cropped)

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
    best = find_largest_contour(contours)

    # Creates a 4 point polygon
    outer = find_outer_points(best)

    if outer is None:
        print("Cannot find outer coordinates")
        return

    # Gathers actual x, y coordinates for cropping

    # Sort by x value
    outer = sorted(outer, key=lambda x: x[0][0])
    x1 = np.maximum(outer[0][0][0], outer[1][0][0])
    x2 = np.minimum(outer[2][0][0], outer[3][0][0])

    # Sort by y value
    outer = sorted(outer, key=lambda x: x[0][1])
    y1 = np.maximum(outer[0][0][1], outer[1][0][1])
    y2 = np.minimum(outer[2][0][1], outer[3][0][1])

    # Crops image with points found above
    print("\nCropped Coordinates:")
    print("%d:%d , %d:%d" % (y1, y2, x1, x2))
    cropped = image[y1: y2, x1: x2]

    return cropped


# Returns largest contour
def find_largest_contour(contours):
    max_area = 0
    best = None
    for c in contours:
        cur = cv2.contourArea(c)
        if cur > max_area:
            max_area = cur
            best = c
    return best


# Attempts to approximate given contour down to 4 points
def find_outer_points(contour):
    i = 0.05
    approx = cv2.approxPolyDP(contour, i * cv2.arcLength(contour, True), True)
    previous = ""
    while len(approx) != 4:
        if len(approx) < 4:
            if previous == "increment": return None
            i -= 0.01
            previous = "decrement"
        elif len(approx) > 4:
            if previous == "decrement": return None
            i += 0.01
            previous = "increment"
        approx = cv2.approxPolyDP(contour, i * cv2.arcLength(contour, True), True)
    return approx


if __name__ == '__main__':
    main()
