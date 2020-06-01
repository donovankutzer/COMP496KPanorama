import numpy as np
import argparse
import cv2
import os

# Get the name of all files in images dir to add to list
path, dirs, files = next(os.walk('images/'))
images: list = []

# Add all images in dir to cv2 and add to list
for file in files:
    print('images/' + file)
    image = cv2.imread('images/' + file)
    images.append(image)

# To stitch all images together
stitcher: cv2.Stitcher = cv2.Stitcher_create()


# Passes image list to stitcher to stitch together
(status, stitched) = stitcher.stitch(images)

if status == 0:
    stitched = cv2.resize(stitched, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output.jpg', stitched)
    cv2.imshow('Result', stitched)
    cv2.waitKey(0)
else:
    print('Stitching failed (' + str(status) + ')')

# TODO, expand on this more? Do more manually?
# TODO, crop the pano image to a nice resolution so no black space is visible
