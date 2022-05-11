import cv2 as cv2
import os
import numpy as np

test_path = next(os.walk('testing/'))[2]
img_set = []
for x, i in enumerate(test_path):
    semantic_mask = cv2.imread('testing/' + i)
    crop_img = semantic_mask[0:368, 0:1232]
    cv2.imwrite('testing_2/%d.png' % x, crop_img)
    img_set.append(crop_img)
