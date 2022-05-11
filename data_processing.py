import cv2 as cv2
import os
import numpy as np
import glob

height = 375
width = 1242

everything_else = 'void'
color_code = 7
vegetation = 21
human = np.array([24, 25])
automobile = np.array([26, 27, 28])

test_path = next(os.walk('semantic_rgb/'))[2]

new_img_set = []
############ process train segmented data with 4 categories
for x, i in enumerate(test_path):
    semantic_mask = cv2.imread('semantic_rgb/' + i, 1)
    new_img = np.zeros((height, width))
    new_img = new_img.astype('int8')
    for a in range(0, 375):
        for b in range(0, 1242):
            if semantic_mask[a][b][0] == 35:
                new_img[a][b] = 1
            elif semantic_mask[a][b][0] == 142:
                new_img[a][b] = 2
            elif semantic_mask[a][b][0] == 70 and semantic_mask[a][b][1] == 0:
                new_img[a][b] = 2
            elif semantic_mask[a][b][1] == 60:
                new_img[a][b] = 2
            elif semantic_mask[a][b][2] == 255:
                new_img[a][b] = 3
            elif semantic_mask[a][b][0] == 60:
                new_img[a][b] = 3
    new_img_set.append(new_img)

for a in range(0,200):
    cv2.imwrite('image_1/' + test_path[a], new_img_set[a])

######## image crop
test_path = next(os.walk('testing/'))[2]
img_set = []
for x, i in enumerate(test_path):
    semantic_mask = cv2.imread('testing/' + i)
    crop_img = semantic_mask[0:368, 0:1232]
    cv2.imwrite('testing_2/%d.png' % x, crop_img)
    img_set.append(crop_img)
