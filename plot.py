import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
from pylab import rcParams
from matplotlib import gridspec
import numpy as np

###########plot for report

img_set = []

for path in glob.glob("New folder/"):
    for img_path in glob.glob(os.path.join(path, "*.png")):
        img = mpimg.imread(img_path)
        img_set.append(img)

# importing required library
import matplotlib.pyplot as plt
import numpy as np

# creating grid for subplots
fig, (ax1, ax2) = plt.subplots(2, 1,dpi=200,figsize=[5,3.25])

plt.rcParams["figure.autolayout"] = True

ax1.imshow(img_set[2])
ax1.set_ylabel('Original')

ax2.imshow(img_set[3])
ax2.set_ylabel('Segmented')

# ax3.imshow(img_set[4])
# ax3.set_ylabel('Original')
#
# ax4.imshow(img_set[5])
# ax4.set_ylabel('Segmented')

plt.savefig('pic2.png')

plt.show()