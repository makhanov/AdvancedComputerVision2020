from __future__ import print_function
import random 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.transform import rescale, resize, downscale_local_mean
import glob, os
import fnmatch
import time
import math
import cyvlfeat
# from cyvlfeat import hog
import cv2
import imutils
from pyramid_n_sliding_window import pyramid, sliding_window, pyramid_score

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload

leftL_path = '/Users/makhanov/Yandex.Disk.localized/NU-PhD/Fall2020/ComputerVision/project/datasets/archive2/lungs/left/'
rightL_path = '/Users/makhanov/Yandex.Disk.localized/NU-PhD/Fall2020/ComputerVision/project/datasets/archive2/lungs/right/'
left_lung_imgs = fnmatch.filter(os.listdir(leftL_path), '*.png')
right_lung_imgs = fnmatch.filter(os.listdir(rightL_path), '*.png')

x, y = 500, 500
for i, image_path in enumerate(left_lung_imgs):
    h, w = io.imread(leftL_path+image_path, as_gray=True).shape
    if h<y:
        y=h
    if w<x:
        x=w
dim = (min(y, x), min(y, x))

left_lungs_resized = []
for i, image_path in enumerate(left_lung_imgs):
    img = cv2.imread(leftL_path+image_path, 0)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    left_lungs_resized.append(resized)

right_lungs_resized = []
for i, image_path in enumerate(right_lung_imgs):
    img = cv2.imread(rightL_path+image_path, 0)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    right_lungs_resized.append(resized)

n = len(left_lungs_resized)
LL_shape = left_lungs_resized[0].shape
avg_LL = np.zeros((LL_shape))
for i in left_lungs_resized:
    image = i
    avg_LL = np.asarray(image)+np.asarray(avg_LL)
avg_LL = avg_LL/n

(LL_feature, LL_hog) = hog(avg_LL, visualize=True)

path = '/Users/makhanov/Yandex.Disk.localized/NU-PhD/Fall2020/ComputerVision/project/datasets/archive2/COVID/'
# image = cv2.imread(path + 'Covid(1).png', 0)
(winW, winH) = dim

image_one = io.imread(path + 'Covid(1).png', as_gray=True)


max_score, maxr, maxc, max_scale, max_response_map = pyramid_score \
    (image_one, LL_feature, dim, stepSize = 30, scale=0.8)

fig,ax = plt.subplots(1)
ax.imshow(rescale(image_one, max_scale))
rect = patches.Rectangle((maxc,maxr),winW,winH,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()

plt.imshow(max_response_map,cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.show()