import os
import sys

import scipy.ndimage
from skimage.io import imread

import matplotlib.pyplot as plt

import numpy as np

def guassian_blur(img, img_mask, sigma=5):
    mask = 1.0 * (img_mask > 1)[:,:,:1].astype(np.float)
    filter = scipy.ndimage.filters.gaussian_filter(img*mask, sigma=(sigma, sigma, 0))
    weights = scipy.ndimage.filters.gaussian_filter(mask, sigma=(sigma, sigma, 0))
    filter /= weights + 0.001

    filter = filter.astype(np.int)
    inv_mask = (mask < 1.0)
    filter -= filter*inv_mask
    img = (img*inv_mask).astype(np.int)
    img += filter
    img = img.astype(np.uint8)
    return img

def main():
    img_path = sys.argv[1]
    mask_path = sys.argv[2]

    img = imread(img_path)[:,:,:3]
    # img = img.astype(numpy.double)
    mask = imread(mask_path)[:,:,:1].astype(np.float)
    mask = 1.0 * (mask > 1)
    # mask = mask.astype(numpy.double)
    blur = 10

    filter = scipy.ndimage.filters.gaussian_filter(img*mask, sigma=(blur, blur, 0))
    weights = scipy.ndimage.filters.gaussian_filter(mask, sigma=(blur, blur, 0))
    filter /= weights + 0.001

    filter = filter.astype(np.int)
    inv_mask = (mask < 1.0)
    filter -= filter*inv_mask
    img = (img*inv_mask).astype(np.int)
    img += filter
    img = img.astype(np.int)

    plt.imshow(img)
    plt.show()