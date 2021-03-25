import os
import sys

import scipy.ndimage
from skimage.io import imread

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

def guassian_blur(img, img_mask, sigma=5):
    mask = img_mask[:,:,:1].astype(np.float)
    filter = scipy.ndimage.filters.gaussian_filter(img*mask, sigma=(sigma, sigma, 0))
    weights = scipy.ndimage.filters.gaussian_filter(mask, sigma=(sigma, sigma, 0))
    filter /= weights + 0.001

    filter = filter.astype(np.uint8)
    inv_mask = (mask < 1.0)
    filter -= filter*inv_mask
    img = (img*inv_mask)
    img += filter
    img = img.astype(np.uint8)
    return img

def pixelization(img, mask_img):
    dim_x, dim_y = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    inv_mask = (mask_img < 1.0)
    imgSmall = img.resize((dim_x//16,dim_y//16),resample=Image.BILINEAR)
    imgSmall = imgSmall.resize(img.size,Image.NEAREST)
    imgSmall -= imgSmall*inv_mask
    img = (img*inv_mask)
    img += imgSmall
    img = img.astype(np.uint8)
    return np.array(img)


def main():
    print('run')
    img_path = sys.argv[1]
    mask_path = sys.argv[2]

    img = imread(img_path)[:,:,:3]
    # img = img.astype(numpy.double)
    mask = imread(mask_path)[:,:,:1].astype(np.float)
    mask = 1.0 * (mask > 1)
    print(mask)

    x = np.arange(10)
    print(x)
    print(x[1:-1])
    mask = mask.astype(np.double)
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

# if __name__ == 'main':
#     main()