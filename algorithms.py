import os
import sys
import io
import scipy.ndimage
from skimage.io import imread
from pixelsort import pixelsort
import piexif

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

def pixel_sort(img, img_mask):
    # Stores beginning and end of row
    selected_row = [-1,-1]
    # Sort pixels horizontally
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            val = img_mask[row][col][0]
            if val == 255:
                if selected_row[0] == -1:
                    selected_row[0] = col
            else:
                if selected_row[0] != -1:
                    selected_row[1] = col
                    np.random.shuffle(img[row][selected_row[0]:selected_row[1]])
                    selected_row = [-1, -1]
        selected_row = [-1, -1]
    return img

def fill_in(img, img_mask):

    sumPixels = np.array([0, 0, 0]) #RGB
    pixels = []
    N = [0] #Number of pixels in group

    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            val = img_mask[row][col][0]
            if val == 255:
                fill_in_dfs(col, row, pixels, img, img_mask, sumPixels, N)
                avgPixel = sumPixels / N[0]
                for i in pixels:
                    img[i[0]][i[1]][0] = avgPixel[0]
                    img[i[0]][i[1]][1] = avgPixel[1]
                    img[i[0]][i[1]][2] = avgPixel[2]

                # Reset data structres
                sumPixels = np.array([0, 0, 0])
                pixels = []
                N[0] = 0

    return img

def fill_in_dfs(col, row, pixels, img, img_mask, sumPixels, N):

    sumPixels = np.add(sumPixels, img[row][col])
    N[0] += 1
    pixels.append((row, col))
    img_mask[row][col][0] = 0

    #left
    if col > 0 and img_mask[row][col-1][0] == 255:
        fill_in_dfs(col-1, row, pixels, img, img_mask, sumPixels, N)
    #top
    if row > 0 and img_mask[row-1][col][0] == 255:
        fill_in_dfs(col, row-1, pixels, img, img_mask, sumPixels, N)
    #right
    if col < len(img_mask[0])-1 and img_mask[row][col+1][0] == 255:
        fill_in_dfs(col+1, row, pixels, img, img_mask, sumPixels, N)
    #bottom
    if row > len(img_mask)-1 and img_mask[row+1][col][0] == 255:
        fill_in_dfs(col, row+1, pixels, img, img_mask, sumPixels, N)


def pixel_sort2(img, img_mask):
    a = pixelsort(img, interval_image=img_mask, interval_function="file")
    return np.asarray(a)[:,:,:3]

def black_bar(img, img_mask):

    BLACK_COLOR = (0,0,0)
    img_mask = (img_mask[:,:,:1] > 0.9).astype(np.uint8)
    # RUN DFS
    count = 2
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            val = img_mask[row][col][0]
            if val == 1:
                black_bar_dfs(col, row, img_mask, count)
                count += 1
    print("count", count)

    print("should be 1", img_mask[1980][1520][0])
    print("should be 3", img_mask[2040][4030][0])
    print(img.size[0], img.size[1])

    count = 2
    segment = [img.size[0], img.size[1], 0 ,0]
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            if count == img_mask[row][col][0]:
                # left
                if col < segment[0]:
                    segment[0] = col
                # top
                if row < segment[1]:
                    segment[1] = row
                # right
                if col > segment[2]:
                    segment[2] = col
                # bottom
                if row > segment[3]:
                    segment[3] = row
    img.paste( BLACK_COLOR, [segment[0],segment[1],segment[2],segment[3]])
    img = np.array(img)
    # colors = [[24, 67, 88], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    # for row in range(len(img_mask)):
    #     for col in range(len(img_mask[row])):
    #         if img_mask[row][col][0] > 0:
    #             # print(int(int(img_mask[row][col][0])%4))
    #             if int(img_mask[row][col][0]) < 4:
    #                 img[row][col] = colors[int(img_mask[row][col][0])]
    #             else:
    #                 img[row][col] = colors[int(int(img_mask[row][col][0])%4)]

    return img

def black_bar_dfs(col, row, img_mask, count):
    img_mask[row][col][0] = count
    #left
    if col > 0 and img_mask[row][col-1][0] == 1:
        black_bar_dfs(col-1, row, img_mask, count)
    #top
    if row > 0 and img_mask[row-1][col][0] == 1:
        black_bar_dfs(col, row-1, img_mask, count)
    #right
    if col < len(img_mask[0])-1 and img_mask[row][col+1][0] == 1:
        black_bar_dfs(col+1, row, img_mask, count)
    #bottom
    if row > len(img_mask)-1 and img_mask[row+1][col][0] == 1:
        black_bar_dfs(col, row+1, img_mask, count)

def adjust_exif2(tags_chosen,exif):
    new_exif = dict(exif)
    tag_space_list = ["0th", "Exif", "GPS", "1st"]
    i =0
    for index,tag_space in enumerate(tag_space_list):
        print("Tag Space: ",tag_space)
        for chosen in tags_chosen:
            try:
                if index == 0:
                    #tag_num = piexif.ImageIFD.__getattribute__(piexif.ImageIFD,chosen)
                    new_exif[tag_space][piexif.ImageIFD.__getattribute__(piexif.ImageIFD,chosen)] = ""
                elif index == 1:
                    #print(type(new_exif[tag_space][piexif.ExifIFD.__getattribute__(piexif.ExifIFD,chosen)]))
                    new_exif[tag_space][piexif.ExifIFD.__getattribute__(piexif.ExifIFD,chosen)] = b''
                elif index == 2:
                    tagType = type(new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)])
                    if tagType is bytes:
                        new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)] = b''
                    elif tagType is int:
                        new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)] = 0
                    elif tagType is tuple:
                        new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)] = (0,0)
                else:
                   new_exif[tag_space][piexif.InteropIFD.__getattribute__(piexif.InteropIFD, chosen)] = ""
                i+=1
                print("removed: ",chosen, i,"/",len(tags_chosen))
            except: continue #this accounts for the fact that each tag doesnt exist in every tag_space
    return new_exif

def metadata_erase(img, exif, tags):
    exif = piexif.load(exif)
    new_exif = adjust_exif2(tags,exif)
    new_bytes = piexif.dump(new_exif)
    outputImage = io.BytesIO()
    piexif.insert(new_bytes, img, outputImage)
    return outputImage


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
