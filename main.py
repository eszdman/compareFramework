import ctypes
from ctypes import CDLL

import imageio
import numpy as np
import os
from os import walk
from skimage.transform import resize
import random
import math
from ctypes import *

DS_DIR = "dataset/classic5"


# 0-255 range
def read_images(directory="."):
    directory += "/"
    filenames = next(walk(directory), (None, None, []))[2]
    images = []
    for file in filenames:
        img = imageio.imread(directory + file)
        # if resize needed
        # img = resize(img,(TileSize,TileSize))
        images.append(img)
    return images


# placeholder
def processImages(images):
    processed = []
    for image in images:
        processed.append(image)
    return processed


def equalize(image, number_bins=256):
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


PIXEL_MAX = 255.0


def compareProcessed(images_original, images_processed):
    PSNRs = np.empty(shape=(len(images_original)), dtype=np.float32)
    print("len:", len(images_original))
    for i in range(0, len(images_original)):
        mse = np.mean((images_original[i] - images_processed[i]) ** 2)
        if mse == 0:
            PSNRs[i] = 100
            continue

        PSNRs[i] = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        print("weights:", PSNRs[i])
    return PSNRs


def changeImgTest(images_original):
    images = []
    for i in range(0, len(images_original)):
        images.append(np.subtract(images_original[i], 0.1))
    return images


def readDS():
    images = read_images(DS_DIR)
    print(compareProcessed(images, changeImgTest(images)))
    cfft = CFFT()
    cfft.run("dataset/classic5/baboon.bmp",500,500,10,10)
    return images


class CFFT:
    def __init__(self):
        self.cfft = load_plugin("plugins/CFFT.so")

    def run(self, image_path, width, height, block, value):
        result = []
        print(vars(self.cfft))
        result = self.cfft.CFFT(image_path, width, height, value, result)
        return result


def load_plugin(path):
    return cdll.LoadLibrary(path)


if __name__ == '__main__':
    readDS()
