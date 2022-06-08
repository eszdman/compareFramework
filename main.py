import math
import os
import sys
from ctypes import *
from os import walk

import imageio
import numpy as np

DS_DIR = "dataset/classic5"


class Comparator:
    def __init__(self, dataset_path, weight, height, block_size, compression):
        self.dataset_path = dataset_path
        self.weight = weight
        self.height = height
        self.block_size = block_size
        self.compression = compression
        self.algorithms = list()
        self.images = list()

    def add_algo(self, algorithm):
        self.algorithms.append(algorithm)

    def list_images(self):
        files = os.listdir(self.dataset_path)
        for file in files:
            self.images.append(file)

    def run(self):
        for algorithm in self.algorithms:
            processed_images = []
            for image in self.images:
                processed_images.append(algorithm.run(f'{self.dataset_path}/{image}', self.weight, self.height,
                                                      self.compression))


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
    cfft = CFFT_algorithm()
    cfft.run("dataset/classic5/baboon.bmp", 500, 500, 10, 10)
    return images


class CFFT_algorithm:
    def __init__(self):
        sys.path.insert(1, "C:\\Program Files\\MATLAB\\MATLAB Runtime\\v912\\bin\\win64\\")
        self.lib = load_plugin(r"plugins/CFFT/CFFT.dll")

    def run(self, image_path, width, height, block, value):
        result = []
        print(vars(self.lib))
        result = self.lib.CFFT(image_path, width, height, block, value, result)
        return result


def load_plugin(path):
    return cdll.LoadLibrary(path)


if __name__ == '__main__':
    comparator = Comparator(DS_DIR, 512, 512, 10, 10)
    comparator.list_images()

    cfft = CFFT_algorithm()
    comparator.add_algo(cfft)

    comparator.run()
