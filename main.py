import imageio
import numpy as np
import os
from os import walk
from skimage.transform import resize
import random

DS_DIR = "testsets/classic5"


def read_images(directory=""):
    directory += "./"
    filenames = next(walk(directory), (None, None, []))[2]
    images = []
    for file in filenames:
        img = imageio.imread(directory + file)
        # if resize needed
        # img = resize(img,(TileSize,TileSize))
        images.append(img)
    return images


def create_dataset(images):
    # создаём датасет из картинок
    # number = random.randint(0, len(images))
    data_set = np.empty(shape=(len(images), len(images[0]), len(images[0][0]), 1), dtype=np.float32)
    print(data_set.shape)
    # data_set = np.empty([2, 2], dtype=int)
    for n in range(0, len(images)):
        for i in range(0, len(images[n])):
            for j in range(0, len(images[n][i])):
                data_set[n][i][j][0] = (images[n][i][j])
        # data_set.append([inputImgs[n]])
    return data_set


def readDS():
    images = read_images(DS_DIR)
    dataset = create_dataset(images)
    return dataset


if __name__ == '__main__':
    readDS()
