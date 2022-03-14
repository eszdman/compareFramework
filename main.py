import imageio
import numpy as np
import os
from os import walk
from skimage.transform import resize
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
DS_DIR = "testsets/classic5"


def readImages(directory=""):
    directory += "./"
    filenames = next(walk(directory), (None, None, []))[2]
    images = []
    for file in filenames:
        img = imageio.imread(directory + file)
        # if resize needed
        # img = resize(img,(TileSize,TileSize))
        images.append(img)
    return images


def createDataset(images):
    # создаём датасет из картинок
    #number = random.randint(0, len(images))
    dataSet = np.empty(shape=(len(images), len(images[0]), len(images[0][0]), 1), dtype=np.float32)
    print(dataSet.shape)
    # dataSet = np.empty([2, 2], dtype=int)
    for n in range(0, len(images)):
        for i in range(0, len(images[n])):
            for j in range(0, len(images[n][i])):
                dataSet[n][i][j][0] = (images[n][i][j])
        # dataSet.append([inputImgs[n]])
    return dataSet


def readDS():
    imgs = readImages(DS_DIR)
    dataset = createDataset(imgs)
    return dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    readDS()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
