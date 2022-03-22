import imageio
import numpy as np
import os
from os import walk
from skimage.transform import resize
import random
import math
DS_DIR = "testsets/classic5"

# 0-255 range
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

# placeholder
def processImages(imgs):
    processed = []
    for img in imgs:
        processed.append(img)
    return processed

def compareProcessed(imgsOriginal, imgsProcessed):
    PSNRs = np.empty(shape=(len(imgsOriginal)), dtype=np.float32)
    print("len:",len(imgsOriginal))
    for i in range(0, len(imgsOriginal)):
        mse = np.mean((imgsOriginal[i] - imgsProcessed[i]) ** 2)
        if mse == 0:
            PSNRs[i] = 100
            continue
        PIXEL_MAX = 255.0

        PSNRs[i] = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        print("weights:", PSNRs[i])
    return PSNRs
def changeImgTest(imgsOriginal):
    images = []
    for i in range(0, len(imgsOriginal)):
        images.append(np.subtract(imgsOriginal[i], 0.1))
    return images
def readDS():
    images = read_images(DS_DIR)
    print(compareProcessed(images, changeImgTest(images)))
    return images


if __name__ == '__main__':
    readDS()
