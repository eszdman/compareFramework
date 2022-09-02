import os
import math
import numpy as np
from PIL import Image
from oct2py import Oct2Py

PIXEL_MAX = 255.0
CFFT_PATH = os.path.join(os.getcwd(), "plugins", "cfft")


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
        results = dict()
        for algorithm in self.algorithms:
            processed_images = []
            for image in self.images:
                processed_images.append([image,
                                         algorithm.run(os.path.join(self.dataset_path, image), self.weight, self.height,
                                                       self.block_size,
                                                       self.compression)])
            results[algorithm] = processed_images
        return results


class CFFT_algorithm:
    def __init__(self):
        global CFFT_PATH
        self.octave = Oct2Py()
        self.octave.eval("pkg load image")
        self.octave.addpath(CFFT_PATH)

    def run(self, image_path, width, height, block, value):
        return self.octave.CFFT(image_path, width, height, block, value)


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


if __name__ == '__main__':
    comparator = Comparator(os.path.join(CFFT_PATH, "dataset"), 256, 256, 16, 100)
    comparator.list_images()

    cfft = CFFT_algorithm()
    comparator.add_algo(cfft)

    result = comparator.run()
    for algorithm in result.keys():
        for name, image_array in result[algorithm]:
            name = name.split('.')[0] + '.png'
            image = Image.fromarray(image_array)
            image.save(os.path.join(CFFT_PATH, "results", name))
    print(result)
