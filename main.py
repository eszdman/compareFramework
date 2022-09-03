import os
import math
import numpy as np
from PIL import Image
from oct2py import Oct2Py

PIXEL_MAX = 255.0
CFFT_PATH = os.path.join(os.getcwd(), "plugins", "cfft")


class Comparator:
    def __init__(self, dataset_path, block_size, compression):
        self.dataset_path = dataset_path
        self.block_size = block_size
        self.compression = compression
        self.algorithms = list()
        self.images_paths = list()
        self.images_original = list()

    def add_algo(self, algorithm):
        self.algorithms.append(algorithm)

    def list_images(self):
        files = os.listdir(self.dataset_path)
        for file in files:
            self.images_paths.append(file)

    def load_images(self):
        for image_path in self.images_paths:
            image = Image.open(os.path.join(self.dataset_path, image_path))
            self.images_original.append(image)

    def run(self):
        for algorithm in self.algorithms:
            for i, image in enumerate(self.images_paths):
                processed_image = algorithm.run(os.path.join(self.dataset_path, image), self.images_original[i].width,
                                                self.images_original[i].height,
                                                self.block_size, self.compression)
                algorithm.processed_images.append([image, processed_image])

    def compareProcessed(self):
        for algorithm in self.algorithms:
            for i in range(0, len(self.images_paths)):
                mse = np.mean((np.asarray(self.images_original[i]) - algorithm.processed_images[i][1]) ** 2)
                if mse == 0:
                    algorithm.psnrs.append(100)
                    continue

                algorithm.psnrs.append(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

    def print(self):
        for algorithm in self.algorithms:
            algorithm.print()


class CFFT_algorithm:
    def __init__(self):
        self.psnrs = list()
        self.processed_images = list()

        global CFFT_PATH
        self.octave = Oct2Py()
        self.octave.eval("pkg load image")
        self.octave.addpath(CFFT_PATH)

    def run(self, image_path, width, height, block, value):
        return self.octave.CFFT(image_path, width, height, block, value)

    def print(self):
        print("CFFT")
        for i in range(len(self.processed_images)):
            print(self.processed_images[i][0], self.psnrs[i])
        print()


if __name__ == '__main__':
    comparator = Comparator(os.path.join(CFFT_PATH, "dataset"), 16, 100)
    comparator.list_images()
    comparator.load_images()

    cfft = CFFT_algorithm()
    comparator.add_algo(cfft)

    comparator.run()
    comparator.compareProcessed()
    comparator.print()
    # for algorithm in result.keys():
    #     for name, image_array in result[algorithm]:
    #         name = name.split('.')[0] + '.png'
    #         image = Image.fromarray(image_array)
    #         image.save(os.path.join(CFFT_PATH, "results", name))
    # print(result)
