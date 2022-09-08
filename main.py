import os
import math
import numpy as np
from PIL import Image
from oct2py import Oct2Py
import cv2 as cv

PIXEL_MAX = 255.0


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
            if os.path.isfile(os.path.join(self.dataset_path, file)):
                self.images_paths.append(file)

    def load_images(self):
        for image_path in self.images_paths:
            path = os.path.join(self.dataset_path, image_path)
            image = Image.open(path)
            self.images_original.append(image)

    def run(self):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for algorithm in self.algorithms:
            print(algorithm.name)
            for i, image_name in enumerate(self.images_paths):
                processed_image = algorithm.run(os.path.join(self.dataset_path, image_name),
                                                self.images_original[i].width,
                                                self.images_original[i].height,
                                                self.block_size, self.compression)
                processed_image = processed_image.astype("uint8")
                processed_image = clahe.apply(processed_image)
                psnr = self.compareProcessed(i, processed_image)
                print(image_name, psnr)
                algorithm.save_processed_image(self.dataset_path, image_name.split('.')[0], processed_image)

    def compareProcessed(self, i, processed_image):
        mse = np.mean((np.asarray(self.images_original[i]) - processed_image) ** 2)
        if mse == 0:
            return 100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Algorithm:
    def __init__(self, name, path):
        self.name = name
        self.psnrs = list()
        self.processed_images = list()

        self.octave = Oct2Py()
        self.octave.eval("pkg load image")
        self.octave.eval("pkg load signal")
        self.octave.addpath(path)

    def print(self):
        print(self.name)
        for i in range(len(self.processed_images)):
            print(self.processed_images[i][0], self.psnrs[i])
        print()

    def save_processed_image(self, path, name, image):
        image = Image.fromarray(image)
        if image.mode != 'RGB':  # pay attention to this line
            image = image.convert('RGB')  # also this line
        image_name = self.name + '_' + name + ".png"
        image.save(os.path.join(path, "results", image_name))


class CFFT_algorithm(Algorithm):
    def __init__(self):
        super().__init__("CFFT", os.path.join(os.getcwd(), "plugins", "cfft"))

    def run(self, image_path, width, height, block, value):
        return self.octave.CDCT(image_path, width, height, block, value)


class CDCT_algorithm(Algorithm):
    def __init__(self):
        super().__init__("CDCT", os.path.join(os.getcwd(), "plugins", "cfft"))

    def run(self, image_path, width, height, block, value):
        return self.octave.CFFT(image_path, width, height, block, value)


if __name__ == '__main__':
    comparator = Comparator(os.path.join(os.getcwd(), "plugins", "cfft", "dataset"), 50, 0.01)
    comparator.list_images()
    comparator.load_images()

    cfft = CFFT_algorithm()
    # cdct = CDCT_algorithm()
    comparator.add_algo(cfft)
    # comparator.add_algo(cdct)
    comparator.run()
