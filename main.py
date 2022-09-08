import os
import math
import numpy as np
from PIL import Image
from oct2py import Oct2Py
import cv2 as cv

PIXEL_MAX = 255.0


class Comparator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
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
            print(algorithm.algorithm_name)
            for i, image_name in enumerate(self.images_paths):
                processed_image = algorithm.run(os.path.join(self.dataset_path, image_name),
                                                self.images_original[i].width,
                                                self.images_original[i].height)
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
        self.algorithm_name = name
        self.psnrs = list()
        self.processed_images = list()

        self.octave = Oct2Py()
        self.octave.eval("pkg load image")
        self.octave.eval("pkg load signal")
        self.octave.addpath(path)

    def save_processed_image(self, path, name, image):
        image = Image.fromarray(image)
        image_name = self.algorithm_name + '_' + name + ".png"
        save_path = os.path.join(path, "results", self.algorithm_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image.save(os.path.join(save_path, image_name))


class CompressedSensing(Algorithm):
    def __init__(self, name, block, value):
        super().__init__(name, os.path.join(os.getcwd(), "plugins", "compressed_sensing"))
        self.block = block
        self.value = value


class CFFT_algorithm(CompressedSensing):
    def __init__(self, block, value):
        super().__init__("CFFT", block, value)

    def run(self, image_path, width, height):
        return self.octave.CFFT(image_path, width, height, self.block, self.value)


class CDCT_algorithm(CompressedSensing):
    def __init__(self, block, value):
        super().__init__("CDCT", block, value)

    def run(self, image_path, width, height):
        return self.octave.CDCT(image_path, width, height, self.block, self.value)


if __name__ == '__main__':
    comparator = Comparator(os.path.join(os.getcwd(), "dataset"))
    comparator.list_images()
    comparator.load_images()

    cfft = CFFT_algorithm(50, 0.2)
    cdct = CDCT_algorithm(8, 0.2)
    comparator.add_algo(cfft)
    comparator.add_algo(cdct)
    comparator.run()
