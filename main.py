import math
import os
from datetime import datetime

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from oct2py import Oct2Py

PIXEL_MAX = 255.0


class Comparator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.algorithms = list()
        self.images_paths = list()
        self.images_original = list()
        self.images_original_data = list()

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
        # prepare base images
        for i, image_name in enumerate(self.images_paths):
            data = np.asarray(self.images_original[i]).astype("uint8")
            data = clahe.apply(data)
            self.images_original_data.append(data)
            self.save_processed_image(self.dataset_path, 'BASE', image_name.split('.')[0], data)
        # run algorithms with different specified parameters
        for algorithm in self.algorithms:
            print(algorithm.algorithm_name)
            for i, image_name in enumerate(self.images_paths):
                start_time = datetime.now()
                processed_image = algorithm.run(os.path.join(self.dataset_path, image_name),
                                                self.images_original[i].width,
                                                self.images_original[i].height)
                working_time = datetime.now() - start_time
                processed_image = np.clip(processed_image, 0.0, 255.0).astype("uint8")
                processed_image = clahe.apply(processed_image)
                psnr = self.compareProcessed(i, processed_image)
                algorithm.psnrs.append(psnr)
                algorithm.times.append(working_time)
                algorithm.save_processed_image(self.dataset_path, image_name.split('.')[0], processed_image)
                print(image_name, psnr, working_time)

    def save_results(self):
        results = list()
        # results.append(self.images_paths)
        for algorithm in self.algorithms:
            results.append(algorithm.psnrs)
        df = pd.DataFrame(results, columns=self.images_paths,
                          index=[algorithm.algorithm_name for algorithm in self.algorithms])
        df.to_excel(os.path.join(self.dataset_path, 'results', 'report.xlsx'))

    def compareProcessed(self, i, processed_image):
        mse = np.mean((self.images_original_data[i] - processed_image) ** 2)
        if mse == 0:
            return 100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def save_processed_image(self, path, prefix, name, image):
        image = Image.fromarray(image)
        image_name = prefix + '_' + name + ".png"
        save_path = os.path.join(path, "results", prefix)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image.save(os.path.join(save_path, image_name))


class Algorithm:
    def __init__(self, name):
        self.algorithm_name = name
        self.psnrs = list()
        self.times = list()

    def save_processed_image(self, path, name, image):
        image = Image.fromarray(image)
        image_name = self.algorithm_name + '_' + name + ".png"
        save_path = os.path.join(path, "results", self.algorithm_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image.save(os.path.join(save_path, image_name))


class OctaveAlgorithm(Algorithm):
    def __init__(self, name, path):
        super().__init__(name)
        self.octave = Oct2Py()
        self.octave.eval("pkg load image")
        self.octave.eval("pkg load signal")
        # self.octave.eval("pkg load cvx") # TODO: check this
        self.octave.addpath(path)


class CompressedSensing(OctaveAlgorithm):
    def __init__(self, name, block, value):
        super().__init__(name, os.path.join(os.getcwd(), "plugins", "compressed_sensing"))
        self.block = block
        self.value = value


class JPEG_algorithm(Algorithm):
    def __init__(self, value):
        super().__init__("JPEG")
        self.value = value

    def run(self, image_path, _, _1):
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), int(100 * self.value)]
        img = cv.imread(image_path, 0)
        _, enc = cv.imencode('.jpg', img, encode_param)
        decimg = cv.imdecode(enc, 1)
        return cv.cvtColor(decimg, cv.COLOR_BGR2GRAY)


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


class L1_algorithm(CompressedSensing):
    def __init__(self, block, value):
        super().__init__("L1", block, value)

    def run(self, image_path, width, height):
        return self.octave.L1(image_path, width, height, self.block, self.value)


class Utils(Algorithm):
    def __init__(self, block, value, name):
        super().__init__(name)
        self.block = block
        self.value = value

    def loadImg(self, path, cosTiling=True):
        image = cv.imread(path)
        image = np.asarray(image).astype("uint8")
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        size = 0
        size2 = self.windowSize
        if (cosTiling):
            size = self.windowSize // 2
            size2 = self.windowSize // 2
        slices = self.rwindow(image, size2, size)
        slices = slices.reshape(len(slices) * len(slices[0]), self.windowSize, self.windowSize)
        return slices, image

    def rwindow(self, arr, size, oversize=0):
        dx = (size + oversize) - len(arr) % (size + oversize)
        dy = (size + oversize) - len(arr[0]) % (size + oversize)
        arr = np.pad(arr, ((dx // 2, (dx + 1) // 2), (dy // 2, (dy + 1) // 2)), 'symmetric')
        wx = len(arr) // size
        wy = len(arr[0]) // size
        output = np.zeros([wx, wy, size + oversize, size + oversize], dtype=arr.dtype)
        for i in range(wx - 1):
            for j in range(wy - 1):
                iw = i * size
                jw = j * size
                inp = arr[iw:(iw + size + oversize), jw:(jw + size + oversize)]
                output[i, j] = inp
        return output

    def getDCT(self, tiles):
        wsize = len(tiles[0])
        dcttiles = np.zeros([len(tiles), wsize, wsize])
        for i in range(len(tiles)):
            imf = np.float32(tiles[i]) / 255.0  # float conversion/scale
            dcttiles[i] = cv.dct(imf)
        return dcttiles

    def getIDCT(self, tiles):
        wsize = len(tiles[0])
        idcttiles = np.zeros([len(tiles), wsize, wsize])
        for i in range(len(tiles)):
            imf = np.float32(tiles[i])
            idcttiles[i] = cv.idct(imf) * 255.0
        return idcttiles

    def getDWT(self, tiles):
        wsize = len(tiles[0])
        dcttiles = np.zeros([len(tiles), wsize, wsize])
        for i in range(len(tiles)):
            imf = np.float32(tiles[i]) / 255.0  # float conversion/scale
            dcttiles[i] = pywt.dwt2(imf, 'haar')
        return dcttiles

    def getIDWT(self, tiles):
        wsize = len(tiles[0])
        idcttiles = np.zeros([len(tiles), wsize, wsize])
        for i in range(len(tiles)):
            imf = np.float32(tiles[i])
            idcttiles[i] = pywt.idwt2(imf) * 255.0
        return idcttiles

    def getFFT(self, tiles):
        wsize = len(tiles[0])
        ffttiles = np.zeros([len(tiles), wsize, wsize, 2])
        for i in range(len(tiles)):
            imf = np.float32(tiles[i]) / 255.0  # float conversion/scale
            fft2 = np.fft.fft2(imf)
            ffttiles[i] = np.stack([np.real(fft2), np.imag(fft2)], axis=-1)
        return ffttiles

    def getIFFT(self, tiles):
        wsize = len(tiles[0])
        fftitiles = np.zeros([len(tiles), wsize, wsize])
        for i in range(len(tiles)):
            tile = tiles[i]
            tile = np.vectorize(complex)(tile[..., 0], tile[..., 1])
            fft2 = np.fft.ifft2(tile)
            fftitiles[i] = np.real(fft2) * 255.0
        return fftitiles

    def CompressWeights0(self, data, compression, tile):
        self.windowSize = tile
        cnt = 0
        dct = self.getDCT(data)
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    if np.abs(dct[i, j, k]) > compression:
                        dct[i, j, k] = 0.0
                        cnt += 1
        return dct, cnt

    def cosineWindow(self, arr, size):
        x, y = size
        window = self.windowSize // 2
        weightv = np.array([self.weight(idx, window) for idx in range(window * 2)])
        dx = self.windowSize - x % self.windowSize
        dy = self.windowSize - y % self.windowSize
        w00 = np.zeros([self.windowSize, self.windowSize])
        w01 = np.zeros([self.windowSize, self.windowSize])
        w10 = np.zeros([self.windowSize, self.windowSize])
        w11 = np.zeros([self.windowSize, self.windowSize])
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                x0 = i % window + window
                x1 = i % window
                y0 = j % window + window
                y1 = j % window
                w00[x0][y0] = weightv[x0] * weightv[y0]
                w01[x1][y0] = weightv[x1] * weightv[y0]
                w10[x0][y1] = weightv[x0] * weightv[y1]
                w11[x1][y1] = weightv[x1] * weightv[y1]
        dx0 = dx // 2
        dy0 = dy // 2
        xo = x + dx
        yo = y + dy
        output = np.zeros([xo, yo], dtype=arr.dtype)
        arr = arr.reshape(xo // window, yo // window, self.windowSize, self.windowSize)
        for i in range(xo // window):
            for j in range(yo // window):
                x0 = window + window
                x1 = window
                y0 = window + window
                y1 = window
                v00 = self.getTile(arr, i - 1, j - 1)[x1:x0, y1:y0]
                v01 = self.getTile(arr, i, j - 1)[0:x1, y1:y0]
                v10 = self.getTile(arr, i - 1, j)[x1:x0, 0:y1]
                v11 = self.getTile(arr, i, j)[0:x1, 0:y1]
                W00 = w00[x1:x0, y1:y0]
                W01 = w01[0:x1, y1:y0]
                W10 = w10[x1:x0, 0:y1]
                W11 = w11[0:x1, 0:y1]
                output[i * window:(i + 1) * window, j * window:(j + 1) * window] = (
                        W00 * v00 + W01 * v01 + W10 * v10 + W11 * v11)

        return output[dx0:x + dx0, dy0:y + dy0]

    def CompressWeights0(self, data, compression, tile):
        global windowSize
        windowSize = tile
        cnt = 0
        dct = self.getDCT(data)
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    if np.abs(dct[i, j, k]) > compression:
                        dct[i, j, k] = 0.0
                        cnt += 1
        return dct, cnt

    def CompressWeights1(self, data, compression, tile):
        global windowSize
        windowSize = tile
        cnt = 0
        dct = self.getFFT(data)
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    if np.abs(dct[i, j, k, 0]) + np.abs(dct[i, j, k, 1]) > compression:
                        dct[i, j, k, 0] = 0.0
                        dct[i, j, k, 1] = 0.0
                        cnt += 1
        return dct, cnt

    def CompressWeights2(self, data, compression, tile):
        global windowSize
        windowSize = tile
        cnt = 0
        dct = self.getDWT(data)
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    if np.abs(dct[i, j, k]) > compression:
                        dct[i, j, k] = 0.0
                        cnt += 1
        return dct, cnt

    def CompressWeights1(self, data, compression, tile):
        global windowSize
        windowSize = tile
        dct = self.getDCT(data)
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    dct[i, j, k] = (((i * i + j * j)) / (len(dct[i]) ** 2 + len(dct[i, j]) ** 2) > compression) * dct[
                        i, j, k]
        return dct

    def UnCompressWeights01(self, data):
        return self.getIDCT(data)

    def UnCompressWeights11(self, data):
        return self.getIFFT(data)


class DCT_CosinineWindow(Utils):
    def __init__(self, block, value):
        super().__init__(block, value, "DCT_CosinineWindow")

    def run(self, image_path, width, height):
        image_data = self.loadImg(image_path, False)
        compress, cnt = self.CompressWeights0(image_data, self.value, self.block)
        inverse = self.UnCompressWeights01(compress)
        result = self.cosineWindow(inverse, (width, height))  # TODO: add size
        return result, cnt


class FFT_CosineWindow(Utils):
    def __init__(self, block, value):
        super().__init__(block, value, "DCT_CosinineWindow")

    def run(self, image_path, width, height):
        image_data = self.loadImg(image_path, False)
        windowSize = self.tileSize
        compress, cnt = self.CompressWeights1(image_data, self.value, self.block)
        inverse = self.getIFFT(compress)
        result = self.cosineWindow(inverse, (width, height))
        return result, cnt


class DWT_CosineWindow(Utils):
    def __init__(self, block, value):
        super().__init__(block, value, "DCT_CosinineWindow")

    def run(self, image_path, width, height):
        image_data = self.loadImg(image_path, False)
        compress, cnt = self.CompressWeights2(image_data, self.value, self.block)
        inverse = self.getIDWT(compress)
        result = self.cosineWindow(inverse, (width, height))
        return result, cnt


if __name__ == '__main__':
    comparator = Comparator(os.path.join(os.getcwd(), "dataset"))
    comparator.list_images()
    comparator.load_images()

    # jpeg = JPEG_algorithm(0.8)
    # cfft = CFFT_algorithm(50, 0.8)
    # cdct = CDCT_algorithm(8, 0.8)
    dctcw = DCT_CosinineWindow(8, 0.8)
    # l1 = L1_algorithm(8, 0.01)

    # comparator.add_algo(jpeg)
    # comparator.add_algo(cfft)
    # comparator.add_algo(cdct)
    comparator.add_algo(dctcw)
    # comparator.add_algo(l1)
    comparator.run()
    comparator.save_results()
