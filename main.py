import math
import os
import pickle
from datetime import datetime
import pywt
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

# from oct2py import Oct2Py

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
            data = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
            data = clahe.apply(data)
            self.images_original_data.append(data)
            self.save_processed_image(self.dataset_path, 'BASE', image_name.split('.')[0], data)
        # run algorithms with different specified parameters
        for algorithm in self.algorithms:
            print(algorithm.algorithm_name, algorithm.block, algorithm.compression)
            for i, image_name in enumerate(self.images_paths):
                start_time = datetime.now()
                processed_image, cnt = algorithm.run(os.path.join(self.dataset_path, image_name),
                                                     self.images_original[i].height,
                                                     self.images_original[i].width)
                working_time = datetime.now() - start_time

                processed_image = np.clip(processed_image, 0.0, 255.0).astype("uint8")
                processed_image = clahe.apply(processed_image)
                psnr = self.compareProcessed(i, processed_image)

                algorithm.psnrs.append(psnr)
                algorithm.times.append(working_time)
                algorithm.cnts.append(cnt)

                algorithm.save_processed_image(self.dataset_path, image_name.split('.')[0], processed_image)
                print(image_name, psnr, cnt, working_time)
            algorithm.calcMeanPSNRS()
            algorithm.calcMeanCNT()
            print("PSNR mean:", algorithm.psnr, "CNT mean:", algorithm.cnt)

    def save_results(self):
        results = list()
        # results.append(self.images_paths)
        for algorithm in self.algorithms:
            results.append(algorithm.psnrs)
        df = pd.DataFrame(results, columns=self.images_paths,
                          index=[algorithm.algorithm_name for algorithm in self.algorithms])
        df.to_excel(os.path.join(self.dataset_path, 'results', 'report.xlsx'))

    def compareProcessed(self, i, processed_image):
        mse = np.mean((self.images_original_data[i] - processed_image) ** 2)/(self.images_original[i].height*self.images_original[i].width)
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
        self.cnts = list()
        self.psnr = 0
        self.cnt = 0

    def save_processed_image(self, path, name, image):
        image = Image.fromarray(image)
        image_name = self.algorithm_name + '_' + name + ".png"
        save_path = os.path.join(path, "results", self.algorithm_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image.save(os.path.join(save_path, image_name))

    def calcMeanPSNRS(self):
        for i, value in enumerate(self.psnrs):
            self.psnr += value / (len(self.psnrs))

    def calcMeanCNT(self):
        for i, value in enumerate(self.cnts):
            self.cnt += value / (len(self.cnts))


class OctaveAlgorithm(Algorithm):
    def __init__(self, name, path):
        super().__init__(name)
        self.octave = Oct2Py()
        self.octave.eval("pkg load image")
        self.octave.eval("pkg load signal")
        # self.octave.eval("pkg load cvx") # TODO: check this
        self.octave.addpath(path)


class CompressedSensing(OctaveAlgorithm):
    def __init__(self, name, block, compression):
        super().__init__(name, os.path.join(os.getcwd(), "plugins", "compressed_sensing"))
        self.block = block
        self.compression = compression


class JPEG_algorithm(Algorithm):
    def __init__(self, compression):
        super().__init__("JPEG")
        self.compression = compression

    def run(self, image_path, weight, height):
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), int(100 * self.compression)]
        img = cv.imread(image_path, 0)
        _, enc = cv.imencode('.jpg', img, encode_param)
        decimg = cv.imdecode(enc, 1)
        return cv.cvtColor(decimg, cv.COLOR_BGR2GRAY)


class CFFT_algorithm(CompressedSensing):
    def __init__(self, block, compression):
        super().__init__("CFFT", block, compression)

    def run(self, image_path, width, height):
        return self.octave.CFFT(image_path, width, height, self.block, self.compression)


class CDCT_algorithm(CompressedSensing):
    def __init__(self, block, compression):
        super().__init__("CDCT", block, compression)

    def run(self, image_path, width, height):
        return self.octave.CDCT(image_path, width, height, self.block, self.compression)


class L1_algorithm(CompressedSensing):
    def __init__(self, block, compression):
        super().__init__("L1", block, compression)

    def run(self, image_path, width, height):
        return self.octave.L1(image_path, width, height, self.block, self.compression)


class Utils(Algorithm):
    def __init__(self, block, compression, name):
        super().__init__(name)
        self.block = block
        self.compression = compression
        self.windowSize = block

    def loadImg(self, path, cosTiling=True):
        image = cv.imread(path)
        image = np.asarray(image).astype("uint8")
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        size = 0
        size2 = self.windowSize
        if cosTiling:
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
        # print(pywt.wavelist(kind='discrete'))
        # dcttiles = np.zeros([len(tiles), wsize, wsize])
        dctList = []
        for i in range(len(tiles)):
            imf = np.float32(tiles[i]) / 255.0  # float conversion/scale
            output = pywt.wavedec2(imf, 'haar')
            dctList.append(output)
        return dctList

    def getIDWT(self, tiles, tileSize):
        idcttiles = np.zeros([len(tiles), tileSize, tileSize])
        for i in range(len(tiles)):
            inverse = pywt.waverec2(tiles[i], 'haar') * 255.0
            idcttiles[i] = inverse
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

    def weight(self, x, size):
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * (0.5 * (x + 0.5) / size))

    def getTile(self, tile, x, y):
        xn = np.clip(x, 0, len(tile))
        yn = np.clip(y, 0, len(tile[0]))
        return tile[xn, yn]

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

    def output2d(self,arr, size):
        x, y = size
        window = len(arr[0, 0])
        dx = window - (x) % window
        dy = window - (y) % window
        dx0 = dx // 2
        dy0 = dy // 2
        xo = x + dx + (dx + 1) // 2
        yo = y + dy + (dy + 1) // 2
        output = np.zeros([xo, yo], dtype=arr.dtype)
        arr = arr.reshape(xo // window, yo // window, window, window)
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                output[i * window:i * window + window, j * window:j * window + window] = arr[i, j]
        return output[dx0:x + dx0, dy0:y + dy0]

    def CompressWeights0(self, data, compression, tile):
        self.windowSize = tile
        cnt = 0
        cnt2 = 0
        dct = self.getDCT(data)
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    cnt2 += 1
                    if np.abs(dct[i, j, k]) < compression:
                        dct[i, j, k] = 0.0
                        cnt += 1
        return dct, cnt / cnt2

    def CompressWeights1(self, data, compression, tile):
        self.windowSize = tile
        cnt = 0
        dct = self.getFFT(data)
        cnt2 = 0
        for i in range(len(dct)):
            for j in range(len(dct[i])):
                for k in range(len(dct[i, j])):
                    cnt2 += 1
                    if np.sqrt(dct[i, j, k, 0] * dct[i, j, k, 0] + dct[i, j, k, 1] * dct[i, j, k, 1]) < \
                            compression * len(dct[i]):
                        dct[i, j, k, 0] = 0.0
                        dct[i, j, k, 1] = 0.0
                        cnt += 1
        return dct, cnt / cnt2

    def CompressWeights2(self, data, compression, tile):
        self.windowSize = tile
        cnt = 0
        cnt2 = 0
        dct = self.getDWT(data)
        for i in range(len(dct)):
            for j in range(1, len(dct[i])):
                for k in range(len(dct[i][j])):
                    for k2 in range(len(dct[i][j][k])):
                        for k3 in range(len(dct[i][j][k][k2])):
                            cnt2 += 1
                            if np.abs(dct[i][j][k][k2][k3]) < compression:
                                dct[i][j][k][k2][k3] = 0.0
                                cnt += 1
        return dct, cnt / (cnt2 + len(dct))

    def UnCompressWeights01(self, data):
        return self.getIDCT(data)

    def UnCompressWeights11(self, data):
        return self.getIFFT(data)


class DCT_CosineWindow(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "DCT_CosineWindow")

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, True)
        # print(image_data.shape)
        compress, cnt = self.CompressWeights0(image_data, self.compression, self.block)
        inverse = self.UnCompressWeights01(compress)
        result = self.cosineWindow(inverse, (width, height))
        return result, cnt
class DCT_Trivial(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "DCT_Trivial")

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, False)
        # print(image_data.shape)
        compress, cnt = self.CompressWeights0(image_data, self.compression, self.block)
        inverse = self.UnCompressWeights01(compress)
        result = self.output2d(inverse, (width, height))
        return result, cnt

class FFT_CosineWindow(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "FFT_CosineWindow")
        self.tileSize = block

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, True)
        self.windowSize = self.tileSize
        compress, cnt = self.CompressWeights1(image_data, self.compression, self.block)
        inverse = self.getIFFT(compress)
        result = self.cosineWindow(inverse, (width, height))
        return result, cnt
class FFT_Trivial(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "FFT_Trivial")
        self.tileSize = block

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, False)
        self.windowSize = self.tileSize
        compress, cnt = self.CompressWeights1(image_data, self.compression, self.block)
        inverse = self.getIFFT(compress)
        result = self.output2d(inverse, (width, height))
        return result, cnt


class DWT_CosineWindow(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "DWT_CosineWindow")

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, True)
        compress, cnt = self.CompressWeights2(image_data, self.compression, self.block)
        inverse = self.getIDWT(compress, self.block)
        result = self.cosineWindow(inverse, (width, height))
        return result, cnt
class DWT_Trivial(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "DWT_Trivial")

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, False)
        compress, cnt = self.CompressWeights2(image_data, self.compression, self.block)
        inverse = self.getIDWT(compress, self.block)
        result = self.output2d(inverse, (width, height))
        return result, cnt
class DWT_NonTiled(Utils):
    def __init__(self, block, compression):
        super().__init__(block, compression, "DWT_Trivial")

    def run(self, image_path, width, height):
        image_data, original_data = self.loadImg(image_path, False)
        coeffs = pywt.dwt2(image_data, 'haar')
        cnt = 0
        for i in range(len(coeffs)):
            if (np.abs(coeffs[i]) > self.compression):
                coeffs[i] = 0.0
                cnt += 1
        result = pywt.idwt2(coeffs, 'haar')
        return result, cnt

def calc():
    comparator = Comparator(os.path.join(os.getcwd(), "dataset/kodim"))
    comparator.list_images()
    comparator.load_images()
    #for tile in range(8, 32, 2):
    #    for compress in np.arange(0, 0.5, 0.1):
    #        algo = DCT_CosineWindow(tile, compress)
    #        comparator.add_algo(algo)

    # jpeg = JPEG_algorithm(0.8)
    comparator.add_algo(FFT_Trivial(30, 0.4))
    # cdct = CDCT_algorithm(8, 0.8)
    # l1 = L1_algorithm(8, 0.01)
    # fftcw = FFT_CosineWindow(8, 0.8)
    # dwtcw = DWT_CosineWindow(8, 0.8)

    # comparator.add_algo(jpeg)
    # comparator.add_algo(cfft)
    # comparator.add_algo(cdct)
    # comparator.add_algo(l1)

    # comparator.add_algo(dctcw)
    # comparator.add_algo(fftcw)
    # comparator.add_algo(dwtcw)
    comparator.run()
    comparator.save_results()

    with open(os.path.join(os.getcwd(), "dataset/results/comparator.pk"), 'wb') as f:
        pickle.dump(comparator, f)


def toimg(algoSequence):
    maxPsnr = 0.0
    maxTime = 0.0
    maxEfficiency = 0.0
    cnt = 0
    x2 = 0
    y = 0
    output = []

    for tile in range(8, 32, 2):
        outputx = []
        for compress in np.arange(0, 0.5, 0.1):
            algo = algoSequence[cnt]
            efficiency = algo.psnr * algo.cnt
            maxEfficiency = np.maximum(efficiency, maxEfficiency)
            avrTime = 0.0
            for time in algo.times:
                avrTime += time.total_seconds()
            avrTime /= len(algo.times)
            outputx.append([algo.psnr, algo.cnt, avrTime])
            maxPsnr = np.maximum(algo.psnr, maxPsnr)
            maxTime = np.maximum(avrTime, maxTime)
            cnt += 1
        output.append(outputx)
    x2 = 0
    y = 0
    outputNp = np.zeros([len(output), len(output[0]), 3], dtype=float)
    for tile in range(8, 32, 2):
        x2 = 0
        for compress in np.arange(0, 0.5, 0.1):
            efficiency = output[y][x2][0] * output[y][x2][1]
            output[y][x2][0] /= maxPsnr
            output[y][x2][2] = 1.0 - output[y][x2][2] / maxTime

            if np.abs(efficiency - maxEfficiency) < 0.2:
                output[y][x2][2] = 1.0
                output[y][x2][1] = 1.0
                output[y][x2][0] = 1.0
                print('Perfect parameters: compression:', compress, 'tile:', tile)
            outputNp[y, x2] = output[y][x2]
            x2 += 1
        y += 1
    fig, ax = plt.subplots()
    ax.imshow(outputNp, extent=[0, 0.4, 32, 8])
    ax.set_aspect(0.05)
    return output


if __name__ == '__main__':
    calc()
    comparator = None
    with open(os.path.join(os.getcwd(), "dataset/results/comparator.pk"), 'rb') as f:
        comparator = pickle.load(f)

    out = toimg(comparator.algorithms)
    plt.show()
