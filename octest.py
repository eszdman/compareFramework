import os
from oct2py import octave

CFFT_PATH = os.path.join(os.getcwd(), "plugins", "cfft")

octave.eval("pkg load image")
octave.addpath(CFFT_PATH)
result = octave.CFFT(os.path.join(CFFT_PATH, "dataset", "barbara.tif"), 256, 256, 16, 100)
print(result)
