from numba import cuda
from numba import *
from PIL import Image 
import time
import numpy as np

@cuda.jit
def mandel(x,y,maxIt):
    c = complex(x,y);
    z = 0.0j #j is built in imaginary type
    for i in range(maxIt):
        z = z*z + c
        if(z.real ** 2 + z.imag ** 2) >= 4:
            return i
    return maxIt

@cuda.jit
def kernel(xa, xb, ya, yb, image, maxIt):
    height = image.shape[0]
    width = image.shape[1]

    zx = (xb - xa) / width
    zy = (yb - ya) / height

    
