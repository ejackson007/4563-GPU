#cuda is failing because of pillow. Need to research either other image library, or have to pass back rgb values and
#print in serial. Can potentially use OpenCL
#https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480
from numba import cuda
from numba import *
from PIL import Image 
import time
import numpy as np
import math
# drawing area 
xa = -2
xb = 1
ya = -1
yb = 1
  
# max iterations allowed for color (256 max rgb)
maxIt = 255 
# image size 
imgx = 1536 # width
imgy = 1024 # height
area = imgx * imgy


@cuda.jit
def mandel():
    i = cuda.grid(1) #equivalent to threadIdx.x + blockIDx.x * blockDim.x
    if(i < area):
        #instead of for loop to iterate x and y values, we need to use the i to calculate which pixel the 
        #thread should calculate the mandelbrot series on
        x = math.floor(i / imgy)
        y = i % imgy
        zy = y * (yb - ya) / (imgy - 1)  + ya
        zx = x * (xb - xa) / (imgx - 1)  + xa 
        z = zx + zy * 1j
        c = z 
        for j in range(maxIt): 
            if abs(z) > 2.0: break
            z = z * z + c 
        return [x,y,j % 4 * 64, j % 8 * 32, j % 16 * 16]


from_cuda = []
image = Image.new("RGB", (imgx, imgy)) #making image global should let me write to it?
#find how many blocks are needed at 1024 per block. round up to nearest integer
#the kernel will handle a bounds check with an if statement
griddim = math.ceil(area/1024), 1 
blockdim = 1024,1 # max out threads

startTime = time.time();
from_cuda = mandel[griddim,blockdim]()
image.putpixel(( from_cuda[0],from_cuda[1] ), ( from_cuda[2],from_cuda[3],from_cuda[4] ))
endTime = time.time();
image.save('mandelSerial.png', "PNG")
print(f"Execution time for {imgx}x{imgy} image was {endTime-startTime} seconds")

