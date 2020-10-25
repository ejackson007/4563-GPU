# Mandelbrot fractal 
# FB - 201003254 
from PIL import Image 
import time
  
# drawing area 
xa = -2.0
xb = 1.0
ya = -1.5
yb = 1.5
  
# max iterations allowed for color (256 max rgb)
maxIt = 255 
  
# image size 
imgx = 1920
imgy = 1920
image = Image.new("RGB", (imgx, imgy)) 
startTime = time.time();

for y in range(imgy): 
    zy = y * (yb - ya) / (imgy - 1)  + ya 
    for x in range(imgx): 
        zx = x * (xb - xa) / (imgx - 1)  + xa 
        z = zx + zy * 1j
        c = z 
        for i in range(maxIt): 
            if abs(z) > 2.0: break
            z = z * z + c 
        image.putpixel((x, y), (i % 4 * 64, i % 8 * 32, i % 16 * 16)) 

endTime = time.time();
image.show() 
print(f"Execution time for {imgx}x{imgy} image was {endTime-startTime} seconds")