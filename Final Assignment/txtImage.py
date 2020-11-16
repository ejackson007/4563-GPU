from PIL import Image
import time
import math

startTime = time.time();
with open('mandelRGBcuda', 'r') as f:
    width, height = map(int, f.readline().split())
    image = Image.new("RGB", (width, height))
    for i in range(width*height):
        x = math.floor(i / height)
        y = i % height
        r,g,b = map(int, f.readline().split())
        image.putpixel((x,y), (r,g,b))
endTime = time.time();
#image = image.rotate(90, expand=True)
image.save('mandelRead2.png', "PNG")
print(f"Execution time for {width}x{height} image was {endTime-startTime} seconds")