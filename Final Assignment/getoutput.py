# drawing area 
xa = -2
xb = 1
ya = -1
yb = 1
  
# max iterations allowed for color (256 max rgb)
maxIt = 255 
  
# image size 
imgx = 1920
imgy = 1080

for y in range(imgy): 
    zy = y * (yb - ya) / (imgy - 1)  + ya 
    for x in range(imgx): 
        zx = x * (xb - xa) / (imgx - 1)  + xa 
        z = zx + zy * 1j
        c = z 
        print(f"z = {z}")
        for i in range(maxIt): 
            if abs(z) > 2.0: break
            z = z * z + c 
        print(f"j is {i}")