// caculate all the rgb values
// calculating wrong image, but is creating a image!
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 1920.0
#define HEIGHT 1080.0
#define xa -2.0
#define xb 1.0
#define ya -1.0
#define yb 1.0
#define maxIt 255

typedef struct rgb{
    int r;
    int g;
    int b;
} pixel;
typedef struct complex_t {
    double real;
    double imag;
} complex;

double absComplex(complex z){
    return sqrt(z.real*z.real + z.imag*z.imag);
}

complex sqComplex(complex z){
    complex result;
    result.real = z.real*z.real - z.imag*z.imag;
    result.imag = z.real*z.imag + z.imag*z.real;
    return result;
}

complex addComplex(complex z, complex c){
    complex result;
    result.real = z.real + c.real;
    result.imag = z.imag + c.imag;
    return result;
}

void mandel(pixel * image){
    //get thread id
    int j, pos = 0;
    double zy, zx;
    complex z, c;
    for(int y = 0; y < (int)HEIGHT; y++){
        zy = y * (yb - ya) / (HEIGHT - 1.0) + ya;
        for(int x = 0; x < (int)WIDTH; x++){
            zx = x * (xb - xa) / (WIDTH - 1.0) + xa;
            //reset z
            z.real = zx;
            z.imag = zy;
            c = z;
            //printf("z = %0.16f %0.16f\n", z.real, z.imag);
            for(j = 0; j < maxIt; j++){
                if(absComplex(z) > 2){
                    break;
                }
                z = addComplex(sqComplex(z), c);
            }
            //printf("zOut = %0.16f %0.16f\n", z.real, z.imag);
            //printf("j = %d\n", j);
            image[pos].r = j % 4 * 64;
            image[pos].g = j % 8 * 32;
            image[pos].b = j % 16 * 16;
            pos++;
        }
    }
}

int main(){
    int area = WIDTH * HEIGHT;
    int imageSize = sizeof(struct rgb) * area;
    pixel *image = (pixel *)malloc(imageSize);

    mandel(image);

    printf("%d %d\n", (int)WIDTH, (int)HEIGHT);
    for(int i=0; i < area; i++){
        printf("%d %d %d\n", image[i].r, image[i].g, image[i].b);
    }
    return 0;
}
