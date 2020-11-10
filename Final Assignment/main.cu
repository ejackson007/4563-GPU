// caculate all the rgb values
// calculating wrong image, but is creating a image!
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 1920.0
#define HEIGHT 1080.0
#define area WIDTH * HEIGHT
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

__device__ double absComplex(complex z){
    return sqrt(z.real*z.real + z.imag*z.imag);
}

__device__ complex sqComplex(complex z){
    complex result;
    result.real = z.real*z.real - z.imag*z.imag;
    result.imag = z.real*z.imag + z.imag*z.real;
    return result;
}

__device__ complex addComplex(complex z, complex c){
    complex result;
    result.real = z.real + c.real;
    result.imag = z.imag + c.imag;
    return result;
}

__global__
void mandel(pixel * image){
    //get thread id
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    double zy, zx;
    if(i < area){
        double x = i / HEIGHT;
        double y = i % int(HEIGHT); //mod of floating point
        int j;
        complex z, c;
        zy = y * (yb - ya) / (HEIGHT - 1) + ya;
        zx = x * (xb - xa) / (WIDTH - 1) + xa;
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
        image[i].r = j % 4 * 64;
        image[i].g = j % 8 * 32;
        image[i].b = j % 16 * 16;
    }
}

int main(){
    int imageSize = sizeof(struct rgb) * area;
    pixel *image = (pixel *)malloc(imageSize), *imageDevice;
    
    cudaMalloc(&imageDevice, imageSize);

    dim3 gridDim(area/1024, 1, 1);
    dim3 dimBlock(1024,1,1);

    mandel<<<gridDim, dimBlock>>>(imageDevice);

    cudaMemcpy(image, imageDevice, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(imageDevice);
    printf("%d %d\n", (int)HEIGHT, (int)WIDTH);
    for(int i=0; i < area; i++){
        printf("%d %d %d\n", image[i].r, image[i].g, image[i].b);
    }
    return 0;
}
