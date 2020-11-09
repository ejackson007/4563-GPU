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

__global__
void mandel(pixel * image){
    //get thread id
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < area){
        double x = i / HEIGHT;
        double y = i % int(HEIGHT); //mod of floating point
        int j;
        complex z, c;
        z.imag = y * (yb - ya) / (HEIGHT - 1) + ya;
        z.real = x * (xb - xa) / (WIDTH - 1) + xa;
        c = z;
        for(j = 0; j < maxIt; j++){
            if(absComplex(z) > 2.0){
                break;
            }
            z.real = z.real*z.real + c.real;
            z.imag = z.imag*z.imag + c.imag;
        }
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
    printf("%d %d\n", WIDTH, HEIGHT);
    for(int i=0; i < area; i++){
        printf("%d %d %d\n", image[i].r, image[i].g, image[i].b);
    }
    return 0;
}
