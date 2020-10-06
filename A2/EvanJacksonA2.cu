//Evan Jackson
//Cooley-Tukey FFT Cuda
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#define SIZE 8192
#define JUMP 4096

double PI = 22/7;
typedef double complex cplx;
dim3 dimGrid(4,1,1);
dim3 dimBlock(1024,1,1);

//creates array of double complex and returns
//them. It fills the array with the values at the top
//and then fills the remaining with 0s. returns the array
//created
cplx *fillArray(num_elements){
    cplx *all_nums = (cplx *)malloc(sizeof(cplx) * num_elements);
    assert(all_nums != NULL);
    all_nums[0] = 3.6 + 2.6*I;
    all_nums[1] = 2.9 + 6.3*I;
    all_nums[2] = 5.6 + 4*I;
    all_nums[3] = 4.8 + 9.1*I;
    all_nums[4] = 3.3 + 0.4*I;
    all_nums[5] = 5.9 + 4.8*I;
    all_nums[6] = 5 + 2.6*I;
    all_nums[7] = 4.3 + 4.1*I;
    for(int i = 8; i < num_elements; i++)
        all_nums[i] = 0;
    return all_nums;
}

//performs FFT on array of complex doubles. creates a copy thats needed for recursion
void FastFourierTransform(cplx * cdBuf)
{
    cplx *cdOut = (cplx *)malloc(sizeof(cplx) * SIZE);
    
    recursiveFFT(cdBuf, cdOut, SIZE, 1);
    return cdBuf;
}

void recursiveFFT(cplx * cdBuf, cplx * cdOut, int iteration)
{
    if (iteration < SIZE) {
        recursiveFFT(cdOut, cdBuf, iteration * 2);
        recursiveFFT(cdOut + iteration, cdBuf + iteration, iteration * 2);
        
        //change this to an if in a kernel
        for (int i = 0; i < n; i += 2 * iteration) {
            double complex cdExponent = cexp(-I * PI * i / n) * cdOut[i + iteration];
            cdBuf[i / 2]     = cdOut[i] + cdExponent;
            cdBuf[(i + n)/2] = cdOut[i] - cdExponent;
        }
    }
}

__global__
void addFFT(int iteration){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int iJump = i + JUMP; // get location of thread
    //add function to check current i if it would fit with the current iteration
}

int main(){
    size_t arraySize = SIZE * sizeof(cplx);
    //create Arrays
    cplx *cdBufHost = fillArray(SIZE);
    cplx *cdOutHost = fillArray(SIZE);
    cplx *cdBufDevice;
    cplx *cdOutDevice;

    cudaMalloc(&cdBufDevice, arraySize);
    cudaMalloc(&cdOutDevice, arraySize);
    cudaMemcpy(cdBufDevice, cdBufHost, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(cdOutDevice, cdOutHost, arraySize, cudaMemcpyHostToDevice);
}