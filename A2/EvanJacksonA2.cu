//*****************************************************************************
// Assignment #2
// Evan Jackson and Joseph Williamson
// GPU Programming Date: (10/15/2020)
//*****************************************************************************
// This program solves a Fast Fourier Transform using the Cooley-Tukey
// Algorithm, also known as Radix-2. It does this by recursively cutting the 
// table in half creating an "Even" and "Odd" part until they are individuals
// in which case it goes back up the stack operating the function on each
// layer. The GPU is taken advantage of here by working on the arithmetic
// part of the algorithm, as it is the most time consuming part. Given
// the amount of threads, each layer arithmetic can be handled virtually 
// O(1) time. The input is hard coded in the program, and therefore needs 
// no input file. 
// 
// TO RUN PROGRAM:
//
// 1) Put EvanJacksonJosephWilliamson.cu and A2Script in the same directory
//    in maverick2
// 2) run command "sbatch A2Script" in terminal while in the same directory as
//    the files. 
// 3) results will be show in file named "mysimplejob.xxxxxx.out"
//****************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>//used for cos, sin, and PI

#define SIZE 8192
//complex.h is not included in cuda, therefore our own
//complex struct is created.
typedef struct complex_t {
    double real;
    double imag;
} complex;

/*
__device__ complex dPolar()
Parameters: double theta
This is a function that can only be called by the device. This
function carries out the e^(-2.0*M_PI*i/n) part of the FFT function.
theta == (-2.0*M_PI*i/n) where i is the thread id
returns the results of the operation. 
*/
__device__ complex dPolar(double theta){
    complex result;
    result.real = cos(theta);
    result.imag = sin(theta);
    return result;
}

/*
__device__ complex dAdd()
*/
__device__ complex dAdd(complex l, complex r){
    complex result;
    result.real = l.real + r.real;
    result.imag = l.imag + r.imag;
    return result;
}


__device__ complex dSubtract(complex l, complex r){
    complex result;
    result.real = l.real - r.real;
    result.imag = l.imag - r.imag;
    return result;
}


__device__ complex dMultiply(complex l, complex r){
    complex result;
    result.real = l.real*r.real - l.imag*r.imag;
    result.imag = l.real*r.imag + l.imag*r.real;
    return result;
}

__global__
void solveOdd(complex * oddDevice, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n/2){
        oddDevice[i] = dMultiply(dPolar(-2.0*M_PI*i/n), oddDevice[i]);
    }
}

__global__
void solveX(complex * oddDevice, complex * evenDevice, complex * XDevice, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n/2){
        XDevice[i] = dAdd(evenDevice[i], oddDevice[i]);
        XDevice[i + n/2] = dSubtract(evenDevice[i], oddDevice[i]);
    }
}

complex *fillArray(){
    complex *all_nums = (complex *)malloc(sizeof(struct complex_t) * SIZE);
    all_nums[0].real = 3.6;
    all_nums[0].imag = 2.6;
    all_nums[1].real = 2.9;
    all_nums[1].imag = 6.3;
    all_nums[2].real = 5.6;
    all_nums[2].imag = 4.0;
    all_nums[3].real = 4.8;
    all_nums[3].imag = 9.1;
    all_nums[4].real = 3.3;
    all_nums[4].imag = 0.4;
    all_nums[5].real = 5.9;
    all_nums[5].imag = 4.8;
    all_nums[6].real = 5.0;
    all_nums[6].imag = 2.6;
    all_nums[7].real = 4.3;
    all_nums[7].imag = 4.1;
    for(int i = 8; i < SIZE; i++){
        all_nums[i].real = 0;
        all_nums[i].imag = 0;
    }
    return all_nums;
}

complex *CT_FFT(complex* table, int n){
    int arraySize = sizeof(struct complex_t) * n;
    complex *X = (complex *)malloc(sizeof(struct complex_t) * n);
    complex *odd, *even, *ODD, *EVEN, *XDevice, *oddDevice, *evenDevice;

    if(n == 1){
        X[0] = table[0];
        return X;
    }

    even = (complex *)malloc(sizeof(struct complex_t) * n/2);
    odd = (complex *)malloc(sizeof(struct complex_t) * n/2);
    for(int i = 0; i < n/2; i++){
        even[i] = table[2*i];
        odd[i] = table[2*i + 1];
    }

    EVEN = CT_FFT(even, n/2);
    ODD = CT_FFT(odd, n/2);

    //start the mess
    cudaMalloc(&evenDevice, arraySize/2);
    cudaMalloc(&oddDevice, arraySize/2);
    cudaMalloc(&XDevice, arraySize);
    cudaMemcpy(evenDevice, EVEN, arraySize/2, cudaMemcpyHostToDevice);
    cudaMemcpy(oddDevice, ODD, arraySize/2, cudaMemcpyHostToDevice);

    dim3 dimGrid(4,1,1);
    dim3 dimBlock(1024,1,1);

    solveOdd<<<dimGrid, dimBlock>>>(oddDevice, n);
    cudaDeviceSynchronize();
    solveX<<<dimGrid, dimBlock>>>(oddDevice, evenDevice, XDevice, n);
    cudaDeviceSynchronize();

    cudaMemcpy(X, XDevice, arraySize, cudaMemcpyDeviceToHost);
    free(EVEN);
    free(ODD);
    cudaFree(oddDevice);
    cudaFree(evenDevice);
    cudaFree(XDevice);
    return X;
}

int main(){
    complex *table, *printing;
    table = fillArray();
    printing = CT_FFT(table, SIZE);
    printf("TOTAL PROCESSED SAMPLES: %d\n", SIZE);
    printf("=====================================\n");
    for(int i=0; i < 8; i++){
        //print real and imaginary values
        printf("XR[%d]: %f  XI[%d]: %fi\n", i, printing[i].real, i, printing[i].imag);
        printf("=====================================\n");
    }
    return 0;
}
