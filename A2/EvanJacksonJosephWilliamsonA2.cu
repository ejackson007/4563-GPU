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
// 1) Put EvanJacksonJosephWilliamsonA2.cu and A2Script in the same directory
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
Parameters: complex even, complex odd
This functino adds two complex numbers. Since cuda
doesnt support complex.h, I had to make my own complex
struct, meaning I had to make my own addition function.
The function returns the sum of the two.
*/
__device__ complex dAdd(complex even, complex odd){
    complex result;
    result.real = even.real + odd.real;
    result.imag = even.imag + odd.imag;
    return result;
}

/*
__device__ complex dSubtract()
Parameters: complex even, complex odd
This function subtracts two complex numbers. Since cuda
doesnt support complex.h, I had to make my own complex
struct, meaning I had to make my own subtraction function.
The function returns the difference between the two
*/
__device__ complex dSubtract(complex even, complex odd){
    complex result;
    result.real = even.real - odd.real;
    result.imag = even.imag - odd.imag;
    return result;
}

/*
__device__ complex dMultiply()
Parameters: complex polar, complex odd
This function multiplies two complex numbers, Specifically
the polar that was caluculated, and the number form the odd
array. Since cuda doesnt support complex.h, I had to make 
my own complex struct, meaning I had to make my own 
multiplication function.
The function returns the product of the two.
*/
__device__ complex dMultiply(complex polar, complex odd){
    complex result;
    result.real = polar.real*odd.real - polar.imag*odd.imag;
    result.imag = polar.real*odd.imag + polar.imag*odd.real;
    return result;
}

/*
__global__ solveOdd()
Parameters: complex* oddDevice, int n
This kernel handles the solving of the first half of the arithmetic needed,
namely the "twiddle factor" (even will return the same value, so it is only
done on the odd values). The multiplication and polar parts were originally
done inside the kernel, however testing with serial confirmed that doing it inside
results in the same values (I still have no idea why), therefore research was
done into documentation to allow for device specific functions. The documentation
for this can be found here:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers
This function returns nothing, as the table is passed by reference
*/
__global__
void solveOdd(complex * oddDevice, int n){
    //find thread id
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    //first half of the array needs to have twiddle factor found
    if(i < n/2){
        oddDevice[i] = dMultiply(dPolar(-2.0*M_PI*i/n), oddDevice[i]);
    }
}

/*
__global__ solveX()
Parameters: complex * oddDevice, complex * evenDevice, complex * XDevice, int n
This kernel solves for XDevice, which is the completed FFT array. After running
for the twiddle factors, the even positions are equal to the addition of the odd
and even places, while the odd positions are equal to the subtraction of the even
and odd places. Functions were used for addition and subtraction for readability
sake. 
This function returns nothing, as XDevice is passed by refrence and edited directly.
*/
__global__
void solveX(complex * oddDevice, complex * evenDevice, complex * XDevice, int n){
    //find thread id
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    //if thread id can for size of sub array
    if (i < n/2){
        //even positions
        XDevice[i] = dAdd(evenDevice[i], oddDevice[i]);
        //odd positions
        XDevice[i + n/2] = dSubtract(evenDevice[i], oddDevice[i]);
    }
}

/*
complex* fillArray()
Parameters: N/A
This function creates an array of SIZE (8192) and then fills it with the desired values
Returns filled array. 
*/
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

/*
complex *CT_FFT()
Parameters: complex* table, int n)
This function carries out the FFT algorithm recursively. Its main plan is to
continually split itself in half until it is down to length(1), in which case it will 
do the arithmetic on the tables, returning the solved sub tables, until the
full table is completed. Any aritmetic is called on for the kernel to process
Returns completed FFT table.
*/
complex *CT_FFT(complex* table, int n){
    int arraySize = sizeof(struct complex_t) * n;
    complex *X = (complex *)malloc(sizeof(struct complex_t) * n);
    complex *odd, *even, *ODD, *EVEN, *XDevice, *oddDevice, *evenDevice;

    if(n == 1){
        X[0] = table[0];
        return X;
    }

    //allocate space for sub arrays and fill
    even = (complex *)malloc(sizeof(struct complex_t) * n/2);
    odd = (complex *)malloc(sizeof(struct complex_t) * n/2);
    for(int i = 0; i < n/2; i++){
        even[i] = table[2*i];
        odd[i] = table[2*i + 1];
    }

    //run fft on sub arrays
    EVEN = CT_FFT(even, n/2);
    ODD = CT_FFT(odd, n/2);

    //allocate space for the sub array on gpu
    cudaMalloc(&evenDevice, arraySize/2);
    cudaMalloc(&oddDevice, arraySize/2);
    cudaMalloc(&XDevice, arraySize);
    //copy sub arrays to gpu
    cudaMemcpy(evenDevice, EVEN, arraySize/2, cudaMemcpyHostToDevice);
    cudaMemcpy(oddDevice, ODD, arraySize/2, cudaMemcpyHostToDevice);

    dim3 dimGrid(4,1,1);
    dim3 dimBlock(1024,1,1);

    //find the twiddle factor for odd
    solveOdd<<<dimGrid, dimBlock>>>(oddDevice, n);
    cudaDeviceSynchronize();
    //complete the FFT arithmetic
    solveX<<<dimGrid, dimBlock>>>(oddDevice, evenDevice, XDevice, n);
    cudaDeviceSynchronize();

    //copy solved fft table back to host
    cudaMemcpy(X, XDevice, arraySize, cudaMemcpyDeviceToHost);
    //free sub arrays
    free(EVEN);
    free(ODD);
    cudaFree(oddDevice);
    cudaFree(evenDevice);
    cudaFree(XDevice);
    return X;
}

/*
int main()
Parameters: N/A
This function controls the program. It creates the original table, as well as
the table to be printed, which contains the solved FFT. The function also
prints the values of the solved table.
Returns 0
*/
int main(){
    complex *table, *printing;
    //fills initial array
    table = fillArray();
    //sets printing to solved array
    printing = CT_FFT(table, SIZE);
    //print values 0-7
    printf("TOTAL PROCESSED SAMPLES: %d\n", SIZE);
    printf("=====================================\n");
    for(int i=0; i < 8; i++){
        //print real and imaginary values
        printf("XR[%d]: %f  XI[%d]: %fi\n", i, printing[i].real, i, printing[i].imag);
        printf("=====================================\n");
    }
    return 0;
}
