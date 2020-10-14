//Evan Jackson
//Cooley Tukey
//Joseph Williamson

#include <stdio.h>
#include <math.h>

#define SIZE 8192
typedef struct complex_t {
    double real;
    double imag;
} complex;

__global__
void oddMultCalc(complex * oddDevice, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    complex result, polar;
    if(i < n/2){
        polar.real = cos(-2.0*M_PI*(i/n));
        polar.imag = sin(-2.0*M_PI*(i/n));

        result.real = polar.real*oddDevice[i].real - polar.imag*oddDevice[i].imag;
        result.imag = polar.real*oddDevice[i].imag + polar.imag*oddDevice[i].real;

        oddDevice[i] = result;
    }
}

__global__
void addOddEven(complex * oddDevice, complex * evenDevice, complex * XDevice, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n/2){
        XDevice[i].real = evenDevice[i].real + oddDevice[i].real;
        XDevice[i].imag = evenDevice[i].imag + oddDevice[i].imag;
        XDevice[i + n/2].real = evenDevice[i].real - oddDevice[i].real;
        XDevice[i + n/2].imag = evenDevice[i].imag - oddDevice[i].imag;
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
    complex *X = (complex *)malloc(sizeof(struct complex_t) * n);
    complex *odd, *even, *ODD, *EVEN, *XDevice, *oddDevice, *evenDevice;
    

    if (n == 1){
        X[0] = table[0];
        return X;
    }

    even = (complex *)malloc(sizeof(struct complex_t) * n/2);
    odd = (complex *)malloc(sizeof(struct complex_t) * n/2);
    //assing odd and even values
    for(int i = 0; i < n/2; i++){
        even[i] = table[2*i];
        odd[i] = table[2*i + 1];
    }

    EVEN = CT_FFT(even, n/2);
    ODD = CT_FFT(odd, n/2);

    cudaMalloc(&oddDevice, n/2);
    cudaMalloc(&evenDevice, n/2);
    cudaMalloc(&XDevice, n);
    cudaMemcpy(oddDevice, ODD, n/2, cudaMemcpyHostToDevice);
    cudaMemcpy(evenDevice, EVEN, n/2, cudaMemcpyHostToDevice);
    cudaMemcpy(XDevice, X, SIZE, cudaMemcpyHostToDevice);

    dim3 dimGrid(4,1,1);
    dim3 dimBlock(1024,1,1);

    oddMultCalc<<<dimGrid, dimBlock>>>(oddDevice, n);
    cudaDeviceSynchronize();
    addOddEven<<<dimGrid, dimBlock>>>(oddDevice, evenDevice, XDevice, n);
    cudaDeviceSynchronize();

    cudaMemcpy(X, XDevice, SIZE, cudaMemcpyDeviceToHost);
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
        printf("XR[%d]: %f  XI[%d]: %f i\n", i, printing[i].real, i, printing[i].imag);
        printf("=====================================\n");
    }
    return 0;
}