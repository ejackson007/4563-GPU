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

complex *fillArray(int size){
    complex *all_nums = (complex *)malloc(sizeof(struct complex_t) * size);
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
    for(int i = 8; i < size; i++){
        all_nums[i].real = 0;
        all_nums[i].imag = 0;
    }
    return all_nums;
}

complex *CT_FFT(complex* table, int n){
    complex* X = (complex*) malloc(sizeof(struct complex_t) * N);
    complex * d, * e, * D, * E;
    int k;

    if (N == 1) {
        X[0] = x[0];
        return X;
    }

    e = (complex*) malloc(sizeof(struct complex_t) * N/2);
    d = (complex*) malloc(sizeof(struct complex_t) * N/2);
    for(k = 0; k < N/2; k++) {
        e[k] = x[2*k];
        d[k] = x[2*k + 1];
    }

    E = FFT_simple(e, N/2);
    D = FFT_simple(d, N/2);
    
    for(k = 0; k < N/2; k++) {
        /* Multiply entries of D by the twiddle factors e^(-2*pi*i/N * k) */
        D[k] = complex_mult(complex_from_polar(1, -2.0*M_PI*k/N), D[k]);
    }

    for(k = 0; k < N/2; k++) {
        X[k]       = complex_add(E[k], D[k]);
        X[k + N/2] = complex_sub(E[k], D[k]);
    }

    free(D);
    free(E);
    return X;
}

complex complex_from_polar(double r, double theta_radians) {
    complex result;
    result.re = cos(theta_radians);
    result.im = sin(theta_radians);
    return result;
}

int main(){
    complex *table, *printing;
    table = fillArray(8);
    printing = CT_FFT(table, 8);
    printf("TOTAL PROCESSED SAMPLES: %d\n", 8);
    printf("=====================================\n");
    for(int i=0; i < 8; i++){
        //print real and imaginary values
        printf("XR[%d]: %f  XI[%d]: %f i\n", i, printing[i].real, i, printing[i].imag);
        printf("=====================================\n");
    }
    return 0;
}