//this is to test my functions in serial
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 8192
typedef struct complex_t {
    double real;
    double imag;
} complex;

void oddMultCalc(complex * oddDevice, int i, int n){
    complex result, polar;
    polar.real = cos(-2.0*M_PI*(i/n));
    polar.imag = sin(-2.0*M_PI*(i/n));

    result.real = polar.real*oddDevice[i].real - polar.imag*oddDevice[i].imag;
    result.imag = polar.real*oddDevice[i].imag + polar.imag*oddDevice[i].real;

    oddDevice[i] = result;
}

void addOddEven(complex * evenDevice, complex * oddDevice, complex * XDevice, int i, int n){
    //add
    XDevice[i].real = evenDevice[i].real + oddDevice[i].real;
    XDevice[i].imag = evenDevice[i].imag + oddDevice[i].imag;
    //subtract
    XDevice[i + n/2].real = evenDevice[i].real - oddDevice[i].real;
    XDevice[i + n/2].imag = evenDevice[i].imag - oddDevice[i].imag;
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
    complex *odd, *even, *ODD, *EVEN;
    

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

    for(int i = 0; i < n/2; i++){
        oddMultCalc(ODD, i, n);
    }
    for(int i = 0; i < n/2; i++){
        addOddEven(EVEN, ODD, X, i, n);
    }

    free(EVEN);
    free(ODD);
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