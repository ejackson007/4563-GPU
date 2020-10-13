#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 8192
typedef struct complex_t {
    double re;
    double im;
} complex;

complex complex_from_polar(double r, double theta_radians);
double  complex_magnitude(complex c);
complex complex_add(complex left, complex right);
complex complex_sub(complex left, complex right);
complex complex_mult(complex left, complex right);
complex* FFT_simple(complex* x, int N /* must be a power of 2 */);

complex *fillArray(){
    complex *all_nums = (complex *)malloc(sizeof(struct complex_t) * SIZE);
    all_nums[0].re = 3.6;
    all_nums[0].im = 2.6;
    all_nums[1].re = 2.9;
    all_nums[1].im = 6.3;
    all_nums[2].re = 5.6;
    all_nums[2].im = 4.0;
    all_nums[3].re = 4.8;
    all_nums[3].im = 9.1;
    all_nums[4].re = 3.3;
    all_nums[4].im = 0.4;
    all_nums[5].re = 5.9;
    all_nums[5].im = 4.8;
    all_nums[6].re = 5.0;
    all_nums[6].im = 2.6;
    all_nums[7].re = 4.3;
    all_nums[7].im = 4.1;
    for(int i = 8; i < SIZE; i++){
        all_nums[i].re = 0;
        all_nums[i].im = 0;
    }
    return all_nums;
}

int main(){
    complex *table, *printing;
    table = fillArray();
    printing = FFT_simple(table, SIZE);
    printf("TOTAL PROCESSED SAMPLES: %d\n", SIZE);
    printf("=====================================\n");
    for(int i=0; i < 8; i++){
        //print real and imaginary values
        printf("XR[%d]: %f  XI[%d]: %fi\n", i, printing[i].re, i, printing[i].im);
        printf("=====================================\n");
    }
    return 0;
}

complex* FFT_simple(complex* x, int N /* must be a power of 2 */) {
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

double complex_magnitude(complex c) {
    return sqrt(c.re*c.re + c.im*c.im);
}

complex complex_add(complex left, complex right) {
    complex result;
    result.re = left.re + right.re;
    result.im = left.im + right.im;
    return result;
}

complex complex_sub(complex left, complex right) {
    complex result;
    result.re = left.re - right.re;
    result.im = left.im - right.im;
    return result;
}

complex complex_mult(complex left, complex right) {
    complex result;
    result.re = left.re*right.re - left.im*right.im;
    result.im = left.re*right.im + left.im*right.re;
    return result;
}