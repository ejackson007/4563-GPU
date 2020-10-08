//Evan Jackson
//Cooley Tukey

#include <stdio.h>
#include <math.h>
#include <complex.h>

#define SIZE 8192

cplx *fillArray(){
    cplx *all_nums = (cplx *)malloc(sizeof(cplx) * SIZE);
    assert(all_nums != NULL);
    all_nums[0] = 3.6 + 2.6*I;
    all_nums[1] = 2.9 + 6.3*I;
    all_nums[2] = 5.6 + 4*I;
    all_nums[3] = 4.8 + 9.1*I;
    all_nums[4] = 3.3 + 0.4*I;
    all_nums[5] = 5.9 + 4.8*I;
    all_nums[6] = 5 + 2.6*I;
    all_nums[7] = 4.3 + 4.1*I;
    for(int i = 8; i < SIZE; i++)
        all_nums[i] = 0;
    return all_nums;
}

cplx * CT_FFT(cplx * table, int n){
    cplx * X = (cplx *)malloc(sizeof(cplx) * SIZE);
    cplx *odd, *even, *ODD, *EVEN;

    if (n == 1){
        X[0] = table[0];
        return X;
    }

    even = (cplx *)malloc(sizeof(cplx) * SIZE/2);
    odd = (cplx *)malloc(sizeof(cplx) * SIZE/2);
    
}

int main(){

}
