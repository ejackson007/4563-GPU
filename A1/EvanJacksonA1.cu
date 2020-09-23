//Evan Jackson
#include <stdio.h>

#define SIZE 4096

float total = 0;

__global__
void vecAdd(int* aDevice, int* bDevice, int* cDevice){
    int i = threadIdx.x + blockDim.x * blockId.x; // get location of thread
    if(i < SIZE){
        cDevice[i] = aDevice[i] + bDevice[i];//add to c and save to total
        total += cDevice[i];
    }
}

//host code
int main(){
    int arraySize = SIZE * sizeof(int);
    int* aHost, bHost, cHost, aDevice, bDevice, cDevice;

    //allocate for host and store
    aHost = (int*)malloc(arraySize);
    bHost = (int*)malloc(arraySize);
    cHost = (int*)malloc(arraySize);
    //fill array aHost
    for(int i = 0; i < SIZE; i++){
        aHost[i] = i;
    }
    //fill array bHost
    for(int i = 0; i < SIZE; i++){
        bHost[i] = 4905 + i;
    }

    //allocate memory for device and transfer to device
    cudaMalloc(&aDevice, arraySize);
    cudaMalloc(&bDevice, arraySize);
    cudaMalloc(&cDevice, arraySize)
    cudaMemcpy(aDevice, aHost, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, bHost, arraySize, cudaMemcpyHostToDevice);

    dim3 dimGrid(4,1,1); //1D of 4 block, so that each block will have maximum threads
    dim3 dimBlock(1024,1,1);

    //call gpu process
    vecAdd<<<dimGrid,dimBlock>>>(aDevice, bDevice, cDevice);

    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);
    //free device memory;
    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);
    //print values
    printf("First element of vector C: %i\n", cHost[0]);
    printf("Last element of vector C: %i\n", cHost[SIZE - 1]);
    printf("Summation of Elements in vector C: %i\n", total);

    return 0;
}