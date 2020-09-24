//Evan Jackson
#include <stdio.h>

#define SIZE 4096

__global__
void vecAdd(int* aDevice, int* bDevice, int* cDevice){
    int i = threadIdx.x + blockDim.x * blockIdx.x; // get location of thread
    if(i < SIZE){
        cDevice[i] = aDevice[i] + bDevice[i];//add to c and save to total
    }
}

//host code
int main(){
    //create variable to create arrays
    size_t arraySize = SIZE * sizeof(int);
    int total = 0;
    //host vector
    int* aHost; 
    int* bHost;
    int* cHost;
    //device vector
    int* aDevice;
    int* bDevice;
    int* cDevice;

    //allocate for host
    aHost = (int*)malloc(arraySize);
    bHost = (int*)malloc(arraySize);
    cHost = (int*)malloc(arraySize);
    //fill array aHost and bHost
    for(int i = 0; i < SIZE; i++){
        aHost[i] = i;
        Host[i] = 4095 + i;
    }

    //allocate memory for device and transfer to device
    cudaMalloc(&aDevice, arraySize);
    cudaMalloc(&bDevice, arraySize);
    cudaMalloc(&cDevice, arraySize);
    cudaMemcpy(aDevice, aHost, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, bHost, arraySize, cudaMemcpyHostToDevice);

    //the maximum amount of threads is 1024, therefore for optimal use,
    //each block needs to be size 4 (4096/1024 = 4)
    dim3 dimGrid(4,1,1);
    dim3 dimBlock(1024,1,1);

    //call gpu process
    vecAdd<<<dimGrid,dimBlock>>>(aDevice, bDevice, cDevice);

    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);
    //free device memory;
    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);
    //find total of vector
    for(int i = 0; i < SIZE; i++){
        total += cHost[i];
    }

    //print values
    printf("First element of vector C: %i\n", cHost[0]);
    printf("Last element of vector C: %i\n", cHost[SIZE - 1]);
    printf("Summation of Elements in vector C: %i\n", total);

    return 0;
}