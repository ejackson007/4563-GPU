//Evan Jackson
#include <stdio.h>

#define SIZE 10240

__global__
void vecProd(int* aDevice, int* bDevice, int* cDevice){
    int i = threadIdx.x + blockDim.x * blockIdx.x; // get location of thread
    if(i < SIZE){
        cDevice[i] = aDevice[i] * bDevice[i];//add to c and save to total
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
        aHost[i] = 2*i;
        Host[i] = 2*1 + 1;
    }

    //allocate memory for device and transfer to device
    cudaMalloc(&aDevice, arraySize);
    cudaMalloc(&bDevice, arraySize);
    cudaMalloc(&cDevice, arraySize);
    cudaMemcpy(aDevice, aHost, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, bHost, arraySize, cudaMemcpyHostToDevice);

    //maximum amount of threads, with 2 blocks.
    dim3 dimGrid(2,1,1);
    dim3 dimBlock(1024,1,1);

    //call gpu process
    vecProd<<<dimGrid,dimBlock>>>(aDevice, bDevice, cDevice);

    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);

    //print for 2 block size
    printf("2 Blocks (first,last) = {%d, %d)\n", cHost[0], cHost[SIZE - 1]);
    //reinitiliaze a and b
    for(int i = 0; i < SIZE; i++){
        aHost[i] = 2*i;
        Host[i] = 2*1 + 1;
    }
    //reallocate device memory
    cudaMemcpy(aDevice, aHost, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, bHost, arraySize, cudaMemcpyHostToDevice);

    //repeat for 10 blocks

    //maximum amount of threads, with 10 blocks.
    dim3 dimGrid(10,1,1);
    dim3 dimBlock(1024,1,1);

    //call gpu process
    vecProd<<<dimGrid,dimBlock>>>(aDevice, bDevice, cDevice);

    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);

    //free device memory;
    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);
    //print first and last for 10 blocks
    printf("10 Blocks (first,last) = {%d, %d)\n", cHost[0], cHost[SIZE - 1]);

    return 0;
}