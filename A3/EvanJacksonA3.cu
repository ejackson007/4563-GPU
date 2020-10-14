//Evan Jackson
#include <stdio.h>

#define SIZE 10240

__global__
void vecProdCyclic(int* aDevice, int* bDevice, int* cDevice, int block){
    int i = threadIdx.x + blockDim.x * blockIdx.x; // get location of thread
    int jump = SIZE/block;//creates jump for each thread to make
    for(int x = 1; x <= 5; x++)
        cDevice[i + (jump*x)] = aDevice[i + (jump*x)] * bDevice[i + (jump*x)];//add to c and save to total
}

__global__
void vecProdNonCyclic(int* aDevice, int* bDevice, int* cDevice, int block){
    int i = threadIdx.x + blockDim.x * blockIdx.x; // get location of thread
    cDevice[i + block] = aDevice[i + block] * bDevice[i + block];//add to c and save to total
}

//I am unsure if the function above it, or this is what you would consider block, since both solve
//the problem by adding chunks of the array at a time, however i figured it would probably be best
//to add both functions just incase. This approach instead gives each thred an equal amount of 
//consecutive 
__global__
void nonCyclical2(int* aDevice, int* bDevice, int* cDevice){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    for(int x = 0; x < 5; x++){
        cDevice[i*5 + x] = aDevice[i*5 + x] * bDevice[i*5 + x];
    }
}

//host code
int main(){
    //create variable to create arrays
    size_t arraySize = SIZE * sizeof(int);
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
        bHost[i] = 2*i + 1;
    }

    //allocate memory for device and transfer to device
    cudaMalloc(&aDevice, arraySize);
    cudaMalloc(&bDevice, arraySize);
    cudaMalloc(&cDevice, arraySize);
    cudaMemcpy(aDevice, aHost, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, bHost, arraySize, cudaMemcpyHostToDevice);

 /**********************************************************************************
 ******************** 2 Blocks, Non Cyclic *****************************************
 **********************************************************************************/  

    //maximum amount of threads, with 2 blocks.
    dim3 dimGrid2(2,1,1);
    dim3 dimBlock(1024,1,1);

    //call gpu process
    nonCyclical2<<<dimGrid2,dimBlock>>>(aDevice, bDevice, cDevice);


    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);

    //print for 2 block size
    printf("2 Blocks - Not Cyclic(C[0], C[10239]) = {%d, %d)\n", cHost[0], cHost[SIZE - 1]);

    //reset C array
    for(int i = 0; i < SIZE; i++){
        cHost[i] = 0;
    }
    cudaMemcpy(cDevice, cHost, arraySize, cudaMemcpyHostToDevice);


/**********************************************************************************
 ******************** 2 Blocks, Cyclic ********************************************
 **********************************************************************************/ 

    int jump = (SIZE / 1024) / 2;
    //call gpu process
    vecProdCyclic<<<dimGrid2,dimBlock>>>(aDevice, bDevice, cDevice, jump);

    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);

    //print for 2 block size
    printf("2 Blocks - Cyclic(C[0], C[10239]) = {%d, %d)\n", cHost[0], cHost[SIZE - 1]);

    //reset C array
    for(int i = 0; i < SIZE; i++){
        cHost[i] = 0;
    }
    cudaMemcpy(cDevice, cHost, arraySize, cudaMemcpyHostToDevice);

/**********************************************************************************
 ******************** 10, Blocks ********************************************
 **********************************************************************************/ 

    //maximum amount of threads, with 10 blocks.
    dim3 dimGrid10(10,1,1);

    //call gpu process
    jump = (SIZE / 1024) / 10;
    for(int i = 0; i < jump; i++){
        vecProdNonCyclic<<<dimGrid10,dimBlock>>>(aDevice, bDevice, cDevice, i*(SIZE/jump));
    }

    //transfer back to host
    cudaMemcpy(cHost, cDevice, arraySize, cudaMemcpyDeviceToHost);

    //free device memory;
    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);
    //print first and last for 10 blocks
    printf("10 Blocks - (C[0], C[10239]) = {%d, %d)\n", cHost[0], cHost[SIZE - 1]);

    return 0;
}