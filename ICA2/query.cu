#include <cstdio>

using namespace std;

int main()
{
	cudaDeviceProp props;
	int count;
	
	cudaGetDeviceCount(&count);
	
	printf("Number of devices: %i\n", count);
	
	const char* div = "================================================================================";
	
	printf("%s\n",div);
	
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&props, i);
		
		printf("Device number: %i\n", i);
		printf("Device name: %s\n", props.name);
		printf("Shared memeory per block: %i B\n", props.sharedMemPerBlock);
		printf("Number of registers per block: %i\n", props.regsPerBlock);
		printf("Threads per Warp: %i\n", props.warpSize);
		printf("Max threads per block: %i\n", props.maxThreadsPerBlock);
		printf("Max x, y, and z threads per block: %i, %i, %i\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		printf("Max x, y, and z blocks: %i, %i, %i\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
		
		printf("%s\n",div);
	}
	return 0;
}