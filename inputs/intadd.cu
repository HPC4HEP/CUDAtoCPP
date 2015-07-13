//#include "cuda.h"
//#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

int cudaConfigureCall(int gridSize, int blockSize);

//int cudaConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, cudaStream_t stream = 0);

__attribute__((global)) void add(int *a, int *b, int *c){
	*c = *a + *b;
}

int main(void){
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a = 2;
	b = 7;
	
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1,1>>>(d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
