#include "../src/cuda2.h"

__global__ void add(int *a, int *b, int*c){
	*c = *a + *b;
}

__host__ int somethingelse(){
	return 3;
}

__device__ int something(){
	return 2;
}

int main(void){
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);
	
	//cudaStream_t st;

	//cudaMalloc((void**)&d_a, size);
	//cudaMalloc((void**)&d_b, size);
	//cudaMalloc((void**)&d_c, size);
	

	a = 2;
	b = 7;
	
	//cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	//dim3 ggg(1,1,1);
	dim3 bbb(1,1,1);
	//add<<<1,1>>>(d_a, d_b, d_c);

	//cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	//cudaFree(d_a);
	//cudaFree(d_b);
	//cudaFree(d_c);
	return 0;
}
