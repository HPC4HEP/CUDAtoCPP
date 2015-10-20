#define RADIUS 3
#define BLOCK_SIZE 32
#define NELEMENTS 1048576
//#include "tbb/tbb.h"
#include <stdio.h>
__global__ void stencil_1d(int *in, int *out) {

     __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
     
     int g_index = threadIdx.x + blockIdx.x * blockDim.x;
     int s_index = threadIdx.x + RADIUS;

     // Read input elements into shared memory
     temp[s_index] = in[g_index];
     
     if (threadIdx.x < RADIUS) {     
     
     
         temp[s_index - RADIUS] = g_index - RADIUS >= 0? in[g_index - RADIUS]: 0;
         
         
         temp[s_index + BLOCK_SIZE] = g_index + BLOCK_SIZE < NELEMENTS ? in[g_index + BLOCK_SIZE]: 0;
     }

  __syncthreads();

// Apply the stencil
     int result = 0;
     int offset;
     for(offset = -RADIUS ; offset <= RADIUS ; offset++)
         result += temp[s_index + offset];

     // Store the result
     out[g_index] = result;
}

int main(void) {
	
	int * h_in, *h_out;
	int * d_in, *d_out;
	size_t bytes = NELEMENTS*sizeof(int);
	h_in = (int*)malloc(bytes);
    h_out = (int*)malloc(bytes);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    int i;
    for( i = 0; i < NELEMENTS; i++ ) {
      h_in[i] = 1;
    }
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize((int)ceil((float)NELEMENTS/blockSize.x), 1, 1);

    //tbb::tick_count t0 = tbb::tick_count::now();

    cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice);





	stencil_1d<<<gridSize,blockSize>>>(d_in, d_out);


	cudaMemcpy( h_out, d_out, bytes, cudaMemcpyDeviceToHost );

    //tbb::tick_count t1 = tbb::tick_count::now();
    //printf("time for action = %g seconds\n", (t1-t0).seconds() );

	cudaFree(d_in);
	cudaFree(d_out);
	free(h_in);
	free(h_out);
	return 0;
}
