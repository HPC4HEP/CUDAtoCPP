#define RADIUS 3
#define BLOCK_SIZE 32

__global__ void stencil_1d(int *in, int *out) {

     __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
     
     int g_index = threadIdx.x + blockIdx.x * blockDim.x;
     int s_index = threadIdx.x + RADIUS;

     // Read input elements into shared memory
     temp[s_index] = in[g_index];
     
     if (threadIdx.x < RADIUS) {     
     
     
         temp[s_index - RADIUS] = in[g_index - RADIUS];
         
         
         temp[s_index + BLOCK_SIZE] = in[g_index + BLOCK_SIZE];
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
	int n = 50;
	int * h_in, *h_out;
	int * d_in, *d_out;
	size_t bytes = n*sizeof(int);
	h_in = (int*)malloc(bytes);
    h_out = (int*)malloc(bytes);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    int i;
    for( i = 0; i < n; i++ ) {
      h_in[i] = sin(i)*sin(i) + cos(i)* cos(i);
    }
    cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice);
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)n/blockSize);
 	
	stencil_1d<<<gridSize,blockSize>>>(d_in, d_out);
	
	cudaMemcpy( h_out, d_out, bytes, cudaMemcpyDeviceToHost );
	cudaFree(d_in);
	cudaFree(d_out);
	free(h_in);
	free(h_out);
	return 0;
}