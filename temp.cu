#define RADIUS 3
#define BLOCK_SIZE 32

__global__ void stencil_1d(int *in, int *out) {
dim3 threadIdx;
int tid;
int numThreads = 32;
int g_index[numThreads];
int s_index[numThreads];
int result[numThreads];
int offset[numThreads];
int temp[BLOCK_SIZE + 2 * RADIUS];
for(threadIdx.z=0; threadIdx.z < blockDim.z; threadIdx.z++){
for(threadIdx.y=0; threadIdx.y < blockDim.y; threadIdx.y++){
for(threadIdx.x=0; threadIdx.x < blockDim.x; threadIdx.x++){
tid=threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y;

     
     g_index[tid] = threadIdx.x + blockIdx.x * blockDim.x;
     s_index[tid] = threadIdx.x + RADIUS;

     // Read input elements into shared memory
     temp[s_index[tid]] = in[g_index[tid]];
     if (threadIdx.x < RADIUS) {     
     
     
         temp[s_index[tid] - RADIUS] = in[g_index[tid] - RADIUS];
         
         
         temp[s_index[tid] + BLOCK_SIZE] = in[g_index[tid] + BLOCK_SIZE];
     }

  __syncthreads();

// Apply the stencil
     result[tid] = 0;
     
     for(offset[tid] = -RADIUS ; offset[tid] <= RADIUS ; offset[tid]++)
         result[tid] += temp[s_index[tid] + offset[tid]];

     // Store the result
     out[g_index[tid]] = result[tid];
}}}}

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
