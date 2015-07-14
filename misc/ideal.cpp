void vecAdd(double *a, double *b, double *c, int n, dim3 gridDim, dim3 blockDim, dim3 blockIdx)
{
  for(auto threadIdx.x : blockDim.x){
    for(auto threadIdx.y : blockDim.y){
      for(auto threadIdx.z : blockDim.z){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id<n){
          c[id] = a[id] + b[id];
        }
      }
    }
  }
}

int main()
{
  int n = 1000;
  double *h_a, *h_b, *h_c;
  double *d_a, *d_b, *d_c;
  size_t bytes = n*sizeof(double);
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);
  int i;
  for( i = 0; i < n; i++ ) {
    h_a[i] = sin(i)*sin(i);
    h_b[i] = cos(i)*cos(i);
  }
  int blockSize, gridSize;
  blockSize = 1024;
  gridSize = (int)ceil((float)n/blockSize);
  unsigned int AVAILABLE_THREADS = std::thread::hardware_concurrency();
  auto nSteps = (gridSize+AVAILABLE_THREADS-1)/AVAILABLE_THREADS;
  std::vector<std::thread> t(AVAILABLE_THREADS);
  for(int tid=0; tid < AVAILABLE_THREADS; ++tid){
      for(auto k=0; k < nSteps; ++k){
  	auto blockIdx = tid + AVAILABLE_THREADS*k;
	if(blockIdx < blockSize)
	    t[tid] = std::thread(vecAdd, d_a, d_b, d_c, n, gridSize, blockSize, blockIdx);
      }
  }
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
