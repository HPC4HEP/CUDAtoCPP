//Example of the ideal output file we should get from translating inputs/vecAdd.cu (HANDMADE)


/* Translation
 *
 * TODO: find the subset of the ALWAYS-NEEDED #include, or otherwise
 * always inserting them all
 */
#include <math.h>
#include <stddef.h>
#include <thread>
#include <vector>
#include <iostream>

/* Translation:
 *
 * this decl is always needed, because at least we have the threadIdx
 * declaration in the new "kernel"
 *
 */
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

void vecAdd(double *a, double *b, double *c, int n, dim3 gridDim, dim3 blockDim, dim3 blockIdx)
{
  /* Translation:
   *
   * those three for loops should always be inserted in the new "kernel"
   * and also the dim3 threadIdx decl.
   * In this way we can simply copypaste the old cuda kernel body and
   * it will run. (Are we sure about it?)
   *
   */
  dim3 threadIdx(blockDim.x, blockDim.y, blockDim.z);
  for(threadIdx.x=0; threadIdx.x < blockDim.x; threadIdx.x++){
    for(threadIdx.y=0; threadIdx.y < blockDim.y; threadIdx.y++){
      for(threadIdx.z=0; threadIdx.z < blockDim.z; threadIdx.z++){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id<n){
          c[id] = a[id] + b[id];
        }
      }
    }
  }
}


/* Translation:
 *
 * host code seems the trickiest part until now.
 * 1) we have to manage CUDA API calls: not sure if we can simply delete them
 * 		or if we need some informations (maybe from their parameters)
 * 	1b) also, take into account the use of cudaMallocManaged
 *
 * 2) what we are gonna do about device and host versions of the variables?
 *    simply solved using cudaMallocManaged instead of cudaMalloc + cudaMemcpy?
 *
 */
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
  //FIXME?
  unsigned int AVAILABLE_THREADS = 8; //std::thread::hardware_concurrency();

  /* Translation:
   *
   * The kernel call has to be surrounded by some stuff:
   *
   * FIXME: ARE WE TAKING INTO ACCOUNT ONLY 1DIMENSIONAL CASE?
   */
  auto nSteps = (gridSize+AVAILABLE_THREADS-1)/AVAILABLE_THREADS;
  std::vector<std::thread> t;
  for(int tid=0; tid < AVAILABLE_THREADS; ++tid){
      for(auto k=0; k < nSteps; ++k){
    	  auto blockIdx = tid + AVAILABLE_THREADS*k;
    	  if(blockIdx < blockSize){
    		  //the old kernel call becames a thread instance
    		  // \param gridSize maybe useless
    		  t.push_back(std::thread(vecAdd, h_a, h_b, h_c, n, gridSize, blockSize, blockIdx));
    	  }
      }
  }

  //always neeeded?
  for(auto& thread : t)
	  thread.join();

  //NEXT 4 LINES ONLY FOR CHECKING, NOT PART OF THE TRANSLATION
  double sum = 0;
  for(i=0; i<n; i++)
      sum += h_c[i];
  std::cout << "final result: " << sum/n << "\n";

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
