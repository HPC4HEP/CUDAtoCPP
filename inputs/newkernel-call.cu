#include "../src/cuda2.h"

//#define __host__ __attribute__((host))
//#define __device__ __attribute__((device))
//#define __global__ __attribute__((global))
//#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))


//struct dim3 {
//  unsigned int x, y, z;
//  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1)
//      : x(x), y(y), z(z) {}
//};

//int cudaConfigureCall(dim3 gridSize, int blockSize);

//#include "include/cuda_runtime.h"

__global__ void f();
__global__ void g1(int x) {
x++;
}


int main(void) {
  dim3 g(1,1,1);
  g1<<<g, 1>>>(42);
}
