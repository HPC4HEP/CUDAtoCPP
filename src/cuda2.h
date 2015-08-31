/* Minimal declarations for CUDA support.  Testing purposes only. */

#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global)) //putting also extern "C" causes some problems on file ids
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __attribute__((always_inline))

typedef struct {
    unsigned int x, y, z;
} uint3;

//typedef or not?
struct dim3 {
  unsigned int x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1)
      : x(x), y(y), z(z) {}
};

uint3 __device__ extern const threadIdx;
uint3 __device__ extern const blockIdx;
dim3 __device__ extern const blockDim;
dim3 __device__ extern const gridDim;
int __device__ extern const warpSize;

// The following is some bits of the CUDA runtime, currently required for Clang
// to parse kernel invocation expressions correctly.
typedef struct cudaStream* cudaStream_t;

int cudaConfigureCall(dim3 grid_size, dim3 block_size, unsigned shared_size = 0,
                      cudaStream_t stream = 0);

