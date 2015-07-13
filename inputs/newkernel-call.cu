
#define __global__ __attribute__((global))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

int cudaConfigureCall(int gridSize, int blockSize);

__attribute__((global)) void f();
__attribute__((global)) void g1(int x) {
x++;
}


int main(void) {
  g1<<<1, 1>>>(42);
}
