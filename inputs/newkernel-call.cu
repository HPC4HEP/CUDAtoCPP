#include "../src/cuda2.h"

__global__ void f();
__global__ void g1(int x) {
x++;
}


int main(void) {
  dim3 g(1,1,1);
  g1<<<g, 1>>>(42);
}
