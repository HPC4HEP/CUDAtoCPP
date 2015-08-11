__host__ void f();

__global__ void g1(int x) {
	x++;
}

int main(void) {
  dim3 g(1,1,1);
  int k = 2;
  g1<<<1, 1>>>(42);
  g1<<<1, k>>>(42);
  g1<<<1, dim3(1,1,1)>>>(42);
  g1<<<1, g>>>(42);
  f();
}
