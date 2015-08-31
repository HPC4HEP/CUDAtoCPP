__global__ void g1(int x) {
	int j;
	int i = 0;

	if(i < 1) {
		i++;
	} else {
		__syncthreads();
	}
	
	while(i<1){
		i++;
		__syncthreads();
		i--
	}
	
	for(i = 0; i < 2; i++){
		a();
		__syncthreads();
		b();
	}

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
