
#include <iostream>
#include "tbb/tbb.h"

#define RADIUS 3
#define NELEMENTS 1048576

int main(void) {
	int * h_in, *h_out;
	size_t bytes = NELEMENTS*sizeof(int);
	h_in = (int*)malloc(bytes);
    h_out = (int*)malloc(bytes);

	//initialization 
    for( int i = 0; i < NELEMENTS; i++ ) {
      	h_in[i] = 1;
    }

    //Applying the stencil
    tbb::tick_count t0 = tbb::tick_count::now();
	tbb::parallel_for(0, NELEMENTS, 1, [=](int i){
		for(int j = -RADIUS; j <= RADIUS; j++){
			h_out[i] += (i+j < 0) ||  (i+j >= NELEMENTS) ? 0 : h_in[i+j];
		}
	});
    tbb::tick_count t1 = tbb::tick_count::now();
    printf("time for action = %g seconds\n", (t1-t0).seconds() );

	//print/check
    //for(int j=0; j < NELEMENTS; j++){
    //	std::cout << "h_out[" << j << "]= " << h_out[j] << "\n";
    //}


	free(h_in);
	free(h_out);
	return 0;
}
