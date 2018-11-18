#pragma once

#include "cuda_setup.h"

__global__ void kernMatReduce(int n, float** gMat, float* iMat) {
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

	// keep in matrix bounds
	// matrix always square
	if (idx >= n) return;


}