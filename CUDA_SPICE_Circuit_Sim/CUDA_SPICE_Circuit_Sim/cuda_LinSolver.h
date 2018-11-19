#pragma once

#include "cuda_setup.h"

__global__ void kernMatReduce(int n, float* gMat, float* iMat, int k);
__global__ void kernPlugKnownV(int n, float* gMat, float* iMat, float* vMat);
__global__ void kernMatSolve(int n, float* gMat, float* iMat, float* vMat);

void gpuMatReduce(int n, float* gMat, float* iMat);

void gpuMatSolve(int n, float** gMat, float* iMat, float* vMat);