#pragma once

#include "cuda_setup.h"

// Constants
#define TOL 1e-6
#define MAX_FLOAT 	3.402823466e38

// Kernel functions
__global__ void kernMatReduce(int n, float* gMat, float* iMat, int k);
__global__ void kernPlugKnownV(int n, float* gMat, float* iMat, float* vMat);
__global__ void kernMatSolve(int n, float* gMat, float* iMat, float* vMat);
__global__ void kernTolCheck(int n, float* vMat, float* vGuess, bool* isConverged);

// CPU side setup and execution of kernels
void gpuMatReduce(int n, float* gMat, float* iMat);

void gpuMatSolve(int n, float** gMat, float* iMat, float* vMat);

void gpuNonLinConverge(int n, float** gMat, float* iMat, float* vMat);