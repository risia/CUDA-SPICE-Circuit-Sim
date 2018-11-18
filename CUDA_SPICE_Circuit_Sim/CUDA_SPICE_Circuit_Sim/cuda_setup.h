#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char *msg, const char *file, int line);

void setupDevMats(int n, float** gMat, float* dev_gMat, float* iMat, float* dev_iMat, float* dev_vMat);
void copyDevMats(int n, float** gMat, float* dev_gMat, float* iMat, float* dev_iMat, float* vMat, float* dev_vMat);
void cleanDevMats(float* dev_gMat, float* dev_iMat, float* dev_vMat);
