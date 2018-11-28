#pragma once
#include "spice.h"
#include "cuda_setup.h"

__global__ void kernDCPassiveMat(int n, Element* elems, float* gMat, float* iMat);