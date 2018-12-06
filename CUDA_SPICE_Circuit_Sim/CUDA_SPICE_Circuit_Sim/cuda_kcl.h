#pragma once
#include "spice.h"
#include "cuda_netlist.h"
#include "cuda_setup.h"

__global__ void kernDCPassiveMat(int n, int n_nodes, CUDA_Elem* passives, float* gMat, float* iMat);

void gpuNetlistToMat(CUDA_Net* dev_net, float** gMat, float* iMat);