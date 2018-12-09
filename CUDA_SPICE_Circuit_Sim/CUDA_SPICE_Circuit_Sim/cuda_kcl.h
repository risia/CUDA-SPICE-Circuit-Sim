#pragma once
#include "spice.h"
#include "cuda_netlist.h"
#include "cuda_setup.h"
#include "transistor.h"


#define BS_1D (64)

__global__ void kernDCPassiveMat(int n, int n_nodes, CUDA_Elem* passives, float* gMat, float* iMat);
__global__ void kernVDCtoMat(int n_v, int n_nodes, CUDA_Elem* elems, float* gMat, float* iMat, float* vMat);

void gpuNetlistToMat(CUDA_Net* dev_net, Netlist* netlist, float* dev_gMat, float* dev_iMat, float* dev_vMat, float* dev_vGuess);

void gpuPassiveVDCToMat(CUDA_Net* dev_net, Netlist* netlist, float* dev_gMat, float* dev_iMat, float* dev_vMat);