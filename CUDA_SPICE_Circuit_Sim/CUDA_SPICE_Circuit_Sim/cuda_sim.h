#pragma once

#include "cuda_netlist.h"
#include "cuda_kcl.h"
#include "cuda_LinSolver.h"

void full_cudaOp(Netlist* net, CUDA_Net* dev_net);
int fullCudaOP_Out(Netlist* net, CUDA_Net* dev_net, float** vOut);