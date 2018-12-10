#pragma once

#include "cuda_netlist.h"
#include "cuda_kcl.h"
#include "cuda_LinSolver.h"

void full_cudaOp(Netlist* net, CUDA_Net* dev_net);
void full_cudaDCSweep(Netlist* net, CUDA_Net* dev_net, char* name, float start, float stop, float step);


int fullCudaOP_Out(Netlist* net, CUDA_Net* dev_net, float** vOut);
int fullCudaSweep_Out(Netlist* net, CUDA_Net* dev_net, char* name, float start, float stop, float step, int n_steps, float** vOut);
int fullCudaTran_Out(Netlist* net, CUDA_Net* dev_net, float start, float stop, float step, int n_steps, int skipped_steps, float** vOut);