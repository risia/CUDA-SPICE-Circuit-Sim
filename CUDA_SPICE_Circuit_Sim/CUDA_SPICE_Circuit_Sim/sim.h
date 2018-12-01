#pragma once
#include "spice.h"
#include "helpers.h"
#include "kcl.h"
#include "cuda_kcl.h"
#include "cuda_LinSolver.h"
#include "transistor.h"

void op(Netlist netlist);

float** dcSweep(Netlist netlist, char* name, float start, float stop, float step);

void cuda_op(CUDA_Net netlist);