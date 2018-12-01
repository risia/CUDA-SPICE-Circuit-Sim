#pragma once
#include "spice.h"
#include "helpers.h"
#include "kcl.h"
#include "cuda_kcl.h"
#include "cuda_LinSolver.h"
#include "transistor.h"

void op(Netlist netlist);

void cuda_op(CUDA_Net netlist);