#pragma once
#include "spice.h"
#include "helpers.h"
#include "kcl.h"
#include "cuda_kcl.h"
#include "cuda_LinSolver.h"
#include "transistor.h"
#include "linSolver.h"

void op(Netlist* netlist);

void cuda_op(Netlist* netlist);

void dcSweep(Netlist* netlist, char* name, float start, float stop, float step);

void transient(Netlist* netlist, float start, float stop, float step);

void OP_CPUtest(Netlist* netlist);