#pragma once
#include "spice.h"

/*

Functions to Populate G and I matrices for Linear Solver

*/

void R_toMat(Resistor* R, float** gMat);

void Vdc_toMat(Vdc* V, float** gMat, float* iMat, int num_nodes);

void Idc_toMat(Idc* I, float* iMat);