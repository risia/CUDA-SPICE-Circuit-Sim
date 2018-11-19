#pragma once
#include "spice.h"

/*

Functions to Populate G and I matrices for Linear Solver
Using Kirchoff's Current Law

*/

void R_toMat(Resistor* R, float** gMat);

int Vdc_toMat(Vdc* V, float** gMat, float* iMat, float* vMat, int num_nodes);

void Idc_toMat(Idc* I, float* iMat);

void VCCS_toMat(VCCS* I, float** gMat);