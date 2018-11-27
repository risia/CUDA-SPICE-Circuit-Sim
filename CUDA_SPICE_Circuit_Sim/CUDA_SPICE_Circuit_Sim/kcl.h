#pragma once
#include "spice.h"

/*

Functions to Populate G and I matrices for Linear Solver
Using Kirchoff's Current Law

*/

void R_toMat(Element* R, float** gMat);

int Vdc_toMat(Element* V, float** gMat, float* iMat, float* vMat, int num_nodes);

void Idc_toMat(Element* I, float* iMat);

void VCCS_toMat(Element* I, float** gMat);

// For op matrix gen
void linNetlistToMat(Netlist netlist, float** gMat, float* iMat, float* vMat);

// For dc sweep need to find swept element
// Since we're looping through anyway, easiest here
void  linNetlistToMatFindElem(Netlist netlist, float** gMat, float* iMat, float* vMat, char* name, char &type, int &index);