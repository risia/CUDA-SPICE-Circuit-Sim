#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include "spice.h"


/*

Functions to Populate G and I matrices for Linear Solver
Using Kirchoff's Current Law

*/

// Passive/DC Components
void R_toMat(Element* R, float** gMat);
int Vdc_toMat(Element* V, float** gMat, float* iMat, float* vMat, int num_nodes);
void Idc_toMat(Element* I, float* iMat);
void VCCS_toMat(Element* I, float** gMat);

// Transient only
void C_toMat(Element* C, float** gMat, float* iMat, float* vMat, float h);
void VTran_toMat(Element* V, float** gMat, float* iMat, float* vMat, float time, int num_nodes);

// For op matrix gen
void linNetlistToMat(Netlist* netlist, float** gMat, float* iMat);

// Transient matrix gen, includes caps, vMat from prev. timestep
void tranNetlistToMat(Netlist* netlist, float** gMat, float* iMat, float* vMat, float h);
void tranJustCToMat(Netlist* netlist, float** gMat, float* iMat, float* vMat, float h);


// For dc sweep need to find swept element
// Since we're looping through anyway, easiest here
Element* linNetlistToMatFindElem(Netlist* netlist, float** gMat, float* iMat, float* vMat, char* name);