#pragma once
#include "spice.h"
#include "kcl.h"

#define PERMITTIVITY  8.854187817e-14
#define V_THERMAL 2.585199e-2

//float calcId(Element* T, float* vMat);

void MOS_toMat(Element* T, float** gMat, float* iMat, float* vGuess, int n);
void transientMOS_toMat(Element* T, float** gMat, float* iMat, float* vGuess, float* vPrev, int n, float h);