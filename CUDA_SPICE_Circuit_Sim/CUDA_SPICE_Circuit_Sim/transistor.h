#pragma once
#include "spice.h"

#define PERMITTIVITY  8.854187817e-14

float calcId(Element* T, float* vMat);

void MOS_toMat(Element* T, float** gMat, float* iMat, float* vGuess, int n);