#pragma once
#include "spice.h"

#define PERMITTIVITY  8.854187817e-14

float calcId(Transistor* T, float* vMat);
float calcError(float** gMat, float* iMat, float* vMat, int row, int n);

void MOS_toMat(Transistor* T, float** gMat, float* iMat, float* vGuess, int n);