#pragma once
#include <cstdlib>
#include <string> 
#include <sstream>
using namespace std;


// Matrix helper functions
float** mat2D(int m, int n);

float* mat1D(int n);

void freeMat2D(float** mat, int n);

string  mat2DToStr(float** mat, int m, int n);

string mat1DToStr(float* mat, int n);

void matCpy(float* dst, float* src, int n);

void resetMat1D(float* mat, int n);
void resetMat2D(float** mat, int m, int n);

float maxDiff(float* mat1, float* mat2, int n);
bool matDiffCmp(float* mat1, float* mat2, int n, float tol);
