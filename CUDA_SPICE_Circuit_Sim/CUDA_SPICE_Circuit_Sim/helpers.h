#pragma once
#include <cstdlib>
#include <string> 
#include <sstream>
using namespace std;

// Constants
#define TOL 1e-6
#define MAX_FLOAT 	3.402823466e38

// Matrix helper functions
float** mat2D(int m, int n);

float* mat1D(int n);

void freeMat2D(float** mat, int n);

string  mat2DToStr(float** mat, int m, int n);

string mat1DToStr(float* mat, int n);

void matCpy(float* dst, float* src, int n);

float maxDiff(float* mat1, float* mat2, int n);
