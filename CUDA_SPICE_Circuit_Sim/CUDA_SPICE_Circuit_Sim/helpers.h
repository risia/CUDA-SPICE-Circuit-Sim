#pragma once
#include <cstdlib>
#include<string> 
using namespace std;

// Matrix helper functions
float** mat2D(int m, int n);

float* mat1D(int n);

void freeMat2D(float** mat, int n);

string  mat2DToStr(float** mat, int m, int n);

string mat1DToStr(float* mat, int n);


