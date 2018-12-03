#pragma once
#include <cstdlib>
#include <string> 
#include <sstream>
#include<iostream> 
#include<fstream>
using namespace std;


// Matrix helper functions
float** mat2D(int m, int n);

float* mat1D(int n);

void freeMat2D(float** mat, int n);

string  mat2DToStr(float** mat, int m, int n);

string mat1DToStr(float* mat, int n);

void matCpy(float* dst, float* src, int n);
void mat2DCpy(float** dst, float** src, int m, int n);

void resetMat1D(float* mat, int n);
void resetMat2D(float** mat, int m, int n);

float maxDiff(float* mat1, float* mat2, int n);
bool matDiffCmp(float* mat1, float* mat2, int n, float tol);

void mat2DtoCSV(char** labels, float** mat, int m, int n, char* filename);