#pragma once

// Gaussian Elimination, to be put in a loop from k = 0 to m-1 
// to fully reduce matrix to upper triangle
void matReduce(float** gMat, float* iMat, int m, int n, int k);

// Solve reduced matrix, if it's solveable
void matSolve(float** gMat, float* iMat, float* vMat, int m, int n, int k);