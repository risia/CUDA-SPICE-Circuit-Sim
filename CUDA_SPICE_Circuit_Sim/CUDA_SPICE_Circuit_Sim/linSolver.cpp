#include "linSolver.h"

// Gaussian Elimination, to be put in a loop from k = 0 to m-1 
// to fully reduce matrix to upper triangle
void matReduce(float** gMat, float* iMat, int m, int n, int k) {

	float* row0 = gMat[k];
	float val0 = row0[k];
	if (val0 == 0.0f) return;

	float* row;
	float val;
	float ratio;
	for (int i = 0; i < m; i++) {
		if (i == k) continue;

		row = gMat[i];
		val = row[k];
		ratio = val / val0;

		for (int j = 0; j < n; j++) {
			row[j] -= ratio * row0[j];
		}
		iMat[i] -= ratio * iMat[k];
	}
}

// Solve reduced matrix, if it's solveable
void matSolve(float** gMat, float* iMat, float* vMat, int m, int n, int k) {
	if (vMat[k] != 0.0f) return;

	//error?
	if (gMat[k][k] == 0) return;

	float v = iMat[k] / gMat[k][k];
	vMat[k] = v;
}