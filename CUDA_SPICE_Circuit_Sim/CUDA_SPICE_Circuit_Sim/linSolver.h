#pragma once

// Gaussian Elimination, to be put in a loop from k = 0 to m-1 
// to fully reduce matrix to upper triangle
void matReduce(float** gMat, float* iMat, int m, int n, int k) {
	if (m == k + 1) return;

	float* row0 = gMat[k];
	float val0 = row0[k];
	if (val0 == 0.0f) return;

	float* row;
	float val;
	float ratio;
	for (int i = k + 1; i < m; i++) {
		row = gMat[i];
		val = row[k];
		ratio = val / val0;
		for (int j = k; j < n; j++) {
			row[j] -= ratio * row0[j];
		}
		iMat[i] -= ratio * iMat[k];
	}
}

// Solve reduced matrix, if it's solveable
void matSolve(float** gMat, float* iMat, float* vMat, int m, int n, int k) {
	float v = iMat[m - k - 1];
	float r = gMat[m - k - 1][m - k - 1];


	if (r != 0.0f) v /= r;
	else {
		// insert error here. No solution known.
		printf("Divide by 0 error\n");
		return;
	}

	vMat[m - k - 1] = v;

	float c;
	for (int i = 0; i < m - k; i++) {
		c = gMat[i][m - k - 1] * v;
		iMat[i] -= c;
		//gMat[i][m - k - 1] = 0.0f;
	}
}