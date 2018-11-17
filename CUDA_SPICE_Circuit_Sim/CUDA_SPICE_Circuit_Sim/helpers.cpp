#include "helpers.h"

float** mat2D(int m, int n) {
	float** mat = (float**)malloc(m * sizeof(float*));
	for (int i = 0; i < m; i++) {
		mat[i] = (float*)malloc(n * sizeof(float));
		for (int j = 0; j < n; j++) {
			mat[i][j] = 0.0f;
		}
	}

	return mat;
}

float* mat1D(int n) {
	float* mat = (float*)malloc(n * sizeof(float));
	for (int j = 0; j < n; j++) {
		mat[j] = 0.0f;
	}
	return mat;
}

void freeMat2D(float** mat, int n) {
	for (int i = 0; i < n; i++) {
		free(mat[i]);
	}
}

string  mat2DToStr(float** mat, int m, int n) {
	string matStr = "";
	for (int i = 0; i < m; i++) {
		matStr += string("[ ");
		for (int j = 0; j < n; j++) {
			matStr += to_string(mat[i][j]) + string(" ");
		}
		matStr += ("]\n");
	}
	return matStr;
}

string mat1DToStr(float* mat, int n) {
	string matStr = "[ ";
	for (int i = 0; i < n; i++) {
		matStr += to_string(mat[i]) + string(" ");
	}
	matStr += ("]\n");
	return matStr;
}