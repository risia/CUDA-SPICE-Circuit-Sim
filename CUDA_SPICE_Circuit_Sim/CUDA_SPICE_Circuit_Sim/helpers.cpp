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
	free(mat);
}

string  mat2DToStr(float** mat, int m, int n) {
	std::ostringstream out;
	out.precision(6);
	for (int i = 0; i < m; i++) {
		out << "[ ";
		for (int j = 0; j < n; j++) {
			out << std::scientific << mat[i][j] << " ";
		}
		out << "]\n";
	}
	return out.str();
}

string mat1DToStr(float* mat, int n) {
	std::ostringstream out;
	out.precision(6);
	out << "[ ";
	for (int i = 0; i < n; i++) {
		out << std::scientific << mat[i] << " ";
	}
	out << "]\n";

	return out.str();
}

void matCpy(float* dst, float* src, int n) {
	for (int i = 0; i < n; i++) {
		dst[i] = src[i];
	}
}

void mat2DCpy(float** dst, float** src, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			dst[i][j] = src[i][j];
		}
	}
}

void resetMat1D(float* mat, int n) {
	for (int i = 0; i < n; i++) {
		mat[i] = 0.0f;
	}
}
void resetMat2D(float** mat, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			mat[i][j] = 0.0f;
		}
	}
}

// find max diff between two arrays
float maxDiff(float* mat1, float* mat2, int n) {
	float max = 0.0f;
	float diff;
	for (int i = 0; i < n; i++) {
		diff = fabs(mat1[i] - mat2[i]);
		if (diff > max) max = diff;
	}

	return max;
}

// Compare diff to tolerance
bool matDiffCmp(float* mat1, float* mat2, int n, float tol) {
	float diff;
	for (int i = 0; i < n; i++) {
		diff = fabs(mat1[i] - mat2[i]);
		if (diff > tol ) return false; //|| diff > (tol * mat2[i])
	}
	return true;
}

void mat2DtoCSV(char** labels, float** mat, int m, int n, char* filename) {
	ofstream file;
	file.open(filename, std::ios_base::app);

	file.precision(6);

	for (int j = 0; j < n; j++) {
		file << labels[j] << ", ";
	}
	file << "\n";

	file << std::scientific;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			file << mat[i][j] << ", ";
		}
		file << "\n";
	}

	file.close();
}