#include "kcl.h"

void R_toMat(Resistor* R, float** gMat) {
	int a = R->node1 - 1;
	int b = R->node2 - 1;

	float val = R->val;

	if (a >= 0) {
		gMat[a][a] += 1 / val;
	}
	if (b >= 0) {
		gMat[b][b] += 1 / val;
	}
	if (a >= 0 && b >= 0) {
		gMat[a][b] -= 1 / val;
		gMat[b][a] -= 1 / val;
	}
}

int Vdc_toMat(Vdc* V, float** gMat, float* iMat, int num_nodes) {
	int n_n = V->node_n - 1; // pos node
	int n_p = V->node_p - 1; // neg node


	if ((n_n < 0 && n_p < 0) || n_n == n_p) {
		// error, shorted
		return -1;
	}

	float self;
	float c;

	// negative node grounded
	if (n_p >= 0 && n_n < 0) {
		// abstract vsource to current
		self = gMat[n_p][n_p];
		c = V->val * self;

		for (int i = 0; i < num_nodes; i++) {
			if (i != n_p) gMat[n_p][i] = 0.0f;
		}
		iMat[n_p] += c;
	}
	// positive node grounded
	else if (n_p < 0 && n_n >= 0) {
		// abstract vsource to current
		self = gMat[n_n][n_n];
		c = V->val * self;

		for (int i = 0; i < num_nodes; i++) {
			if (i != n_n) gMat[n_n][i] = 0.0f;
		}
		iMat[n_n] -= c;
	}

	// neither grounded
	else {

		self = gMat[n_n][n_n];
		c = V->val * self;
		float temp;

		for (int i = 0; i < num_nodes; i++) {
			temp = gMat[n_p][i];
			gMat[n_p][i] += gMat[n_n][i];
			gMat[n_n][i] += temp;
		}

		gMat[n_p][n_p] += self;
		gMat[n_p][n_n] -= self;

		iMat[n_p] += c;
	}


	return 0;
}

void Idc_toMat(Idc* I, float* iMat) {
	int n_n = I->node_n - 1; // pos node
	int n_p = I->node_p - 1; // neg node

	if (n_p >= 0) iMat[n_p] -= I->val;
	if (n_n >= 0) iMat[n_n] += I->val;
}