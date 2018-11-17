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

void Vdc_toMat(Vdc* V, float** gMat, float* iMat, int num_nodes) {
	int n_n = V->node_n - 1; // pos node
	int n_p = V->node_p - 1; // neg node

	// abstract vsource to current
	float self = gMat[n_p][n_p];
	float c = V->val * self;

	// edit pos. node:
	if (n_p >= 0) {
		for (int i = 0; i < num_nodes; i++) {
			if (i != n_p) gMat[n_p][i] = 0.0f;
		}
		iMat[n_p] += c;
	}

	if (n_n >= 0) iMat[n_n] -= c;

	if (n_n >= 0 && n_p >= 0) {
		for (int i = 0; i < num_nodes; i++) {
			if (i != n_p) gMat[n_n][i] += gMat[n_p][i];
		}
	}
}

void Idc_toMat(Idc* I, float* iMat) {
	int n_n = I->node_n - 1; // pos node
	int n_p = I->node_p - 1; // neg node

	if (n_p >= 0) iMat[n_p] -= I->val;
	if (n_n >= 0) iMat[n_n] += I->val;
}