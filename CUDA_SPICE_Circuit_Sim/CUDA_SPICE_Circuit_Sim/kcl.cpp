#include "kcl.h"

void R_toMat(Resistor* R, float** gMat) {
	int a = R->node1 - 1;
	int b = R->node2 - 1;

	float val = R->val;

	if (a >= 0) {
		gMat[a][a] += 1.0f / val;
	}
	if (b >= 0) {
		gMat[b][b] += 1.0f / val;
	}
	if (a >= 0 && b >= 0) {
		gMat[a][b] -= 1.0f / val;
		gMat[b][a] -= 1.0f / val;
	}
}

int Vdc_toMat(Vdc* V, float** gMat, float* iMat, float* vMat, int num_nodes) {
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
		iMat[n_p] = c;

		vMat[n_p] = V->val;
	}
	// positive node grounded
	else if (n_p < 0 && n_n >= 0) {
		// abstract vsource to current
		self = gMat[n_n][n_n];
		c = V->val * self;

		for (int i = 0; i < num_nodes; i++) {
			if (i != n_n) gMat[n_n][i] = 0.0f;
		}
		iMat[n_n] = -c;

		vMat[n_n] = -V->val;
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

		iMat[n_p] = c;

		if (vMat[n_p] != 0) {
			vMat[n_n] = vMat[n_p] - V->val;
		}
		if (vMat[n_n] != 0) {
			vMat[n_p] = vMat[n_n] + V->val;
		}
	}


	return 0;
}

void Idc_toMat(Idc* I, float* iMat) {
	int n_n = I->node_n - 1; // pos node
	int n_p = I->node_p - 1; // neg node

	if (n_p >= 0) iMat[n_p] -= I->val;
	if (n_n >= 0) iMat[n_n] += I->val;
}

void VCCS_toMat(VCCS* I, float** gMat) {
	int ip = I->ip - 1;
	int in = I->in - 1;
	int vp = I->vp - 1;
	int vn = I->vn - 1;

	float g = I->g;
	if (ip >= 0) {
		if (vp >= 0) {
			gMat[ip][vp] += g;
		}
		if (vn >= 0) {
			gMat[ip][vn] -= g;
		}
	}
	if (in >= 0) {
		if (vp >= 0) {
			gMat[in][vp] -= g;
		}
		if (vn >= 0) {
			gMat[in][vn] += g;
		}
	}
}


void linNetlistToMat(Netlist netlist, float** gMat, float* iMat, float* vMat) {
	Resistor* rList = netlist.rList.data();
	Vdc* vdcList = netlist.vdcList.data();
	Idc* idcList = netlist.idcList.data();
	VCCS* vccsList = netlist.vccsList.data();

	int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_r = netlist.rList.size();
	int num_vdc = netlist.vdcList.size();
	int num_idc = netlist.idcList.size();
	int num_vccs = netlist.vccsList.size();

	// Populate G matrix from Resistor Elements
	for (int i = 0; i < num_r; i++) {
		R_toMat(rList + i, gMat);
	}
	// IDC Sources populate I matrix
	for (int i = 0; i < num_idc; i++) {
		Idc_toMat(idcList + i, iMat);
	}
	for (int i = 0; i < num_vccs; i++) {
		VCCS_toMat(vccsList + i, gMat);
	}
	// VDC Source populates G and I matrices
	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
	}

}