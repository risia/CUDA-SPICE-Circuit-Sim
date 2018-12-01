#include "kcl.h"

void R_toMat(Element* R, float** gMat) {
	int a = R->nodes[0] - 1;
	int b = R->nodes[1] - 1;

	float val = R->params[0];

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

int Vdc_toMat(Element* V, float** gMat, float* iMat, float* vMat, int num_nodes) {
	int n_n = V->nodes[1] - 1; // pos node
	int n_p = V->nodes[0] - 1; // neg node


	if ((n_n < 0 && n_p < 0) || n_n == n_p) {
		// error, shorted
		return -1;
	}

	float self;
	float c;
	float val = V->params[0];

	// negative node grounded
	if (n_p >= 0 && n_n < 0) {
		// abstract vsource to current
		self = gMat[n_p][n_p];
		c = val * self;

		for (int i = 0; i < num_nodes; i++) {
			if (i != n_p) gMat[n_p][i] = 0.0f;
		}
		iMat[n_p] = c;

		vMat[n_p] = val;
	}
	// positive node grounded
	else if (n_p < 0 && n_n >= 0) {
		// abstract vsource to current
		self = gMat[n_n][n_n];
		c = val * self;

		for (int i = 0; i < num_nodes; i++) {
			if (i != n_n) gMat[n_n][i] = 0.0f;
		}
		iMat[n_n] = -c;

		vMat[n_n] = -val;
	}

	// neither grounded
	else {
		for (int i = 0; i < num_nodes; i++) {
			gMat[n_n][i] += gMat[n_p][i];
			gMat[n_p][i] = 0.0f;
		}

		iMat[n_n] += iMat[n_p];

		iMat[n_p] = val;
		gMat[n_p][n_n] = -1.0f;
		gMat[n_p][n_p] = 1.0f;

		if (vMat[n_p] != 0.0f) {
			vMat[n_n] = vMat[n_p] - val;
		}
		if (vMat[n_n] != 0.0f) {
			vMat[n_p] = vMat[n_n] + val;
		}
	}


	return 0;
}

void Idc_toMat(Element* I, float* iMat) {
	int n_p = I->nodes[0] - 1; // neg node
	int n_n = I->nodes[1] - 1; // pos node

	float val = I->params[0];

	if (n_p >= 0) iMat[n_p] -= val;
	if (n_n >= 0) iMat[n_n] += val;
}

void VCCS_toMat(Element* I, float** gMat) {
	int ip = I->nodes[0] - 1;
	int in = I->nodes[1] - 1;
	int vp = I->nodes[2] - 1;
	int vn = I->nodes[3] - 1;

	float g = I->params[0];
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


void linNetlistToMat(Netlist* netlist, float** gMat, float* iMat, float* vMat) {
	//Element* vdcList = netlist.vdcList.data();
	Element* passives = netlist->elements.data();

	//int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_passive = netlist->elements.size();
	//int num_vdc = netlist.vdcList.size();

	char type;

	// Populate Matrices from passive elements
	for (int i = 0; i < num_passive; i++) {
		type = passives[i].type;
		if (type == 'R') R_toMat(passives + i, gMat);
		else if (type == 'I') Idc_toMat(passives + i, iMat);
		else if (type == 'G') VCCS_toMat(passives + i, gMat);

	}
	/*
	// VDC Source populates G and I matrices
	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
	}
	*/

}

Element* linNetlistToMatFindElem(Netlist* netlist, float** gMat, float* iMat, float* vMat, char* name) {
	//Element* vdcList = netlist.vdcList.data();
	Element* passives = netlist->elements.data();

	//int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_passive = netlist->elements.size();
	//int num_vdc = netlist.vdcList.size();

	char t;

	Element* elem = NULL;

	// Populate Matrices from passive elements
	for (int i = 0; i < num_passive; i++) {
		t = passives[i].type;
		if (t == 'R') R_toMat(passives + i, gMat);
		else if (t == 'I') Idc_toMat(passives + i, iMat);
		else if (t == 'G') VCCS_toMat(passives + i, gMat);

		if (strcmp(passives[i].name, name) == 0) {
			elem = &(passives[i]);
		}

	}
	/*
	// VDC Source populates G and I matrices
	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);

		if (strcmp(vdcList[i].name, name) == 0) {
			elem = &(passives[i]);
		}
	}
	*/

	return elem;
}