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
	int n = V->nodes[1] - 1; // pos node
	int p = V->nodes[0] - 1; // neg node

	// shorted
	if (n == p) return -1;

	float val = V->params[0];

	// negative node grounded
	if (p >= 0 && n < 0) {
		for (int i = 0; i < num_nodes; i++) {
			gMat[p][i] = 0.0f;
		}

		gMat[p][p] = 1.0f;
		iMat[p] = val;
		vMat[p] = val;
	}
	// positive node grounded
	else if (p < 0 && n >= 0) {
		for (int i = 0; i < num_nodes; i++) {
			gMat[n][i] = 0.0f;
		}

		gMat[n][n] = 1.0f;
		iMat[n] = -val;
		vMat[n] = -val;
	}
	// neither grounded
	else {
		for (int i = 0; i < num_nodes; i++) {
			gMat[n][i] += gMat[p][i];
			gMat[p][i] = 0.0f;
		}
		iMat[n] += iMat[p];
		iMat[p] = val;
		gMat[p][n] = -1.0f;
		gMat[p][p] = 1.0f;

		if (vMat[p] != 0.0f) vMat[n] = vMat[p] - val;
		else if (vMat[n] != 0.0f) vMat[p] = vMat[n] + val;
	}

	return 0;
}

void Idc_toMat(Element* I, float* iMat) {
	int p = I->nodes[0] - 1; // neg node
	int n = I->nodes[1] - 1; // pos node

	float val = I->params[0];

	if (p >= 0) iMat[p] -= val;
	if (n >= 0) iMat[n] += val;
}

void VCCS_toMat(Element* I, float** gMat) {
	int ip = I->nodes[0] - 1;
	int in = I->nodes[1] - 1;
	int vp = I->nodes[2] - 1;
	int vn = I->nodes[3] - 1;

	float g = I->params[0];
	if (ip >= 0) {
		if (vp >= 0) gMat[ip][vp] += g;
		if (vn >= 0) gMat[ip][vn] -= g;
	}
	if (in >= 0) {
		if (vp >= 0) gMat[in][vp] -= g;
		if (vn >= 0) gMat[in][vn] += g;
	}
}

// h = timestep to approximate
// USE vMat FROM PREVIOUS TIME STEP
void C_toMat(Element* C, float** gMat, float* iMat, float* vMat, float h) {
	int p = C->nodes[0] - 1;
	int n = C->nodes[1] - 1;

	// shorted/ both ends grounded
	if (p == n) return;

	float cap = C->params[0];
	if (cap == 0) return;

	float G = cap / h;

	if (p >= 0) {
		gMat[p][p] += G;
		iMat[p] += G * vMat[p];
	}
	if (n >= 0) {
		gMat[n][n] += G;
		iMat[n] += G * vMat[n];
	}
	if (n >= 0 && p >= 0) {
		gMat[p][n] -= G;
		gMat[n][p] -= G;
		iMat[p] -= G * vMat[n];
		iMat[n] -= G * vMat[p];
	}
}

void VTran_toMat(Element* V, float** gMat, float* iMat, float* vMat, float time, int num_nodes) {
	// Normal DC voltage source
	if (V->type == 'V') Vdc_toMat(V, gMat, iMat, vMat, num_nodes);

	// Pulse source
	else if (V->type == 'P') {
		// Pulse Parameters
		float V1 = V->params[1]; // initial val
		float V2 = V->params[2]; // peak val
		float td = V->params[3]; // initial delay
		float tr = V->params[4]; // rise time
		float tf = V->params[5]; // fall time
		float width = V->params[6]; // pulse width
		float period = V->params[7]; // period, time for one cycle

		// put time in context of current pulse
		float p_time = (time - td) - period * floor((time - td) / period);

		// Calculate voltage value for time instance
		float val;
		if (time < td ) val = 0.0f; // time is before pulse start
		else if (p_time < tr) val = V1 + (p_time * (V2 - V1)) / tr; // interpolate value
		else if (p_time < width) val = V2;
		else if (p_time < tf + width) val = V2 - ((p_time - width) * (V2 - V1)) / tf; // interpolate value
		else val = V1;

		// just so I don't need to make a new function or anything for this
		// I'm overwriting the DC value, using the DC function, and restoring the value
		float vdc = V->params[0];
		V->params[0] = val;
		Vdc_toMat(V, gMat, iMat, vMat, num_nodes);
		V->params[0] = vdc;

		//cout << "Time: " << time << " Source: " << V->name << " Val: " << val << " Rel. Time: " << p_time << " TF: " << tf << "\n";
	}

	// Sinusoid source
	else if (V->type == 'S') {
		float Vo = V->params[0]; // DC offset
		float Va = V->params[1]; // Amplitude
		float f = V->params[2]; // Frequency
		float td = V->params[3]; // initial delay
		float theta = V->params[4]; // dampening factor / second
		float p = V->params[5]; // phase offset in degrees

		// amplitude
		float A = Va * expf(-theta * (time - td));
		// sinusoid
		float sine = sinf(2.0f * M_PI * f * (time - td) + (p / 360.0f));

		if (time >= td && f != 0.0f) V->params[0] += A * sine;

		//cout << "Time: " << time << " Source: " << V->name << "Val: " << V->params[0] << "\n";

		Vdc_toMat(V, gMat, iMat, vMat, num_nodes);
		V->params[0] = Vo;

		
	}
}




void linNetlistToMat(Netlist* netlist, float** gMat, float* iMat) {
	Element* passives = netlist->elements.data();
	int num_passive = netlist->elements.size();

	char type;

	// Populate Matrices from passive elements
	for (int i = 0; i < num_passive; i++) {
		type = passives[i].type;
		if (type == 'R') R_toMat(passives + i, gMat);
		else if (type == 'I') Idc_toMat(passives + i, iMat);
		else if (type == 'G') VCCS_toMat(passives + i, gMat);
	}
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

void tranNetlistToMat(Netlist* netlist, float** gMat, float* iMat, float* vMat, float h) {
	Element* passives = netlist->elements.data();
	int num_passive = netlist->elements.size();

	char type;

	// Populate Matrices from passive elements
	for (int i = 0; i < num_passive; i++) {
		type = passives[i].type;
		if (type == 'R') R_toMat(passives + i, gMat);
		else if (type == 'I') Idc_toMat(passives + i, iMat);
		else if (type == 'G') VCCS_toMat(passives + i, gMat);
		else if (type == 'C' && h != 0.0f) C_toMat(passives + i, gMat, iMat, vMat, h);
	}
}

void tranJustCToMat(Netlist* netlist, float** gMat, float* iMat, float* vMat, float h) {
	Element* passives = netlist->elements.data();
	int num_passive = netlist->elements.size();

	// Populate Matrices from passive elements
	for (int i = 0; i < num_passive; i++) {
		if (passives[i].type == 'C' && h != 0.0f) C_toMat(passives + i, gMat, iMat, vMat, h);
	}
}