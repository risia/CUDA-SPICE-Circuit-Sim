#include "transistor.h"

// WIP

// guessed voltage matrix vMat as input
// calcId calculates current of a transistor for given v matrix for testing
float calcId(Element* T, float* vMat) {
	float Id;

	int n_d = T->nodes[0];
	int n_g = T->nodes[1];
	int n_s = T->nodes[2];

	// Load guessed node voltages
	float Vg = 0.0f;
	if (n_g > 0) Vg = vMat[n_g - 1];
	float Vs = 0.0f;
	if (n_s > 0) Vs = vMat[n_s - 1];
	float Vd = 0.0f;
	if (n_d > 0) Vd = vMat[n_d - 1];

	// switch drain and source if necessary
	// right now assuming symmetric transistors
	if (Vs > Vd) {
		n_s = T->nodes[0];
		Vs = 0.0f;
		if (n_s > 0) Vs = vMat[n_s - 1];

		n_d = T->nodes[2];
		Vd = 0.0f;
		if (n_d > 0) Vd = vMat[n_d - 1];
	}

	float vth = T->model->vt0;
	// subthreshold, ignoring for now
	if (Vg - Vs <= vth && T->model->type == 'n') {
		Id = 0.0f;
	}
	if (Vg - Vs >= vth && T->model->type == 'p') {
		Id = 0.0f;
	}

	float k = (T->params[1] / T->params[0]) * T->model->u0 * (3.9f * PERMITTIVITY / (T->model->tox * 100.f));

	float Vov = Vg - Vs - vth;

	// saturation region
	if (Vd - Vs > Vov) {
		Id = (0.5f * k * Vov * Vov);
	}
	// "linear" region
	else {
		Id = k * ((Vov * (Vd - Vs)) + (0.5 * (Vd - Vs) * (Vd - Vs)));
	}

	return Id;
}

// Vmat should be most recent guesses, gMat and iMat restored to passive elements
void MOS_toMat(Element* T, float** gMat, float* iMat, float* vGuess, int n) {
	//float Id;
	float I;
	float g;

	int n_d = T->nodes[0] - 1;
	int n_g = T->nodes[1] - 1;
	int n_s = T->nodes[2] - 1;

	// Load guessed node voltages
	float Vg = 0.0f;
	if (n_g >= 0) Vg = vGuess[n_g];
	float Vs = 0.0f;
	if (n_s >= 0) Vs = vGuess[n_s];
	float Vd = 0.0f;
	if (n_d >= 0) Vd = vGuess[n_d];


	// switch drain and source if necessary
	// right now assuming symmetric transistors
	if ((Vs > Vd && T->model->type == 'n') || (Vs < Vd && T->model->type == 'p')) {
		n_s = T->nodes[0] - 1;
		Vs = 0.0f;
		if (n_s >= 0) Vs = vGuess[n_s];

		n_d = T->nodes[2] - 1;
		Vd = 0.0f;
		if (n_d >= 0) Vd = vGuess[n_d];
	}

	float vth = T->model->vt0;
	// subthreshold, ignoring for now
	if (Vg - Vs <= vth && T->model->type == 'n') {
		return;
	}
	if (Vg - Vs >= vth && T->model->type == 'p') {
		return;
	}

	float k = (T->params[1] / T->params[0]) * T->model->u0 * (T->model->epsrox * PERMITTIVITY / (T->model->tox * 100.f));
	

	float Vov = Vg - Vs - vth;

	// Channel Length Modulation Multiplier
	// Ids = Ids0 * (1 + lambda * Vds)
	float CLM = T->model->pclm * Vov;

	if (T->model->type == 'p') {
		k = -k;
		CLM = -CLM;
	}

	// saturation region
	if ((Vd - Vs > Vov && T->model->type == 'n') || (Vd - Vs < Vov && T->model->type == 'p')) {
		g = 0.5f * k * Vov;
		I = g * (1 - CLM) * vth;

		if (n_d >= 0) {
			iMat[n_d] += I;
			gMat[n_d][n_d] += g * CLM;
			if (n_g >= 0) gMat[n_d][n_g] += g * (1 - CLM);
			if (n_s >= 0) gMat[n_d][n_s] -= g;
		}
		if (n_s >= 0) {
			iMat[n_s] -= I;
			gMat[n_s][n_s] += g;
			if (n_g >= 0) gMat[n_s][n_g] -= g * (1 - CLM);
			if (n_d >= 0) gMat[n_s][n_d] -= g * CLM;
		}
	}
	// "linear" region
	else {
		//Id = k * ((Vov * (Vd - Vs)) + (0.5 * (Vd - Vs) * (Vd - Vs)));
		g = k * Vov;
		I = k * 0.5 * (Vd - Vs) * (Vd - Vs);

		if (n_d >= 0) {
			gMat[n_d][n_d] += g;
			iMat[n_d] += I;
			if (n_s >= 0) gMat[n_d][n_s] -= g;
		}
		if (n_s >= 0) {
			gMat[n_s][n_s] += g;
			iMat[n_s] -= I;
			if (n_d >= 0) gMat[n_s][n_d] -= g;
		}
	}
}


void transientMOS_toMat(Element* T, float** gMat, float* iMat, float* vGuess, float* vPrev, int n, float h) {
	float I;
	float g;

	int n_d = T->nodes[0] - 1;
	int n_g = T->nodes[1] - 1;
	int n_s = T->nodes[2] - 1;
	int n_b = T->nodes[3] - 1;

	// Load guessed node voltages
	float Vg = 0.0f;
	if (n_g >= 0) Vg = vGuess[n_g];
	float Vs = 0.0f;
	if (n_s >= 0) Vs = vGuess[n_s];
	float Vd = 0.0f;
	if (n_d >= 0) Vd = vGuess[n_d];

	if ((Vs > Vd && T->model->type == 'n') || (Vs < Vd && T->model->type == 'p')) {
		n_s = T->nodes[0] - 1;
		Vs = 0.0f;
		if (n_s >= 0) Vs = vGuess[n_s];

		n_d = T->nodes[2] - 1;
		Vd = 0.0f;
		if (n_d >= 0) Vd = vGuess[n_d];
	}

	float vth = T->model->vt0;
	if (Vg - Vs <= vth && T->model->type == 'n') return;
	if (Vg - Vs >= vth && T->model->type == 'p') return;

	float L = T->params[0];
	float W = T->params[1];
	float Cox = (T->model->epsrox * PERMITTIVITY / (T->model->tox * 100.f));

	float k = (W / L) * T->model->u0 * Cox;
	float Vov = Vg - Vs - vth;
	float CLM = T->model->pclm * Vov;

	if (T->model->type == 'p') {
		k = -k;
		CLM = -CLM;
	}

	// Cgb
	Element C;
	C.nodes.push_back(T->nodes[1]);
	C.nodes.push_back(T->nodes[3]);
	C.params.push_back(T->model->CGBO * L);

	C_toMat(&C, gMat, iMat, vPrev, h);

	if ((Vd - Vs > Vov && T->model->type == 'n') || (Vd - Vs < Vov && T->model->type == 'p')) {
		g = 0.5f * k * Vov;
		I = g * (1 - CLM) * vth;

		// Cgs
		//float Cgcs = (2.0f * L * Cox * 1e4) / 3.0f;
		float Cgs = W * T->model->CGSO; //W * (Cgcs + T->model->CGSO);

		C.nodes[0] = T->nodes[1];
		C.nodes[1] = T->nodes[2];
		C.params[0] = Cgs;

		C_toMat(&C, gMat, iMat, vPrev, h);

		// Cgd
		float Cgd = W * T->model->CGDO;

		C.nodes[0] = T->nodes[1];
		C.nodes[1] = T->nodes[0];
		C.params[0] = Cgd;

		C_toMat(&C, gMat, iMat, vPrev, h);


		if (n_d >= 0) {
			iMat[n_d] += I;
			gMat[n_d][n_d] += g * CLM;
			if (n_g >= 0) gMat[n_d][n_g] += g * (1 - CLM);
			if (n_s >= 0) gMat[n_d][n_s] -= g;
		}
		if (n_s >= 0) {
			iMat[n_s] -= I;
			gMat[n_s][n_s] += g;
			if (n_g >= 0) gMat[n_s][n_g] -= g * (1 - CLM);
			if (n_d >= 0) gMat[n_s][n_d] -= g * CLM;
		}
	}
	// "linear" region
	else {
		// Cgs
		float Cgcs = 0.5f * L * Cox * 1e4;
		float Cgs = W * T->model->CGSO; //W * (Cgcs + T->model->CGSO);

		C.nodes[0] = T->nodes[1];
		C.nodes[1] = T->nodes[2];
		C.params[0] = Cgs;

		C_toMat(&C, gMat, iMat, vPrev, h);

		// Cgd
		float Cgd = W * T->model->CGDO; //  W * (Cgcs + T->model->CGDO);

		C.nodes[0] = T->nodes[1];
		C.nodes[1] = T->nodes[0];
		C.params[0] = Cgd;

		C_toMat(&C, gMat, iMat, vPrev, h);


		g = k * Vov;
		I = k * 0.5 * (Vd - Vs) * (Vd - Vs);

		if (n_d >= 0) {
			gMat[n_d][n_d] += g;
			iMat[n_d] += I;
			if (n_s >= 0) gMat[n_d][n_s] -= g;
		}
		if (n_s >= 0) {
			gMat[n_s][n_s] += g;
			iMat[n_s] -= I;
			if (n_d >= 0) gMat[n_s][n_d] -= g;
		}
	}
}