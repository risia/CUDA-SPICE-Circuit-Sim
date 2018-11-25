#include "transistor.h"

// WIP

// guessed voltage matrix vMat as input
// keep solved values separate?
// calcId calculates current of a transistor for given v matrix for testing
float calcId(Transistor* T, float* vMat) {
	float Id;

	int n_g = T->g;
	int n_d = T->d;
	int n_s = T->s;

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
		n_s = T->d;
		Vs = 0.0f;
		if (n_s > 0) Vs = vMat[n_s - 1];

		n_d = T->s;
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

	float k = (T->w / T->l) * T->model->u0 * (3.9f * PERMITTIVITY / (T->model->tox * 100.f));

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

// calc voltage "error" on CPU
// Use ORIGINAL gMat and iMat, before solving, 
// then modified w/ new transistor currents from solution
float calcError(float** gMat, float* iMat, float* vMat, int row, int n) {
	/*
		Error = f(V) / f'(V);
	*/

	float f = iMat[row];
	for (int i = 0; i < n; i++) {
		f -= gMat[row][i] * vMat[i];
	}

	float df = gMat[row][row];

	return f / df;
}

// Vmat should be most recent guesses, gMat and iMat restored to passive elements
// currently considering single transistor system
void MOS_toMat(Transistor* T, float** gMat, float* iMat, float* vGuess, int n) {
	//float Id;
	float I;
	float gm;

	int n_g = T->g;
	int n_d = T->d;
	int n_s = T->s;

	// Load guessed node voltages
	float Vg = 0.0f;
	if (n_g > 0) Vg = vGuess[n_g - 1];
	float Vs = 0.0f;
	if (n_s > 0) Vs = vGuess[n_s - 1];
	float Vd = 0.0f;
	if (n_d > 0) Vd = vGuess[n_d - 1];

	// switch drain and source if necessary
	// right now assuming symmetric transistors
	if (Vs > Vd) {
		n_s = T->d;
		Vs = 0.0f;
		if (n_s > 0) Vs = vGuess[n_s - 1];

		n_d = T->s;
		Vd = 0.0f;
		if (n_d > 0) Vd = vGuess[n_d - 1];
	}

	float vth = T->model->vt0;
	// subthreshold, ignoring for now
	if (Vg - Vs <= vth && T->model->type == 'n') {
		//Id = 0.0f;
		gm = 0.0f;
		I = 0.0f;
	}
	if (Vg - Vs >= vth && T->model->type == 'p') {
		//Id = 0.0f;
		gm = 0.0f;
		I = 0.0f;
	}

	float k = (T->w / T->l) * T->model->u0 * (3.9f * PERMITTIVITY / (T->model->tox * 100.f));

	float Vov = Vg - Vs - vth;

	// saturation region
	if (Vd - Vs > Vov) {
		//Id = (0.5f * k * Vov * Vov);
		I = -(0.5f * k * Vov * Vov);
		gm = 0.0f;
	}
	// "linear" region
	else {
		//Id = k * ((Vov * (Vd - Vs)) + (0.5 * (Vd - Vs) * (Vd - Vs)));
		gm = k * Vov;
		I = k * 0.5 * (Vd - Vs) * (Vd - Vs);
	}

	if (n_d > 0) {
		//iMat[n_d - 1] -= Id;
		gMat[n_d - 1][n_d - 1] += gm;
		iMat[n_d - 1] += I;
		if (n_s > 0) gMat[n_d - 1][n_s - 1] -= gm;
	}
	if (n_s > 0) {
		//iMat[n_s - 1] += Id;
		gMat[n_s - 1][n_s - 1] -= gm;
		iMat[n_s - 1] -= I;
		if (n_d > 0) gMat[n_s - 1][n_d - 1] += gm;
	}
}