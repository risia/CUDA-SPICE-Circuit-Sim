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

// Vmat should be most recent guesses, gMat and iMat restored to passive elements
// currently considering single transistor system
void MOS_toMat(Transistor* T, float** gMat, float* iMat, float* vGuess, int n) {
	//float Id;
	float I;
	float g;

	int n_g = T->g - 1;
	int n_d = T->d - 1;
	int n_s = T->s - 1;

	// Load guessed node voltages
	float Vg = 0.0f;
	if (n_g >= 0) Vg = vGuess[n_g];
	float Vs = 0.0f;
	if (n_s >= 0) Vs = vGuess[n_s];
	float Vd = 0.0f;
	if (n_d >= 0) Vd = vGuess[n_d];

	// switch drain and source if necessary
	// right now assuming symmetric transistors
	if (Vs > Vd) {
		n_s = T->d - 1;
		Vs = 0.0f;
		if (n_s >= 0) Vs = vGuess[n_s];

		n_d = T->s - 1;
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

	float k = (T->w / T->l) * T->model->u0 * (3.9f * PERMITTIVITY / (T->model->tox * 100.f));

	float Vov = Vg - Vs - vth;

	// saturation region
	if (Vd - Vs > Vov) {
		g = 0.5f * k * Vov;
		I = g * vth;

		if (n_d >= 0) {
			iMat[n_d] += I;
			if (n_g >= 0) gMat[n_d][n_g] += g;
			if (n_s >= 0) gMat[n_d][n_s] -= g;
		}
		if (n_s >= 0) {
			iMat[n_s] -= I;
			gMat[n_s][n_s] += g;
			if (n_g >= 0) gMat[n_s][n_g] -= g;
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
	/*
	if (n_d > 0) {
		//iMat[n_d - 1] -= Id;
		gMat[n_d - 1][n_d - 1] += g;
		iMat[n_d - 1] += I;
		if (n_s > 0) gMat[n_d - 1][n_s - 1] -= g;
	}
	if (n_s > 0) {
		//iMat[n_s - 1] += Id;
		gMat[n_s - 1][n_s - 1] -= g;
		iMat[n_s - 1] -= I;
		if (n_d > 0) gMat[n_s - 1][n_d - 1] += g;
	}
	*/
}