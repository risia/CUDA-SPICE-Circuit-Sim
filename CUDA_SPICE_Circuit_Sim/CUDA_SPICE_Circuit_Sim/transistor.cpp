#include "transistor.h"

// WIP

// guessed voltage matrix vMat as input
// keep solved values separate.
float calcId(Transistor* T, float* vMat) {
	// Load guessed node voltages
	float Vg = 0;
	if (T->g > 0) Vg = vMat[T->g - 1];
	float Vs = 0;
	if (T->s > 0) Vs = vMat[T->s - 1];
	float Vd = 0;
	if (T->d > 0) Vs = vMat[T->d - 1];

	float vth = T->model->vt0;
	// subthreshold
	if (Vg - Vs <= vth && T->model->type == 'n') return 0.0f;
	if (Vg - Vs >= vth && T->model->type == 'p') return 0.0f;

	float k = (T->w / T->l) * T->model->u0 * (3.9f * PERMITTIVITY / (T->model->tox * 100.f));

	// saturation region
	if (Vd > (Vg - vth)) {
		return (0.5f * k * (Vg - vth) * (Vg - vth));
	}
	// "linear" region
	else {
		return (k * ((Vg - vth) * Vd) + (0.5 * Vd * Vd));
	}
}

void MOS_toMat(Transistor* T, float** gMat, float* iMat, float* vGuess) {
	// Load guessed node voltages
	float Vg = 0.0f;
	if (T->g > 0) Vg = vGuess[T->g - 1];
	float Vs = 0.0f;
	if (T->s > 0) Vs = vGuess[T->s - 1];
	float Vd = 0.0f;
	if (T->d > 0) Vs = vGuess[T->d - 1];


}