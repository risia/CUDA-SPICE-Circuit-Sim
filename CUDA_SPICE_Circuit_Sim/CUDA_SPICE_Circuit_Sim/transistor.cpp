#include "transistor.h"

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

	float k = T->model->u0 * 3.9 * PERMITTIVITY / (T->model->tox * 100.f);

	// saturation
	if (Vd > (Vg - vth)) {
		return (0.5f * k * (Vg - vth) * (Vg - vth));
	}
	// "linear"
	else {
		return (k * ((Vg - vth) * Vd) + (0.5 * Vd * Vd));
	}
}