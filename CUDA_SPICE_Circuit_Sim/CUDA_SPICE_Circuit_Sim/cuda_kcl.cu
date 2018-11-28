#include "cuda_kcl.h"

// Copy netlist to GPU
void gpuNetlist(Netlist netlist) {

}


/*
// Parallelize R, IDC, and VCCS list by element
__global__ void kernDCPassiveMat(int n, Element* elems, float* gMat, float* iMat) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index

	if (idx >= n) return;

	Element* e = elems + idx;
	char type = e->type;

	int a = e->nodes[0] - 1;
	int b = e->nodes[1] - 1;

	float val = e->params[0];
	if (type == 'R') val = 1.0f / val;

	// If it's shorted, no contribution
	if (a == b ) return;

	// DC Current Source
	if (type == 'I') {
		if (a >= 0) atomicAdd(iMat + a, -val);
		if (b >= 0) atomicAdd(iMat + b, val);
		return;
	}

	int c = (type == 'G') ? e->nodes[2] : a;
	int d = (type == 'G') ? e->nodes[3] : b;

	if (c == d) return;

	// Resistor or VCCS
	int gidx = a * n + c;
	if (a >= 0 && c >= 0) atomicAdd(gMat + gidx, val);

	gidx = b * n + d;
	if (b >= 0 && d >= 0) atomicAdd(gMat + gidx, val);

	if (b >= 0 && a >= 0 && d >= 0 && c >= 0) {
		gidx = a * n + d;
		atomicAdd(gMat + gidx, -val);

		gidx = b * n + c;
		atomicAdd(gMat + gidx, -val);
	}
}


void kernVDCtoMat(int n, Element* elems, float* gMat, float* iMat) {
	// 
}
*/