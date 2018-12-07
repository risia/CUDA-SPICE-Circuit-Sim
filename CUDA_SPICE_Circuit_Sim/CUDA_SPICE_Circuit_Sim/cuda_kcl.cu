#include "cuda_kcl.h"

// Parallelize R, IDC, and VCCS list by element
__global__ void kernDCPassiveMat(int n, int n_nodes, CUDA_Elem* passives, float* gMat, float* iMat) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index

	if (idx >= n) return;
	
	CUDA_Elem* e = passives + idx;
	char type = e->type;

	if (type != 'R' && type != 'I' && type != 'G') return;
	
	int a = e->nodes[0] - 1;
	int b = e->nodes[1] - 1;

	
	float val = e->params[0];


	if (type == 'R') {
		val = 1.0f / val;
	}

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
	if (a >= 0 && c >= 0) atomicAdd(gMat + (a * n_nodes + c), val);
	if (b >= 0 && d >= 0) atomicAdd(gMat + (b * n_nodes + d), val);

	if (b >= 0 && a >= 0 && d >= 0 && c >= 0) {
		atomicAdd(gMat + (a * n_nodes + d), -val);
		atomicAdd(gMat + (b * n_nodes + c), -val);
	}
	
}


__global__ void kernVDCtoMat(int n_v, int n_nodes, CUDA_Elem* elems, float* gMat, float* iMat, float* vMat) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n_v) return;
	
	CUDA_Elem* e = elems + idx;
	int n = e->nodes[1] - 1; // pos node
	int p = e->nodes[0] - 1; // neg node

	// shorted
	if (n == p) return;

	float val = e->params[0];

	// negative node grounded,
	// most common case
	if (p >= 0 && n < 0) {

		gMat[p * (n_nodes + 1)] = 1.0f;
		iMat[p] = val;
		vMat[p] = val;
	}
	// positive node grounded
	else if (p < 0 && n >= 0) {

		gMat[n * (n_nodes + 1)] = 1.0f;
		iMat[n] = -val;
		vMat[n] = -val;
	}
	// neither grounded
	else {

		iMat[p] = val;
		gMat[p * n_nodes + n] = -1.0f;
		gMat[p * (n_nodes + 1)] = 1.0f;

		if (vMat[p] != 0.0f) vMat[n] = vMat[p] - val;
		else if (vMat[n] != 0.0f) vMat[p] = vMat[n] + val;
	}

}


__global__ void kernelMOStoMat(int n, int n_nodes, CUDA_Elem* elems, Model* models, float* gMat, float* iMat, float* vGuess) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n) return;

	CUDA_Elem* T = elems + idx;
	Model* M = models + T->model;

	char type = M->type;

	float I;
	float g;

	int n_d = T->nodes[0] - 1;
	int n_g = T->nodes[1] - 1;
	int n_s = T->nodes[2] - 1;
	//int n_b = T->nodes[3] - 1;

	// Load guessed node voltages
	float Vg = 0.0f;
	if (n_g >= 0) Vg = vGuess[n_g];
	float Vs = 0.0f;
	if (n_s >= 0) Vs = vGuess[n_s];
	float Vd = 0.0f;
	if (n_d >= 0) Vd = vGuess[n_d];

	// Switch S and D depending on voltage seen
	if ((Vs > Vd && type == 'n') || (Vs < Vd && type == 'p')) {
		n_s = T->nodes[0] - 1;
		Vs = 0.0f;
		if (n_s >= 0) Vs = vGuess[n_s];

		n_d = T->nodes[2] - 1;
		Vd = 0.0f;
		if (n_d >= 0) Vd = vGuess[n_d];
	}

	float vth = M->vt0;
	float Vov = Vg - Vs - vth;

	float Vds = Vd - Vs;

	// Ideally no current flows through channel for Vds = 0
	//if (Vds == 0.0f) return;

	//Subthreshold, not yet handled
	if ((Vov <= 0 && type == 'n') || (Vov >= 0 && type == 'p')) return;

	// "Constants"
	float L = T->params[0];
	float W = T->params[1];
	float Cox = (M->epsrox * PERMITTIVITY / (M->tox * 100.f));

	float k = (W / L) * M->u0 * Cox;

	float CLM = M->pclm * Vov;

	if (type == 'p') {
		k = -k;
		CLM = -CLM;
	}

	// Saturation, usually desired case
	if ((Vds > Vov && type == 'n') || (Vds < Vov && type == 'p')) {

		g = 0.5f * k * Vov;
		I = g * (1 - CLM) * vth;

		if (n_d >= 0) {
			atomicAdd(iMat + n_d, I);
			atomicAdd(gMat + (n_d * (n_nodes + 1)), g * CLM);
			if (n_g >= 0) atomicAdd(gMat + (n_d * n_nodes + n_g), g * (1 - CLM));
			if (n_s >= 0) atomicAdd(gMat + (n_d * n_nodes + n_s), -g);
		}
		if (n_s >= 0) {
			atomicAdd(iMat + n_s, -I);
			atomicAdd(gMat + (n_s * (n_nodes + 1)), g);
			if (n_g >= 0) atomicAdd(gMat + (n_s * n_nodes + n_g), -g * (1 - CLM));
			if (n_d >= 0) atomicAdd(gMat + (n_s * n_nodes + n_d), -g * CLM);
		}
	}
	// "linear" region
	else {
		g = k * Vov;
		I = k * 0.5 * (Vds * Vds);

		if (n_d >= 0) {
			atomicAdd(gMat + (n_d * (n_nodes + 1)), g);
			atomicAdd(iMat + n_d, I);
			if (n_s >= 0) atomicAdd(gMat + (n_d * n_nodes + n_s), -g);
		}
		if (n_s >= 0) {
			atomicAdd(gMat + (n_s * (n_nodes + 1)), g);
			atomicAdd(iMat + n_s, -I);
			if (n_d >= 0) atomicAdd(gMat + (n_s * n_nodes + n_d), -g);
		}
	}
}

// First part of setting up Voltage sources in matrices
__global__ void kernelAddandZero(int n, float* gMat_d, float* gMat_s, float* i_d, float* i_s) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx > n) return;

	if (idx == n) {
		if (i_d != NULL) i_d[0] += i_s[0];
		i_s[0] = 0.0f;
		return;
	}

	if (gMat_d != NULL) gMat_d[idx] += gMat_s[idx];
	gMat_s[idx] = 0.0f;

}

// Populates Matrices on GPU side
// use CPU netlist for CPU operations
void gpuNetlistToMatTest(CUDA_Net* dev_net, Netlist* netlist, float** gMat, float* iMat, float* vMat) {
	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;

	int n = dev_net->n_nodes;
	if (n == 0) return;
	

	// alloc device memory
	cudaMalloc((void**)&dev_iMat, n * sizeof(float));
	cudaMalloc((void**)&dev_vMat, n * sizeof(float));
	cudaMalloc((void**)&dev_gMat, n * n * sizeof(float));
	checkCUDAError("Malloc Failure!\n");

	cudaMemset(dev_iMat, 0, n * sizeof(float));
	cudaMemset(dev_vMat, 0, n * sizeof(float));
	cudaMemset(dev_gMat, 0, n * n * sizeof(float));
	checkCUDAError("Memset Failure!\n");

	int n_passive = dev_net->n_passive;

	int numBlocks = ceil(float(n_passive) / BS_1D);

	dim3 numBlocks3D = dim3(numBlocks, 1, 1);
	dim3 blockSize = dim3(BS_1D, 1, 1);

	// Passives

	if (n_passive > 0) kernDCPassiveMat << < numBlocks3D, blockSize >> >(n_passive, n, dev_net->passives, dev_gMat, dev_iMat);
	checkCUDAError("Matrix Gen Kernel Failure!\n");


	// Voltage sources

	int n_vdc = dev_net->n_vdc;
	// For each add copy of the + row to the -, then 0 it
	// Then set the Vp - Vn = VDC equation
	int n_p;
	int n_n;

	if (n_vdc > 0) {
		// We're assuming more nodes than voltage sources, generally correct
		numBlocks = ceil(float(n) / float(BS_1D));
		for (int i = 0; i < n_vdc; i++) {
			n_p = netlist->vdcList[i].nodes[0] - 1;
			n_n = netlist->vdcList[i].nodes[1] - 1;
			if (n_p >= 0 && n_n >= 0) kernelAddandZero << < numBlocks3D, blockSize >> > (n, dev_gMat + n * n_n, dev_gMat + n * n_p, dev_iMat + n_n, dev_iMat + n_p);
			else if (n_p >= 0) kernelAddandZero << < numBlocks3D, blockSize >> > (n, NULL, dev_gMat + n * n_p, NULL, dev_iMat + n_p);
			else if (n_n >= 0) kernelAddandZero << < numBlocks3D, blockSize >> > (n, NULL, dev_gMat + n * n_n, NULL, dev_iMat + n_n);
		}

		numBlocks = ceil(float(n_vdc) / float(BS_1D));
		kernVDCtoMat << < numBlocks3D, blockSize >> >(n_vdc, n, dev_net->vdcList, dev_gMat, dev_iMat, dev_vMat);
	}
	
	copyFromDevMats(n, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);

	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);

	checkCUDAError("Device Matrix Free Failure!\n");
}

void gpuPassiveToMat(CUDA_Net* dev_net, float* dev_gMat, float* dev_iMat) {
	int n = dev_net->n_nodes;
	if (n == 0) return;


	int n_passive = dev_net->n_passive;

	int numBlocks = ceil(float(n_passive) / BS_1D);

	// Passives

	if (n_passive > 0) kernDCPassiveMat << < numBlocks, BS_1D >> >(n_passive, n, dev_net->passives, dev_gMat, dev_iMat);
	checkCUDAError("Matrix Gen Kernel Failure!\n");
}

void gpuPassiveVDCToMat(CUDA_Net* dev_net, Netlist* netlist, float* dev_gMat, float* dev_iMat, float* dev_vMat) {
	int n = dev_net->n_nodes;
	if (n == 0) return;


	int n_passive = dev_net->n_passive;

	int numBlocks = ceil(float(n_passive) / float(BS_1D));

	// Passives

	if (n_passive > 0) kernDCPassiveMat << < numBlocks, BS_1D >> >(n_passive, n, dev_net->passives, dev_gMat, dev_iMat);
	checkCUDAError("Matrix Gen Kernel Failure!\n");


	// Voltage sources

	int n_vdc = dev_net->n_vdc;
	// For each add copy of the + row to the -, then 0 it
	// Then set the Vp - Vn = VDC equation
	int n_p;
	int n_n;

	if (n_vdc > 0) {
		numBlocks = ceil(float(n) / float(BS_1D));
		// We're assuming more nodes than voltage sources, generally correct
		for (int i = 0; i < n_vdc; i++) {
			n_p = netlist->vdcList[i].nodes[0] - 1;
			n_n = netlist->vdcList[i].nodes[1] - 1;
			if (n_p >= 0 && n_n >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, dev_gMat + n * n_n, dev_gMat + n * n_p, dev_iMat + n_n, dev_iMat + n_p);
			else if (n_p >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, NULL, dev_gMat + n * n_p, NULL, dev_iMat + n_p);
			else if (n_n >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, NULL, dev_gMat + n * n_n, NULL, dev_iMat + n_n);
		}
		numBlocks = ceil(float(n_vdc) / float(BS_1D));
		kernVDCtoMat << < numBlocks, BS_1D >> >(n_vdc, n, dev_net->vdcList, dev_gMat, dev_iMat, dev_vMat);
	}

}




void gpuNetlistToMat(CUDA_Net* dev_net, Netlist* netlist, float* dev_gMat, float* dev_iMat, float* dev_vMat, float* dev_vGuess) {
	int n = dev_net->n_nodes;
	if (n == 0) return;


	int n_passive = dev_net->n_passive;

	int numBlocks = ceil(float(n_passive) / float(BS_1D));

	// Passives

	if (n_passive > 0) kernDCPassiveMat << < numBlocks, BS_1D >> >(n_passive, n, dev_net->passives, dev_gMat, dev_iMat);
	checkCUDAError("Matrix Gen Kernel Failure!\n");


	// Transistors

	int n_active = dev_net->n_active;
	if (n_active > 0) {

		kernelMOStoMat<<<numBlocks, BS_1D>>>(n_active, n, dev_net->actives, dev_net->modelList, dev_gMat, dev_iMat, dev_vGuess);
	}


	// Voltage sources

	int n_vdc = dev_net->n_vdc;
	// For each add copy of the + row to the -, then 0 it
	// Then set the Vp - Vn = VDC equation
	int n_p;
	int n_n;

	if (n_vdc > 0) {
		// We're assuming more nodes than voltage sources, generally correct
		numBlocks = ceil(float(n) / float(BS_1D));
		for (int i = 0; i < n_vdc; i++) {
			n_p = netlist->vdcList[i].nodes[0] - 1;
			n_n = netlist->vdcList[i].nodes[1] - 1;
			if (n_p >= 0 && n_n >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, dev_gMat + n * n_n, dev_gMat + n * n_p, dev_iMat + n_n, dev_iMat + n_p);
			else if (n_p >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, NULL, dev_gMat + n * n_p, NULL, dev_iMat + n_p);
			else if (n_n >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, NULL, dev_gMat + n * n_n, NULL, dev_iMat + n_n);
			checkCUDAError("vdc setup failed!");
		}
		numBlocks = ceil(float(n_vdc) / float(BS_1D));
		kernVDCtoMat << < numBlocks, BS_1D >> >(n_vdc, n, dev_net->vdcList, dev_gMat, dev_iMat, dev_vMat);
		checkCUDAError("vdc setup fail!");
	}

}
