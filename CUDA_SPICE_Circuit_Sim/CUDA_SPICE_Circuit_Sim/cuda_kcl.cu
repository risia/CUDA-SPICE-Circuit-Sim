#include "cuda_kcl.h"

// Parallelize R, IDC, and VCCS list by element
__global__ void kernDCPassiveMat(int n, int n_nodes, CUDA_Elem* passives, float* gMat, float* iMat) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index

	if (idx >= n) return;
	
	CUDA_Elem* e = passives + idx;
	char type = e->type;
	
	int a = e->nodes[0] - 1;
	int b = e->nodes[1] - 1;

	
	float val = e->params[0];
	if (type == 'R') {
		val = 1.0f / val;
		/*
		if (a >= 0) {
			atomicAdd(gMat + (a * n_nodes + a), val);
		}
		if (b >= 0) {
			atomicAdd(gMat + (b * n_nodes + b), val);
		}
		if (a >= 0 && b >= 0) {
			val = -val;
			atomicAdd(gMat + (a * n_nodes + b), val);
			atomicAdd(gMat + (b * n_nodes + a), val);
		}
		*/
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
	int gidx = a * n_nodes + c;
	if (a >= 0 && c >= 0) atomicAdd(gMat + gidx, val);

	gidx = b * n_nodes + d;
	if (b >= 0 && d >= 0) atomicAdd(gMat + gidx, val);

	if (b >= 0 && a >= 0 && d >= 0 && c >= 0) {
		gidx = a * n_nodes + d;
		atomicAdd(gMat + gidx, -val);

		gidx = b * n_nodes + c;
		atomicAdd(gMat + gidx, -val);
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
void gpuNetlistToMat(CUDA_Net* dev_net, Netlist* netlist, float** gMat, float* iMat, float* vMat) {
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
		for (int i = 0; i < n_vdc; i++) {
			n_p = netlist->vdcList[i].nodes[0] - 1;
			n_n = netlist->vdcList[i].nodes[1] - 1;
			if (n_p >= 0 && n_n >= 0) kernelAddandZero << < numBlocks3D, blockSize >> > (n, dev_gMat + n * n_n, dev_gMat + n * n_p, dev_iMat + n_n, dev_iMat + n_p);
			else if (n_p >= 0) kernelAddandZero << < numBlocks3D, blockSize >> > (n, NULL, dev_gMat + n * n_p, NULL, dev_iMat + n_p);
			else if (n_n >= 0) kernelAddandZero << < numBlocks3D, blockSize >> > (n, NULL, dev_gMat + n * n_n, NULL, dev_iMat + n_n);
		}


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

	int numBlocks = ceil(float(n_passive) / BS_1D);

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
		// We're assuming more nodes than voltage sources, generally correct
		for (int i = 0; i < n_vdc; i++) {
			n_p = netlist->vdcList[i].nodes[0] - 1;
			n_n = netlist->vdcList[i].nodes[1] - 1;
			if (n_p >= 0 && n_n >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, dev_gMat + n * n_n, dev_gMat + n * n_p, dev_iMat + n_n, dev_iMat + n_p);
			else if (n_p >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, NULL, dev_gMat + n * n_p, NULL, dev_iMat + n_p);
			else if (n_n >= 0) kernelAddandZero << < numBlocks, BS_1D >> > (n, NULL, dev_gMat + n * n_n, NULL, dev_iMat + n_n);
		}

		kernVDCtoMat << < numBlocks, BS_1D >> >(n_vdc, n, dev_net->vdcList, dev_gMat, dev_iMat, dev_vMat);
	}

}
