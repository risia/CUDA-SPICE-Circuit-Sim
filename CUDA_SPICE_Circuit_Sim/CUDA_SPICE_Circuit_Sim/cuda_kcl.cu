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
	}

	// If it's shorted, no contribution
	if (a == b ) return;

	// DC Current Source
	if (type == 'I') {
		if (a >= 0) atomicAdd(iMat + a, -val);
		if (b >= 0) atomicAdd(iMat + b, val);
		return;
	}



	/*
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
	*/
}


void kernVDCtoMat(int n, Element* elems, float* gMat, float* iMat) {
	// 
}

void gpuNetlistToMat(CUDA_Net* dev_net, float** gMat, float* iMat) {
	float* dev_gMat = NULL;
	float* dev_iMat = NULL;

	int n = dev_net->n_nodes;
	if (n == 0) return;
	

	// alloc device memory
	cudaMalloc((void**)&dev_iMat, n * sizeof(float));
	cudaMalloc((void**)&dev_gMat, n * n * sizeof(float));
	checkCUDAError("Malloc Failure!\n");

	cudaMemset(dev_iMat, 0, n * sizeof(float));
	cudaMemset(dev_gMat, 0, n * n * sizeof(float));
	checkCUDAError("Memset Failure!\n");

	int n_passive = dev_net->n_passive;
	if (n_passive == 0) return;

	int numBlocks = ceil(float(n_passive) / 64.0f);

	dim3 numBlocks3D = dim3(numBlocks, 1, 1);
	dim3 blockSize = dim3(64.0f, 1, 1);

	CUDA_Elem e;
	float val;
	int node;
	cout << "TEST:\n";
	for (int i = 0; i < n_passive; i++) {
		cudaMemcpy(&e, dev_net->passives + i, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
		cout << "Element Type: " << e.type;
		cudaMemcpy(&val, e.params, sizeof(float), cudaMemcpyDeviceToHost);
		cout << "\nVal: " << val;
		cudaMemcpy(&node, e.nodes, sizeof(int), cudaMemcpyDeviceToHost);
		cout << "\nNodes: " << node << " ";
		cudaMemcpy(&node, e.nodes + 1, sizeof(int), cudaMemcpyDeviceToHost);
		cout << node << "\n";
	}

	
	
	kernDCPassiveMat << < numBlocks3D, blockSize >> >(n_passive, n, dev_net->passives, dev_gMat, dev_iMat);
	checkCUDAError("Matrix Gen Kernel Failure!\n");

	

	
	copyFromDevMats(n, gMat, dev_gMat, iMat, dev_iMat, NULL, NULL);

	cudaFree(dev_gMat);
	cudaFree(dev_iMat);

	checkCUDAError("Device Matrix Free Failure!\n");
}
