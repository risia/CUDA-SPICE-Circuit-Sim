#include "cuda_kcl.h"

// Parallelize R, IDC, and VCCS list by element
__global__ void kernDCPassiveMat(int n, CUDA_Elem* passives, float* gMat, float* iMat) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index

	if (idx >= n) return;
	
	//CUDA_Elem e = passives[idx];
	//char type = e.type;
	/*
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

	// test
	char test;
	cudaMemcpy(&test, &(dev_net->passives->type), sizeof(char), cudaMemcpyDeviceToHost);
	printf("TEST: %c\n\n", test);
	checkCUDAError("Device Passive MemCpy Failure!\n");

	int n_passive = dev_net->n_passive;
	if (n_passive == 0) return;

	int numBlocks = ceil(float(n_passive) / 64.0f);

	dim3 numBlocks3D = dim3(numBlocks, 1, 1);
	dim3 blockSize = dim3(64.0f, 1, 1);

	//kernDCPassiveMat << < numBlocks3D, blockSize >> >(n_passive, dev_net->passives, dev_gMat, dev_iMat);
	//checkCUDAError("Matrix Gen Kernel Failure!\n");

	

	/*
	cudaMemcpy(iMat, dev_iMat, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Device iMat MemCpy Failure!\n");


	for (int i = 0; i < n; i++) {
		cudaMemcpy(gMat[i], dev_gMat + i*n, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("Device gMat MemCpy Failure!\n");
	}
	*/

	cudaFree(dev_gMat);
	cudaFree(dev_iMat);

	checkCUDAError("Device Matrix Free Failure!\n");
}
