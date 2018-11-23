#include "cuda_LinSolver.h"

#define BS_X 32
#define BS_Y 32

// k is the current row being used to reduce the rest
__global__ void kernMatReduce(int n, float* gMat, float* iMat, int k) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;  // Row index
	int j = blockDim.y * blockIdx.y + threadIdx.y;  // Column index

	// keep in matrix bounds
	// matrix always square
	// extra column for iMat
	if (i >= n || j > n) return;
	if (i == k) return; // skip reference row

	int ref_idx = k * n;
	// error, need to return somehow?
	if (gMat[ref_idx + k] == 0) return;

	int idx = i * n;

	float ratio = gMat[idx + k] / gMat[ref_idx + k];

	if (j == n) {
		iMat[i] -= ratio * iMat[k];
		return;
	}

	gMat[idx + j] -= ratio * gMat[ref_idx + j];
}

__global__ void kernPlugKnownV(int n, float* gMat, float* iMat, float* vMat) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;  // Row index
	int j = blockDim.y * blockIdx.y + threadIdx.y;  // Column index

	if (i >= n || j >= n) return;
	
	float v = vMat[j];
	float g = gMat[i * n + j];
	if (v != 0.0f && g != 0.0f) {
		float c = -g * v;
		atomicAdd(iMat + i, c);
	}
}

__global__ void kernMatSolve(int n, float* gMat, float* iMat, float* vMat) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;  // Row index

	// keep in matrix bounds
	// matrix always square
	if (i >= n) return;
	if (vMat[i] != 0.0f) return;

	// error?
	if (gMat[i * n + i] == 0) return;

	float v = iMat[i] / gMat[i * n + i];
	vMat[i] = v;
}

void gpuMatReduce(int n, float* dev_gMat, float* dev_iMat) {

	int numBlocks = ceil(float(n) / 32.0f);

	dim3 numBlocks3D = dim3(numBlocks, numBlocks, 1);
	dim3 blockSize = dim3(BS_X, BS_Y, 1);

	
	for (int i = 0; i < n; i++) {
		kernMatReduce << < numBlocks3D, blockSize >> > (n, dev_gMat, dev_iMat, i);
	}
	checkCUDAError("Reduction Failure!\n");
}

void gpuMatSolve(int n, float** gMat, float* iMat, float* vMat) {

	float* dev_gMat = NULL;
	float* dev_vMat = NULL;
	float* dev_iMat = NULL;

	//setupDevMats(n, gMat, dev_gMat, iMat, dev_iMat, dev_vMat);
	// alloc device memory
	cudaMalloc((void**)&dev_iMat, n * sizeof(float));
	cudaMalloc((void**)&dev_gMat, n * n * sizeof(float));
	cudaMalloc((void**)&dev_vMat, n * sizeof(float));
	checkCUDAError("Malloc Failure!\n");

	// copy host to device
	for (int i = 0; i < n; i++) {
		cudaMemcpy(dev_gMat + i*n, gMat[i], n * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("Host gMat MemCpy Failure!\n");
	}
	cudaMemcpy(dev_iMat, iMat, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("Host iMat MemCpy Failure!\n");
	cudaMemcpy(dev_vMat, vMat, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("Host vMat MemCpy Failure!\n");


	int numBlocks = ceil(float(n) / BS_X);

	dim3 numBlocks3D = dim3(numBlocks, numBlocks, 1);
	dim3 blockSize = dim3(BS_X, BS_Y, 1);


	for (int i = 0; i < n; i++) {
		kernMatReduce << < numBlocks3D, blockSize >> > (n, dev_gMat, dev_iMat, i);

		checkCUDAError("Reduction Failure!\n");
	}
	cudaDeviceSynchronize();

	kernPlugKnownV << < numBlocks3D, blockSize >> > (n, dev_gMat, dev_iMat, dev_vMat);

	kernMatSolve<<<numBlocks, BS_X>>>(n, dev_gMat, dev_iMat, dev_vMat);
	checkCUDAError("Solution Failure!\n");

	cudaDeviceSynchronize();

	//copyDevMats(n, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);
	for (int i = 0; i < n; i++) {
		cudaMemcpy(gMat[i], dev_gMat + i*n, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("Device gMat MemCpy Failure!\n");
	}

	cudaMemcpy(vMat, dev_vMat, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Device vMat MemCpy Failure!\n");

	cudaMemcpy(iMat, dev_iMat, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Device iMat MemCpy Failure!\n");

	cleanDevMats(dev_gMat, dev_iMat, dev_vMat);
}