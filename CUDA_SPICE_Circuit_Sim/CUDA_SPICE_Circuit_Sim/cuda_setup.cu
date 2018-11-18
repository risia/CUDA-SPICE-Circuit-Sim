#include "cuda_setup.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

// Mallocs arrays and copies host data to device
void setupDevMats(int n, float** gMat, float* dev_gMat, float* iMat, float* dev_iMat, float* dev_vMat) {

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

	cudaDeviceSynchronize();
}

// copy data to host
void copyDevMats(int n, float** gMat, float* dev_gMat, float* iMat, float* dev_iMat, float* vMat, float* dev_vMat) {

	for (int i = 0; i < n; i++) {
		cudaMemcpy(gMat[i], dev_gMat + i*n, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("Device gMat MemCpy Failure!\n");
	}

	cudaMemcpy(vMat, dev_vMat, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Device vMat MemCpy Failure!\n");

	cudaMemcpy(iMat, dev_iMat, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Device iMat MemCpy Failure!\n");

	cudaDeviceSynchronize();
}

void cleanDevMats(float* dev_gMat, float* dev_iMat, float* dev_vMat) {
	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
}