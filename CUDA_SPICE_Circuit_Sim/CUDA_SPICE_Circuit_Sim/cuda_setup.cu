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
	system("pause");
	exit(EXIT_FAILURE);
}


// Mallocs arrays and copies host data to device
void setupDevMats(int n, float** gMat, float* &dev_gMat, float* iMat, float* &dev_iMat, float* vMat, float* &dev_vMat) {

	// alloc device memory
	cudaMalloc((void**)&dev_iMat, n * sizeof(float));
	cudaMalloc((void**)&dev_gMat, n * n * sizeof(float));
	cudaMalloc((void**)&dev_vMat, n * sizeof(float));
	checkCUDAError("Malloc Failure!\n");

	copyToDevMats(n, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);
}


// copy data to host
void copyFromDevMats(int n, float** gMat, float* dev_gMat, float* iMat, float* dev_iMat, float* vMat, float* dev_vMat) {
	if (gMat != NULL) {
		for (int i = 0; i < n; i++) {
			cudaMemcpy(gMat[i], dev_gMat + i*n, n * sizeof(float), cudaMemcpyDeviceToHost);
			checkCUDAError("Device gMat MemCpy Failure!\n");
		}
	}
	if (vMat != NULL) {
		cudaMemcpy(vMat, dev_vMat, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("Device vMat MemCpy Failure!\n");
	}
	if (iMat != NULL) {
		cudaMemcpy(iMat, dev_iMat, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("Device iMat MemCpy Failure!\n");
	}
}

void copyToDevMats(int n, float** gMat, float* dev_gMat, float* iMat, float* dev_iMat, float* vMat, float* dev_vMat) {
	if (gMat != NULL) {// && dev_gMat != NULL) {
		for (int i = 0; i < n; i++) {
			cudaMemcpy(dev_gMat + i*n, gMat[i], n * sizeof(float), cudaMemcpyHostToDevice);
			checkCUDAError("Host gMat MemCpy Failure!\n");
		}
	}
	if (iMat != NULL) {// && dev_iMat != NULL) {
		cudaMemcpy(dev_iMat, iMat, n * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("Host iMat MemCpy Failure!\n");
	}
	if (vMat != NULL) {// && dev_vMat != NULL) {
		cudaMemcpy(dev_vMat, vMat, n * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("Host vMat MemCpy Failure!\n");
	}
}

void cleanDevMats(float* dev_gMat, float* dev_iMat, float* dev_vMat) {
	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
}