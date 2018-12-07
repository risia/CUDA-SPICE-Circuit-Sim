#include "cuda_sim.h"

__global__ void kernelMatCmp(int n, float* mat1, float* mat2, int* isSame) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n) return;

	if (fabs((mat1[idx] - mat2[idx])) > TOL) {
		isSame[0] = 0;
	}
}

void full_cudaOp(Netlist* net, CUDA_Net* dev_net) {
	int num_nodes = dev_net->n_nodes;
	int num_blocks = ceil(float(num_nodes) / float(BS_1D));

	// CPU Matrices for Output
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);

	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;
	float* dev_vGuess = NULL;

	// GPU Matrices
	setupDevMats(num_nodes, NULL, dev_gMat, NULL, dev_iMat, NULL, dev_vMat);
	cudaMalloc((void**)&dev_vGuess, num_nodes * sizeof(float));

	// Passive Elements to Matrices
	gpuPassiveVDCToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat);

	// Solve on GPU
	gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);


	// Insert transistor convergence loop here
	int isConverged = 0;
	int* dev_isConv;
	cudaMalloc((void**)&dev_isConv, sizeof(int));
	checkCUDAError("convergence bool malloc failed!");

	if (dev_net->n_active == 0) isConverged = 1;

	int n = 0;
	while (isConverged == 0 && n < 1000) {
		cudaMemcpy(dev_vGuess, dev_vMat, num_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

		// Reset matrices
		cudaMemset(dev_gMat, 0,  num_nodes * num_nodes * sizeof(float));
		cudaMemset(dev_iMat, 0, num_nodes * sizeof(float));
		cudaMemset(dev_vMat, 0, num_nodes * sizeof(float));
		checkCUDAError("Memory reset failed!");

		cudaMemset(dev_isConv, 1, sizeof(int));
		isConverged = 1;
		checkCUDAError("Convergence reset failed!");

		gpuNetlistToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat, dev_vGuess);
		checkCUDAError("Netlist to matrix failed!");

		// Attempt solution
		gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);
		checkCUDAError("Solution failed!");

		// Measure error beteen old and new guess
		//isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
		kernelMatCmp << <num_blocks, BS_1D >> > (num_nodes, dev_vGuess, dev_vMat, dev_isConv);
		checkCUDAError("vMat comparison failed!");
		cudaMemcpy(&isConverged, dev_isConv, sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError("Memcpy failed!");


		// Iteration counter
		n++;
	}



	// Copy to CPU for output
	copyFromDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);

	cout << "\n*******************\n" << "CUDA Netlist Test" << "\n*******************\n";
	cout << mat2DToStr(gMat, num_nodes, num_nodes) << "\n" << mat1DToStr(iMat, num_nodes) << "\n" << mat1DToStr(vMat, num_nodes);

	free(iMat);
	free(vMat);
	freeMat2D(gMat, num_nodes);


	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
	cudaFree(dev_vGuess);
}

void full_cudaDCSweep(Netlist* net, CUDA_Net* dev_net) {

}