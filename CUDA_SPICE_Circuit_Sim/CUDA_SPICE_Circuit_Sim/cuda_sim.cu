#include "cuda_sim.h"

__global__ void kernelMatCmp(int n, float* mat1, float* mat2, int* isSame) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n) return;

	if (fabs((mat1[idx] - mat2[idx])) > TOL) {
		isSame[0] = 0;
	}
}

__global__ void kernelFindElem(int n_elem, CUDA_Elem* elems, char* name, int* elem_id) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n_elem) return;
	if (elem_id[0] != -1) return; // already found element

	CUDA_Elem* e = elems + idx;

	bool matched = true;
	for (int i = 0; name[i] != '\0' && e->name[i] != '\0'; i++) {
		if (name[i] != e->name[i]) matched = false;
		if (name[i + 1] == '\0' && e->name[i] != '\0') matched = false;
		if (name[i + 1] != '\0' && e->name[i] == '\0') matched = false;
	}

	if (matched == true) elem_id[0] = idx;
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
	
	/*
	cout << "\n*******************\n" << "CUDA Netlist Test" << "\n*******************\n";
	cout << mat2DToStr(gMat, num_nodes, num_nodes) << "\n" << mat1DToStr(iMat, num_nodes) << "\n" << mat1DToStr(vMat, num_nodes);
	*/


	free(iMat);
	free(vMat);
	freeMat2D(gMat, num_nodes);


	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
	cudaFree(dev_vGuess);
}

void full_cudaDCSweep(Netlist* net, CUDA_Net* dev_net, char* name, float start, float stop, float stepq) {
	int n_nodes = dev_net->n_nodes;
	int n_blocks = ceil(float(n_nodes) / float(BS_1D));

	// CPU Matrices for Output
	float** gMat = mat2D(n_nodes, n_nodes);
	float* iMat = mat1D(n_nodes);
	float* vMat = mat1D(n_nodes);

	// GPU Matrices
	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;
	float* dev_vGuess = NULL;

	setupDevMats(n_nodes, NULL, dev_gMat, NULL, dev_iMat, NULL, dev_vMat);
	cudaMalloc((void**)&dev_vGuess, n_nodes * sizeof(float));

	// Find swept element
	int* dev_id = NULL;
	cudaMalloc((void**)&dev_id, sizeof(int));
	cudaMemset(dev_id, -1, sizeof(int));

	kernelFindElem<<<n_blocks, BS_1D>>>(dev_net->n_vdc, dev_net->vdcList, name, dev_id);
	kernelFindElem << <n_blocks, BS_1D >> >(dev_net->n_passive, dev_net->passives, name, dev_id);
	kernelFindElem << <n_blocks, BS_1D >> >(dev_net->n_active, dev_net->actives, name, dev_id);
	
	// Retrieve parameter list pointer to set swept parameter


}


int fullCudaOP_Out(Netlist* net, CUDA_Net* dev_net, float** vOut) {
	int num_nodes = dev_net->n_nodes;
	int num_blocks = ceil(float(num_nodes) / float(BS_1D));

	// CPU Matrices for Output
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
		cudaMemset(dev_gMat, 0, num_nodes * num_nodes * sizeof(float));
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
	copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vMat, dev_vMat);

	vOut[0] = vMat;

	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
	cudaFree(dev_vGuess);



	return 1;
}