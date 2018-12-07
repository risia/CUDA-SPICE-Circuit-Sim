#include "cuda_sim.h"


void full_cudaOp(Netlist* net, CUDA_Net* dev_net) {
	int num_nodes = dev_net->n_nodes;

	// CPU Matrices for Output
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);

	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;

	// GPU Matrices
	setupDevMats(num_nodes, NULL, dev_gMat, NULL, dev_iMat, NULL, dev_vMat);

	// Passive Elements to Matrices
	gpuPassiveVDCToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat);

	// Solve on GPU
	gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);



	// Insert transistor convergence loop here




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
}