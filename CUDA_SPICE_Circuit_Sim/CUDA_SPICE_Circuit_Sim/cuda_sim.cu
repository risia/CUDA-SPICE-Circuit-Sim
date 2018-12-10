#include "cuda_sim.h"

__global__ void kernelMatCmp(int n, float* mat1, float* mat2, int* isSame) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n) return;

	if (fabs((mat1[idx] - mat2[idx])) > TOL) {
		isSame[0] = 0;
	}
}

__global__ void kernelFindElem(int n_elem, CUDA_Elem* elems, char* name, int* e_id) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index
	if (idx >= n_elem) return;
	if (e_id[0] != -1) return; // already found element

	CUDA_Elem* e = elems + idx;

	bool matched = true;
	for (int i = 0; name[i] != '\0' && e->name[i] != '\0'; i++) {
		if (name[i] != e->name[i]) matched = false;
		if (name[i + 1] == '\0' && e->name[i] != '\0') matched = false;
		if (name[i + 1] != '\0' && e->name[i] == '\0') matched = false;
	}

	if (matched == true) e_id[0] = idx;
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

void full_cudaDCSweep(Netlist* net, CUDA_Net* dev_net, char* name, float start, float stop, float step) {
	int n_nodes = dev_net->n_nodes;
	int n_blocks = ceil(float(n_nodes) / float(BS_1D));

	// Find swept element
	CUDA_Elem e;
	int id = -1;

	for (int i = 0; i < dev_net->n_vdc && id == -1; i++) {
		if (strcmp(net->vdcList[i].name, name) == 0) {
			id = i;
			cudaMemcpy(&e, dev_net->vdcList + id, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
			checkCUDAError("Element retrieval Failed!\n");
		}
	}
	for (int i = 0; i < dev_net->n_passive && id == -1; i++) {
		if (strcmp(net->elements[i].name, name) == 0) {
			id = i;
			cudaMemcpy(&e, dev_net->passives + id, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
			checkCUDAError("Element retrieval Failed!\n");
		}
	}
	for (int i = 0; i < dev_net->n_active && id == -1; i++) {
		if (strcmp(net->active_elem[i].name, name) == 0) {
			id = i;
			cudaMemcpy(&e, dev_net->actives + id, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
			checkCUDAError("Element retrieval Failed!\n");
		}
	}

	if (id == -1) {
		cout << "Element NOT found!\n";
		return;
	}

	float* dev_param = e.params;
	

	// Save original value and set
	float* orig_val;
	cudaMalloc((void**)&orig_val, sizeof(float));
	checkCUDAError("Orig. val malloc error!\n");
	cudaMemcpy(orig_val, dev_param, sizeof(float), cudaMemcpyDeviceToDevice);
	checkCUDAError("Orig. val cpy error!\n");

	float cur_val = start;



	// GPU Matrices
	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;
	float* dev_vGuess = NULL;

	setupDevMats(n_nodes, NULL, dev_gMat, NULL, dev_iMat, NULL, dev_vMat);
	cudaMalloc((void**)&dev_vGuess, n_nodes * sizeof(float));

	int n_steps = floor(1 + (stop - start) / step);
	if (stop > (start + step * n_steps)) n_steps++;

	// CPU Matrices for Output
	float** vSweepMat = mat2D(n_steps, n_nodes + 1);


	int isConverged = 0;
	int* dev_isConv;
	cudaMalloc((void**)&dev_isConv, sizeof(int));
	checkCUDAError("convergence bool malloc failed!");
	
	int n = 0;

	// loop steps
	for (int i = 0; i < n_steps; i++) {
		vSweepMat[i][0] = cur_val;
		cudaMemcpy(dev_param, &cur_val, sizeof(float), cudaMemcpyHostToDevice);

		// Reset matrices
		cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
		cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
		cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
		checkCUDAError("Memory reset failed!");

		// Solve Just Passives
		gpuPassiveVDCToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat);
		gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);

		/*
		Transistor convergence loop
		*/
		n = 0;
		isConverged = 0;
		if (dev_net->n_active == 0) isConverged = 1;

		while (isConverged == 0 && n < 1000) {
			cudaMemcpy(dev_vGuess, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

			// Reset matrices
			cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
			cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
			cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
			checkCUDAError("Memory reset failed!");

			cudaMemset(dev_isConv, 1, sizeof(int));
			isConverged = 1;
			checkCUDAError("Convergence reset failed!");

			gpuNetlistToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat, dev_vGuess);
			checkCUDAError("Netlist to matrix failed!");

			// Attempt solution
			gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);
			checkCUDAError("Solution failed!");

			// Measure error beteen old and new guess
			//isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
			kernelMatCmp << <n_blocks, BS_1D >> > (n_nodes, dev_vGuess, dev_vMat, dev_isConv);
			checkCUDAError("vMat comparison failed!");
			cudaMemcpy(&isConverged, dev_isConv, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy failed!");

			// Iteration counter
			n++;
		}

		cudaMemcpy(vSweepMat[i] + 1, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToHost);

		if (i == n_steps - 2) cur_val = stop;
		else cur_val += step;
	}
	// restore element parameter val
	cudaMemcpy(dev_param, orig_val, sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(orig_val);

	char** names = net->netNames.data();
	names[0] = name;
	mat2DtoCSV(names, vSweepMat, n_steps, n_nodes + 1, "dcSweep.csv");
	names[0] = "gnd";

	// cleanup
	freeMat2D(vSweepMat, n_steps);
	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
	cudaFree(dev_vGuess);
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

int fullCudaSweep_Out(Netlist* net, CUDA_Net* dev_net, char* name, float start, float stop, float step, int n_steps, float** vOut) {
	int n_nodes = dev_net->n_nodes;
	int n_blocks = ceil(float(n_nodes) / float(BS_1D));

	// Find swept element
	CUDA_Elem e;
	int id = -1;

	for (int i = 0; i < dev_net->n_vdc && id == -1; i++) {
		if (strcmp(net->vdcList[i].name, name) == 0) {
			id = i;
			cudaMemcpy(&e, dev_net->vdcList + id, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
			checkCUDAError("Element retrieval Failed!\n");
		}
	}
	for (int i = 0; i < dev_net->n_passive && id == -1; i++) {
		if (strcmp(net->elements[i].name, name) == 0) {
			id = i;
			cudaMemcpy(&e, dev_net->passives + id, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
			checkCUDAError("Element retrieval Failed!\n");
		}
	}
	for (int i = 0; i < dev_net->n_active && id == -1; i++) {
		if (strcmp(net->active_elem[i].name, name) == 0) {
			id = i;
			cudaMemcpy(&e, dev_net->actives + id, sizeof(CUDA_Elem), cudaMemcpyDeviceToHost);
			checkCUDAError("Element retrieval Failed!\n");
		}
	}

	if (id == -1) {
		cout << "Element NOT found!\n";
		return id;
	}

	float* dev_param = e.params;


	// Save original value and set
	float* orig_val;
	cudaMalloc((void**)&orig_val, sizeof(float));
	checkCUDAError("Orig. val malloc error!\n");
	cudaMemcpy(orig_val, dev_param, sizeof(float), cudaMemcpyDeviceToDevice);
	checkCUDAError("Orig. val cpy error!\n");

	float cur_val = start;



	// GPU Matrices
	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;
	float* dev_vGuess = NULL;

	setupDevMats(n_nodes, NULL, dev_gMat, NULL, dev_iMat, NULL, dev_vMat);
	cudaMalloc((void**)&dev_vGuess, n_nodes * sizeof(float));

	// CPU Matrices for Output
	float* vSweepMat;


	int isConverged = 0;
	int* dev_isConv;
	cudaMalloc((void**)&dev_isConv, sizeof(int));
	checkCUDAError("convergence bool malloc failed!");

	int n = 0;

	// loop steps
	for (int i = 0; i < n_steps; i++) {
		vSweepMat = mat1D(n_nodes + 1);
		vOut[i] = vSweepMat;
		vSweepMat[0] = cur_val;
		cudaMemcpy(dev_param, &cur_val, sizeof(float), cudaMemcpyHostToDevice);

		// Reset matrices
		cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
		cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
		cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
		checkCUDAError("Memory reset failed!");

		// Solve Just Passives
		gpuPassiveVDCToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat);
		gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);

		/*
		Transistor convergence loop
		*/
		n = 0;
		isConverged = 0;
		if (dev_net->n_active == 0) isConverged = 1;

		while (isConverged == 0 && n < 1000) {
			cudaMemcpy(dev_vGuess, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

			// Reset matrices
			cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
			cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
			cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
			checkCUDAError("Memory reset failed!");

			cudaMemset(dev_isConv, 1, sizeof(int));
			isConverged = 1;
			checkCUDAError("Convergence reset failed!");

			gpuNetlistToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat, dev_vGuess);
			checkCUDAError("Netlist to matrix failed!");

			// Attempt solution
			gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);
			checkCUDAError("Solution failed!");

			// Measure error beteen old and new guess
			//isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
			kernelMatCmp << <n_blocks, BS_1D >> > (n_nodes, dev_vGuess, dev_vMat, dev_isConv);
			checkCUDAError("vMat comparison failed!");
			cudaMemcpy(&isConverged, dev_isConv, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy failed!");

			// Iteration counter
			n++;
		}

		cudaMemcpy(vSweepMat + 1, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToHost);

		if (i == n_steps - 2) cur_val = stop;
		else cur_val += step;
	}
	// restore element parameter val
	cudaMemcpy(dev_param, orig_val, sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(orig_val);
	cudaFree(dev_gMat);
	cudaFree(dev_iMat);
	cudaFree(dev_vMat);
	cudaFree(dev_vGuess);

	return n_steps;
}

int fullCudaTran_Out(Netlist* net, CUDA_Net* dev_net, float start, float stop, float step, int n_steps, int skipped_steps, float** vOut) {
	
	int n_nodes = dev_net->n_nodes;
	int n_blocks = ceil(float(n_nodes) / float(BS_1D));

	float time = 0.0f;

	// GPU Matrices
	float* dev_gMat = NULL;
	float* dev_iMat = NULL;
	float* dev_vMat = NULL;
	float* dev_vGuess = NULL;
	float* dev_vPrev = NULL;

	setupDevMats(n_nodes, NULL, dev_gMat, NULL, dev_iMat, NULL, dev_vMat);
	cudaMalloc((void**)&dev_vGuess, n_nodes * sizeof(float));
	cudaMalloc((void**)&dev_vPrev, n_nodes * sizeof(float));

	float* vSweepMat = mat1D(n_nodes + 1);


	// Solve for t = 0s
	// Solve Just Passives
	gpuPassiveVDCToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat);
	gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);

	int isConverged = 0;
	int* dev_isConv;
	cudaMalloc((void**)&dev_isConv, sizeof(int));
	checkCUDAError("convergence bool malloc failed!");

	int n = 0;
	while (isConverged == 0 && n < 1000) {
		cudaMemcpy(dev_vGuess, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

		// Reset matrices
		cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
		cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
		cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
		checkCUDAError("Memory reset failed!");

		cudaMemset(dev_isConv, 1, sizeof(int));
		isConverged = 1;
		checkCUDAError("Convergence reset failed!");

		gpuNetlistToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat, dev_vGuess);
		checkCUDAError("Netlist to matrix failed!");

		// Attempt solution
		gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);
		checkCUDAError("Solution failed!");

		// Measure error beteen old and new guess
		//isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
		kernelMatCmp << <n_blocks, BS_1D >> > (n_nodes, dev_vGuess, dev_vMat, dev_isConv);
		checkCUDAError("vMat comparison failed!");
		cudaMemcpy(&isConverged, dev_isConv, sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError("Memcpy failed!");

		// Iteration counter
		n++;
	}
	// Previous timestep voltage Matrix
	cudaMemcpy(dev_vPrev, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

	for (int t = 0; t < n_steps; t++) {
		// Reset matrices
		cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
		cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
		cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
		checkCUDAError("Memory reset failed!");

		// Solve Just Passives
		gpuTranPassVToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat, dev_vPrev, time, step);
		gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);

		/*
		Transistor convergence loop
		*/
		n = 0;
		isConverged = 0;
		if (dev_net->n_active == 0) isConverged = 1;

		while (isConverged == 0 && n < 1000) {
			cudaMemcpy(dev_vGuess, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToDevice);

			// Reset matrices
			cudaMemset(dev_gMat, 0, n_nodes * n_nodes * sizeof(float));
			cudaMemset(dev_iMat, 0, n_nodes * sizeof(float));
			cudaMemset(dev_vMat, 0, n_nodes * sizeof(float));
			checkCUDAError("Memory reset failed!");

			cudaMemset(dev_isConv, 1, sizeof(int));
			isConverged = 1;
			checkCUDAError("Convergence reset failed!");

			gpuTranNetToMat(dev_net, net, dev_gMat, dev_iMat, dev_vMat, dev_vGuess, dev_vPrev, time, step);
			checkCUDAError("Netlist to matrix failed!");

			// Attempt solution
			gpuDevMatSolve(n_nodes, dev_gMat, dev_iMat, dev_vMat);
			checkCUDAError("Solution failed!");

			// Measure error beteen old and new guess
			//isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
			kernelMatCmp << <n_blocks, BS_1D >> > (n_nodes, dev_vGuess, dev_vMat, dev_isConv);
			checkCUDAError("vMat comparison failed!");
			cudaMemcpy(&isConverged, dev_isConv, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy failed!");

			// Iteration counter
			n++;
		}


		// copy solution to output
		if (t >= skipped_steps) {
			vSweepMat = mat1D(n_nodes + 1);
			vSweepMat[0] = time;
			cudaMemcpy(vSweepMat + 1, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToHost);
			vOut[t - skipped_steps] = vSweepMat;
		}
		cudaMemcpy(dev_vPrev, dev_vMat, n_nodes * sizeof(float), cudaMemcpyDeviceToDevice);


		if (t == n_steps - 2) time = stop;
		else if (t == skipped_steps - 1) time = start;
		else time += step;
	}

	// Cleanup
	cleanDevMats(dev_gMat, dev_iMat, dev_vMat);
	cudaFree(dev_vGuess);
	cudaFree(dev_vPrev);

	return n_steps;
}