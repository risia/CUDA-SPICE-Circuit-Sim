#include "cuda_netlist.h"

__global__ void kernElementPointers(int n, CUDA_Elem* elemList, int** nodeLists, float** paramLists, int* modelIdx, char* type) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index

	if (idx >= n) return;

	elemList[idx].type = type[idx];
	elemList[idx].model = modelIdx[idx];
	elemList[idx].nodes = nodeLists[idx];
	elemList[idx].params = paramLists[idx];
}

int findModelN(Model** modelList, char* name, int n) {
	int i = 0;
	for (i = 0; i < n; i++) {
		if (strcmp(name, modelList[i]->name) == 0) return i;
	}
	return -1;
}
/*
__global__ void kernFreeArrays(int n_max, CUDA_Net* netlist) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // element index

	if (idx >= n_max) return;

	int n = netlist->n_passive;
	if (idx < n && n > 0) {
		cudaFree(netlist->passives[idx].nodes);
		cudaFree(netlist->passives[idx].params);
	}
	n = netlist->n_active;
	if (idx < n && n > 0) {
		cudaFree(netlist->actives[idx].nodes);
		cudaFree(netlist->actives[idx].params);
	}
	n = netlist->n_vdc;
	if (idx < n && n > 0) {
		cudaFree(netlist->vdcList[idx].nodes);
		cudaFree(netlist->vdcList[idx].params);
	}


	if (idx == 0) {
		cudaFree(netlist->modelList);
		cudaFree(netlist->passives);
		cudaFree(netlist->vdcList);
		cudaFree(netlist->actives);
		cudaFree(netlist->modelList);
	}
}
*/

void gpuElementCpy(int n, Element* elemList, CUDA_Elem* dev_elemList, Model** modelList, int num_models) {
	
	int model = 0;
	int n_nodes;
	int n_params;

	int** dev_nodeLists;
	float** dev_paramLists;
	int* dev_modelIdxs;
	char* dev_types;

	int* dev_nList;
	float* dev_pList;

	cudaMalloc((void**)&dev_nodeLists, n * sizeof(int*));
	cudaMalloc((void**)&dev_paramLists, n * sizeof(float*));
	cudaMalloc((void**)&dev_modelIdxs, n * sizeof(int));
	cudaMalloc((void**)&dev_types, n * sizeof(char));

	checkCUDAError("Temp Pointer Array Malloc Failure!\n");

	// setup element data arrays and arrays to store their pointers
	for (int i = 0; i < n; i++) {
		n_nodes = elemList[i].nodes.size();
		n_params = elemList[i].params.size();

		// allocate node list and parameter list

		cudaMalloc((void**)&(dev_nList), n_nodes * sizeof(int));
		cudaMalloc((void**)&(dev_pList), n_params * sizeof(float));
		checkCUDAError("Element Arrays Malloc Failure!\n");

		// copy from host
		cudaMemcpy(dev_nList, elemList[i].nodes.data(), n_nodes * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_pList, elemList[i].params.data(), n_params * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_types, &(elemList[i].type), sizeof(char), cudaMemcpyHostToDevice);
		checkCUDAError("Element Arrays Copy Failure!\n");

		// copy pointer to array
		cudaMemcpy((dev_nodeLists + i), &dev_nList, sizeof(int*), cudaMemcpyHostToDevice);
		cudaMemcpy((dev_paramLists + i), &dev_pList, sizeof(float*), cudaMemcpyHostToDevice);

		// find which model if MOSFET
		if (modelList != NULL) {
			model = findModelN(modelList, elemList[i].model->name, num_models);
			cudaMemcpy(dev_modelIdxs, &model, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Model Index Copy Failure!\n");
		}
	}

	// Copy Pointers to Structs
	int numBlocks = ceil(float(n) / 64.0f);

	dim3 numBlocks3D = dim3(numBlocks, 1, 1);
	dim3 blockSize = dim3(64.0f, 1, 1);

	kernElementPointers << < numBlocks3D, blockSize >> >(n, dev_elemList, dev_nodeLists, dev_paramLists, dev_modelIdxs, dev_types);
	checkCUDAError("Element Arrays Pointer Copy Failure!\n");

	cudaFree(dev_paramLists);
	cudaFree(dev_nodeLists);
	cudaFree(dev_modelIdxs);
	cudaFree(dev_types);

	checkCUDAError("Temp Pointer Array Free Failure!\n");
}

// Copy netlist to GPU
void gpuNetlist(Netlist* netlist, CUDA_Net* dev_net) {
	// allocate device netlist arrays
	dev_net->actives = NULL;
	dev_net->passives = NULL;
	dev_net->vdcList = NULL;
	dev_net->modelList = NULL;

	dev_net->n_nodes = netlist->netNames.size() - 1;

	CUDA_Elem* dev_passives;
	int n_passive = netlist->elements.size();
	dev_net->n_passive = n_passive;
	if (n_passive > 0) {
		// allocate passive elements
		cudaMalloc((void**)&dev_passives, n_passive * sizeof(CUDA_Elem));
		checkCUDAError("Passive Malloc Failure!\n");

		// copy over elements
		gpuElementCpy(n_passive, netlist->elements.data(), dev_passives, NULL, 0);
		checkCUDAError("Passive Copy Failure!\n");
	}

	int n_vdc = netlist->vdcList.size();
	dev_net->n_vdc = n_vdc;
	if (n_vdc > 0) {
		// allocate vdcs
		cudaMalloc((void**)&(dev_net->vdcList), n_vdc * sizeof(CUDA_Elem));
		checkCUDAError("VDC Malloc Failure!\n");

		// copy over vdc elements
		gpuElementCpy(n_vdc, netlist->vdcList.data(), dev_net->vdcList, NULL, 0);
		checkCUDAError("VDC Copy Failure!\n");
	}

	int n_models = netlist->modelList.size();
	int n_active = netlist->active_elem.size();
	dev_net->n_active = n_active;
	if (n_active > 0) {
		// allocate mosfets
		cudaMalloc((void**)&(dev_net->actives), n_active * sizeof(CUDA_Elem));
		checkCUDAError("MOSFET Malloc Failure!\n");

		// copy over
		gpuElementCpy(n_active, netlist->active_elem.data(), dev_net->actives, netlist->modelList.data(), n_models);
		checkCUDAError("MOSFET Copy Failure!\n");
	}
	if (n_models > 0) {
		// allocate models
		cudaMalloc((void**)&(dev_net->modelList), n_models * sizeof(Model));
		checkCUDAError("Model Malloc Failure!\n");

		// copy models
		Model** M_ptrs = netlist->modelList.data();

		//for (int i = 0; i < n_models; i++) {
			cudaMemcpy(dev_net->modelList, M_ptrs[0], sizeof(Model), cudaMemcpyHostToDevice);
		//}
		
		checkCUDAError("Model Copy Failure!\n");
	}
}

// Probably should figure out how to ensure all arrays in
// the netlist and elements actually free,
// somehow access or track all these pointers?
void freeGpuNetlist(CUDA_Net* dev_net) {
	//cudaFree(dev_net->modelList);
	
	//int n = max(max(netlist->active_elem.size(), netlist->elements.size()), netlist->vdcList.size());
	// Copy Pointers to Structs
	//int numBlocks = ceil(float(n) / 64.0f);

	//dim3 numBlocks3D = dim3(numBlocks, numBlocks, 1);
	//dim3 blockSize = dim3(64.0f, 1, 1);

	//kernFreeArrays << < numBlocks3D, blockSize >> > (n, dev_net);

	// Free Netlist
	if (dev_net->modelList != NULL) cudaFree(dev_net->modelList);
	checkCUDAError("Netlist Free Failure!\n");

	if (dev_net->actives != NULL && dev_net->n_active > 0) cudaFree(dev_net->actives);
	checkCUDAError("Netlist Free Failure!\n");

	if (dev_net->passives != NULL && dev_net->n_passive > 0) cudaFree(dev_net->passives);
	checkCUDAError("Netlist Free Failure!\n");

	if (dev_net->vdcList != NULL && dev_net->n_vdc > 0) cudaFree(dev_net->vdcList);
	checkCUDAError("Netlist Free Failure!\n");

	free(dev_net);

}