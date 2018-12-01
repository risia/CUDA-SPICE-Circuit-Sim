#pragma once
#include "spice.h"
#include "cuda_setup.h"
#include <algorithm>

#ifndef MODEL_STRUCT
#define MODEL_STRUCT
struct Model {
	char* name = "N";
	char type = 'n';

	float u0 = 540.0f;
	float tox = 14.1e-9;
	float vt0 = 0.7f;
	float pclm = 0.6171774f; // CLM parameter
};
#endif // !1

struct CUDA_Elem {
	char type;
	int* nodes;
	float* params;
	int model; // model index if transistor
};

struct CUDA_Net {
	int n_nodes;
	//int n_models;
	int n_passive;
	int n_vdc;
	int n_active;
	Model* modelList;
	CUDA_Elem* passives;
	CUDA_Elem* vdcList;
	CUDA_Elem* actives;
};

__global__ void kernElementPointers(int n, CUDA_Elem* elemList, int** nodeLists, float** paramLists, int* modelIdx, char* type);

int findModelN(Model** modelList, char* name, int n);
void gpuElementCpy(int n, Element* elemList, CUDA_Elem* dev_elemList, Model** modelList, int num_models);

// Copy netlist to GPU
void gpuNetlist(Netlist* netlist, CUDA_Net* dev_net);
void freeGpuNetlist(CUDA_Net* dev_net);