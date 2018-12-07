#pragma once
#include "spice.h"
#include "cuda_setup.h"
#include <algorithm>

#ifndef MODEL_STRUCT
#define MODEL_STRUCT
struct Model {
	char* name = "N";
	char type = 'n';

	float u0 = 540.0f; // 
	float tox = 14.1e-9; // oxide thickness
	float epsrox = 3.9f; // dielectric constant

	float vt0 = 0.7f; // threshold voltage
	float pclm = 0.6171774f; // CLM parameter
	float vsat = 8.0e4; // saturation velocity
	float nfactor = 1.0f; // subthreshold swing factor
						  //float Ld = 0.0f;

	float CGSO = 0.0f; // gate-source overlap cap per unit W
	float CGDO = 0.0f; // gate-drain overlap cap per unit W
	float CGBO = 0.0f; // gate-bulk cap per unit L
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