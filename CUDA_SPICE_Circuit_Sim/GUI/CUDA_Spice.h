#pragma once
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <ratio>
#include <chrono>
#include "sim.h"
#include "cuda_sim.h"



using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

public ref class CUDA_Spice
{
public:

private: CUDA_Net* dev_net = NULL;
private: Netlist* net = NULL;
public: float** output = NULL;
public: int n_rows = 0;
public: int n_col = 0;

public: 
	
	int genNetlists(String^ fileString) {
		if (dev_net != NULL) free(dev_net);
		if (net != NULL) free(net);

		dev_net = new CUDA_Net();
		net = new Netlist();

		char* filepath = (char*)malloc(fileString->Length * sizeof(char));

		sprintf(filepath, "%s", fileString);

		int r = parseNetlist(filepath, net);
		if (r != 0) {


			//free(filepath);
			return r;
		}
		gpuNetlist(net, dev_net);
		//free(filepath);
		return 0;
	}

	void guiOP() {

		output = (float**)malloc(sizeof(float*));
		n_rows =  fullCudaOP_Out(net, dev_net, output);
		n_col = net->netNames.size() - 1;
	}
	
	void outToCSV(String^ outString) {
		char* filepath = (char*)malloc(outString->Length * sizeof(char));
		sprintf(filepath, "%s", outString);

		mat2DtoCSV(net->netNames.data() + 1, output, n_rows, n_col, filepath);
	}

};
