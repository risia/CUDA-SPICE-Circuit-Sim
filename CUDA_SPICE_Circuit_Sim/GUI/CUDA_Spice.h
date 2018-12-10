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
	int checkNetlist() {
		if (net != NULL) return 0;
		else return -1;
	}
	
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

	char** guiOP() {
		if (net == NULL) return NULL;
		if (output != NULL) freeMat2D(output, n_rows);

		output = (float**)malloc(sizeof(float*));
		n_rows =  fullCudaOP_Out(net, dev_net, output);
		n_col = net->netNames.size() - 1;

		return net->netNames.data() + 1;
	}
	
	void outToCSV(String^ outString, char** labels) {
		char* filepath = (char*)malloc(outString->Length * sizeof(char));
		sprintf(filepath, "%s", outString);

		mat2DtoCSV(labels, output, n_rows, n_col, filepath);
	}

	char** guiDCSweep(String^ nameStr, String^ startStr, String^ stopStr, String^ stepStr) {
		if (net == NULL) return NULL;
		// Get the parameters for the sweep
		char* name = (char*)malloc(nameStr->Length * sizeof(char));
		sprintf(name, "%s", nameStr);

		char* startCS = (char*)malloc(startStr->Length * sizeof(char));
		sprintf(startCS, "%s", startStr);
		float start = atof(startCS);
		start = numPrefix(start, startCS[strlen(startCS) - 1]);

		char* stopCS = (char*)malloc(stopStr->Length * sizeof(char));
		sprintf(stopCS, "%s", stopStr);
		float stop = atof(stopCS);
		stop = numPrefix(stop, stopCS[strlen(stopCS) - 1]);

		char* stepCS = (char*)malloc(stepStr->Length * sizeof(char));
		sprintf(stepCS, "%s", stepStr);
		float step = atof(stepCS);
		step = numPrefix(step, stepCS[strlen(stepCS) - 1]);

		// Actually call sweep
		int n_steps = floor(1 + (stop - start) / step);
		if (stop > (start + step * n_steps)) n_steps++;

		if (output != NULL) freeMat2D(output, n_rows);

		output = (float**)malloc(n_steps * sizeof(float*));

		n_rows = fullCudaSweep_Out(net, dev_net, name, start, stop, step, n_steps, output);
		n_col = net->netNames.size();

		char** labels = net->netNames.data();
		labels[0] = name;

		return labels;
	}

	char** guiTran(String^ startStr, String^ stopStr, String^ stepStr) {
		if (net == NULL) return NULL;

		// Get the parameters for the sweep
		char* startCS = (char*)malloc(startStr->Length * sizeof(char));
		sprintf(startCS, "%s", startStr);
		float start = atof(startCS);
		start = numPrefix(start, startCS[strlen(startCS) - 1]);

		char* stopCS = (char*)malloc(stopStr->Length * sizeof(char));
		sprintf(stopCS, "%s", stopStr);
		float stop = atof(stopCS);
		stop = numPrefix(stop, stopCS[strlen(stopCS) - 1]);

		char* stepCS = (char*)malloc(stepStr->Length * sizeof(char));
		sprintf(stepCS, "%s", stepStr);
		float step = atof(stepCS);
		step = numPrefix(step, stepCS[strlen(stepCS) - 1]);

		// Actually call sweep
		int n_steps = floor(1 + (stop - start) / step);
		if (stop > (start + step * n_steps)) n_steps++;

		int skipped_steps = floor(start / step);
		if (start >(step * skipped_steps)) skipped_steps++;

		if (output != NULL) freeMat2D(output, n_rows);

		output = (float**)malloc((n_steps - skipped_steps) * sizeof(float*));

		n_rows = fullCudaTran_Out(net, dev_net, start, stop, step, n_steps, skipped_steps, output);
		n_col = net->netNames.size();

		char** labels = net->netNames.data();
		labels[0] = "Time (s)";

		return labels;
	}

};
