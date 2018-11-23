#include "main.h"

int main() {
	Netlist netlist;

	char* file = "C:/Users/Angelinia/Documents/CIS 565/CUDA-SPICE-Circuit-Sim/test_spi/T_test.spi";
	parseNetlist(file, netlist);

	Resistor* rList = netlist.rList.data();
	Vdc* vdcList = netlist.vdcList.data();
	Idc* idcList = netlist.idcList.data();
	VCCS* vccsList = netlist.vccsList.data();
	Transistor* mosList = netlist.mosList.data();

	int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_r = netlist.rList.size();
	int num_vdc = netlist.vdcList.size();
	int num_idc = netlist.idcList.size();
	int num_vccs = netlist.vccsList.size();
	int num_mos = netlist.mosList.size();

	// List resistors
	cout << "Resistors:\n";
	for (int i = 0; i < num_r; i++) {
		printf(rList[i].name);
		printf(": %f Ohms\n", rList[i].val);
	}

	// List VDCs
	cout << "VDCs:\n";
	for (int i = 0; i < num_vdc; i++) {
		printf(vdcList[i].name);
		printf(": %f V\n", vdcList[i].val);
	}

	// List IDCs
	cout << "IDCs:\n";
	for (int i = 0; i < num_idc; i++) {
		printf(idcList[i].name);
		printf(": %f A\n", idcList[i].val);
	}

	// List MOSFETS
	cout << "MOSFETS:\n";
	for (int i = 0; i < num_mos; i++) {
		printf(mosList[i].name);
		printf(": Model = %s, L = %f um, W = %f um\n", mosList[i].model->name, mosList[i].l * 1e6, mosList[i].w * 1e6);
	}

	cout << "\n";


	// Conductance Matrix
	float** gMat = mat2D(num_nodes, num_nodes);
	
	// Current Matrix
	float* iMat = mat1D(num_nodes);

	// Voltage Matrix (what we're solving)
	float* vMat = mat1D(num_nodes);

	// Populate G matrix from Resistor Elements
	for (int i = 0; i < num_r; i++) {
		R_toMat(rList + i, gMat);
	}

	printf("G Matrix:\n");
	cout << mat2DToStr(gMat, num_nodes, num_nodes);

	// IDC Sources populate I matrix
	for (int i = 0; i < num_idc; i++) {
		Idc_toMat(idcList + i, iMat);
	}
	for (int i = 0; i < num_vccs; i++) {
		VCCS_toMat(vccsList + i, gMat);
	}

	// VDC Source populates G and I matrices
	int err_test;
	for (int i = 0; i < num_vdc; i++) {
		err_test = Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
		if (err_test == -1) {
			cout << "ERROR! VDC Shorted: " << vdcList[i].name << "\n";
			//free(rList);
			//free(vdcList);
			//free(idcList);
			//freeMat2D(gMat, num_nodes);
			free(iMat);
			free(vMat);
			//system("pause");
			return -1;
		}
	}
	
	cout << "G Matrix after Vdc & Idc:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	gpuMatSolve(num_nodes, gMat, iMat, vMat);


	cout << "\nSolution:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	// Testing MOSFET current calc
	/*
	Transistor T;
	T.l = 1.0f;
	T.w = 1.0f;
	T.g = 2;
	T.s = 0;
	T.d = 2;

	Model M;

	T.model = &M;

	float c = calcId(&T, vMat);
	printf("T current calc test: %f uA\n", c * 1000000.0f);
	*/

	//system("pause");
	//free(rList);
	//free(vdcList);
	//free(idcList);
	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	return 0;


}