#include "main.h"

int main() {
	Resistor* rList = testRList3();
	Vdc* vdcList = testVList3();
	Idc* idcList = testIList1();

	int num_nodes = 2; // node 0 = GND
	int num_r = 3;
	int num_vdc = 1;
	int num_idc = 0;

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

	// VDC Source populates G and I matrices
	int err_test;
	for (int i = 0; i < num_vdc; i++) {
		err_test = Vdc_toMat(vdcList + i, gMat, iMat, num_nodes);
		if (err_test == -1) {
			cout << "ERROR! VDC Shorted: " << vdcList[i].name << "\n";
			free(rList);
			free(vdcList);
			free(idcList);
			freeMat2D(gMat, num_nodes);
			free(iMat);
			free(vMat);
			system("pause");
			return -1;
		}
	}
	for (int i = 0; i < num_idc; i++) {
		Idc_toMat(idcList + i, iMat);
	}
	
	cout << "G Matrix after Vdc & Idc:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);

	gpuMatSolve(num_nodes, gMat, iMat, vMat);


	cout << "\nSolution:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	system("pause");

	//cleanDevMats(dev_gMat, dev_iMat, dev_vMat);
	free(rList);
	free(vdcList);
	free(idcList);
	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	return 0;


}