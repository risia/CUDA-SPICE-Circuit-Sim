#include <cstdio>
#include <iostream>
#include "helpers.h"
#include "test_circuit.h"
#include "kcl.h"
#include "linSolver.h"


int main() {
	Resistor* rList = testRList();
	Vdc* vdcList = testVList();
	Idc* idcList = testIList();

	int num_nodes = 3; // node 0 = GND
	int num_r = 5;
	int num_vdc = 1;
	int num_idc = 1;

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
	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, num_nodes);
	}
	for (int i = 0; i < num_idc; i++) {
		Idc_toMat(idcList + i, iMat);
	}
	
	cout << "G Matrix after Vdc & Idc:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);

	for (int i = 0; i < num_nodes - 1; i++) {
		matReduce(gMat, iMat, num_nodes, num_nodes, i);
	}

	cout << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);

	for (int i = 0; i < num_nodes; i++) {
		matSolve(gMat, iMat, vMat, num_nodes, num_nodes, i);
	}

	cout << "\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	system("pause");

	free(rList);
	free(vdcList);
	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	return 0;
}