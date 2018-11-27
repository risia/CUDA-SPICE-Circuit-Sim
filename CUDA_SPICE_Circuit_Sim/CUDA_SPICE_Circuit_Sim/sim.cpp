#include "sim.h"

void op(Netlist netlist) {
	int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_mos = netlist.active_elem.size();
	Element* mosList = netlist.active_elem.data();

	int num_vdc = netlist.vdcList.size();
	Element* vdcList = netlist.vdcList.data();

	// Conductance Matrix
	float** gMat = mat2D(num_nodes, num_nodes);

	// Current Matrix
	float* iMat = mat1D(num_nodes);

	// Voltage Matrix (what we're solving)
	float* vMat = mat1D(num_nodes);

	linNetlistToMat(netlist, gMat, iMat, vMat);

	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
	}

	cout << "Passive G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	gpuMatSolve(num_nodes, gMat, iMat, vMat);


	cout << "\nSolution 0:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	/*
	Transistor convergence loop
	*/

	// Max error between current and previous guess
	float e = MAX_FLOAT;
	// Store prev. guess for calc. and comparison
	float* vGuess = mat1D(num_nodes);
	// Loop counter
	int n = 0;

	// Limit n to 1000 to prevent inf loop
	// in case of no convergence/bad circuit
	while (e > TOL && n < 1000 && num_mos > 0) {
		// copy prev. guess
		matCpy(vGuess, vMat, num_nodes);

		// Reset matrices
		resetMat2D(gMat, num_nodes, num_nodes);
		resetMat1D(iMat, num_nodes);
		resetMat1D(vMat, num_nodes);

		// Apply passive elements
		linNetlistToMat(netlist, gMat, iMat, vMat);

		// Apply Transistor
		for (int i = 0; i < num_mos; i++) {
			MOS_toMat(&mosList[i], gMat, iMat, vGuess, num_nodes);
		}
		
		// Recalc VDC currents?
		for (int i = 0; i < num_vdc; i++) {
			Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
		}
		


		// Attempt solution
		gpuMatSolve(num_nodes, gMat, iMat, vMat);

		// Measure error beteen old and new guess
		e = maxDiff(vGuess, vMat, num_nodes);

		// Iteration counter for testing
		n++;

		// print matrices for testing
		printf("\nSolution %i:\n\n", n);
		cout << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
		cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
		cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
		printf("\nError: %e\n\n", e);
	}

	cout << "\nFinal Solution:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);


	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);
}

void dcSweep(Netlist netlist, char* name, float start, float stop, float step) {
	// variable setup
	// store original parameter value
	// in case we need to do multiple simulations
	float original_val;
	char type = 0;
	int index = 0;

	int num_nodes = netlist.netNames.size();

	// Setup matrices
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);

	// find element named & setup matrices
	linNetlistToMatFindElem(netlist, gMat, iMat, vMat, name, type, index);


	// Make copy of original value
	// Set to start

	// solve

	// loop steps

	// store solutions,
	// allow printing when finished or save to file
}