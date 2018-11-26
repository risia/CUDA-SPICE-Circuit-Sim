#include "sim.h"

void op(Netlist netlist) {
	int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_mos = netlist.mosList.size();
	Transistor* mosList = netlist.mosList.data();

	// Conductance Matrix
	float** gMat = mat2D(num_nodes, num_nodes);

	// Current Matrix
	float* iMat = mat1D(num_nodes);

	// Voltage Matrix (what we're solving)
	float* vMat = mat1D(num_nodes);

	linNetlistToMat(netlist, gMat, iMat, vMat);

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
	while (e > TOL && n < 1000) {
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