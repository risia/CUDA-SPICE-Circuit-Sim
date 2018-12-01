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
	bool isConverged = false;
	if (num_mos == 0) isConverged = true;
	// Store prev. guess for calc. and comparison
	float* vGuess = mat1D(num_nodes);
	// Loop counter
	int n = 0;

	// Limit n to 1000 to prevent inf loop
	// in case of no convergence/bad circuit
	while (!isConverged && n < 1000 && num_mos > 0) {
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
		
		// Recalc VDC currents
		for (int i = 0; i < num_vdc; i++) {
			Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
		}
		


		// Attempt solution
		gpuMatSolve(num_nodes, gMat, iMat, vMat);

		// Measure error beteen old and new guess
		isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);

		// Iteration counter for testing
		n++;

		// print matrices for testing
		printf("\nSolution %i:\n\n", n);
		cout << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
		cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
		cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
	}

	cout << "\nFinal Solution:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	cout << "Converged? : " << (isConverged ? "true" : "false") << "\n";



	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);
}

void cuda_op(CUDA_Net netlist) {
	// Copy netlist passive data to device
	int n_nodes = netlist.n_nodes; // node 0 = GND
	int n_mos = netlist.n_active;
	CUDA_Elem* mosList = netlist.actives;

	int n_vdc = netlist.n_vdc;
	CUDA_Elem* vdcList = netlist.vdcList;
	int n_passive = netlist.n_passive;
}

float** dcSweep(Netlist netlist, char* name, float start, float stop, float step) {
	// variable setup
	// store original parameter value
	// in case we need to do multiple simulations
	float original_val;
	float current_val;

	int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_mos = netlist.active_elem.size();
	Element* mosList = netlist.active_elem.data();

	Element* passives = netlist.elements.data();
	int num_passive = netlist.elements.size();

	int num_vdc = netlist.vdcList.size();
	Element* vdcList = netlist.vdcList.data();

	// Setup matrices
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);
	float* vGuess = mat1D(num_nodes);

	int num_steps = floor(1 + (stop - start) / step) ;
	if (stop > (start + step * num_steps)) num_steps++;

	// rows are sweep step, columns node voltage
	// col 0 is current val of swept parameter for that solution
	float** vSweepMat = mat2D(num_steps, num_nodes + 1);

	// find element named & setup matrices
	Element* swp_elem = NULL; //linNetlistToMatFindElem(netlist, gMat, iMat, vMat, name);

	for (int i = 0; i < num_passive && swp_elem == NULL; i++) {
		if (strcmp(passives[i].name, name) == 0) {
			swp_elem = &(passives[i]);
		}
	}
	for (int i = 0; i < num_vdc && swp_elem == NULL; i++) {
		if (strcmp(vdcList[i].name, name) == 0) {
			swp_elem = &(vdcList[i]);
		}
	}
	for (int i = 0; i < num_mos && swp_elem == NULL; i++) {
		if (strcmp(mosList[i].name, name) == 0) {
			swp_elem = &(mosList[i]);
		}
	}

	// Make copy of original value
	// Set to start
	original_val = swp_elem->params[0]; // I'm just assuming first parameter for now
	current_val = start;


	bool isConverged = false;
	int n;

	// loop steps
	for (int i = 0; i < num_steps; i++) {
		vSweepMat[i][0] = current_val;
		swp_elem->params[0] = current_val;

		// Attempt solution
		resetMat2D(gMat, num_nodes, num_nodes);
		resetMat1D(iMat, num_nodes);
		resetMat1D(vMat, num_nodes);

		linNetlistToMat(netlist, gMat, iMat, vMat);
		for (int i = 0; i < num_vdc; i++) {
			Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
		}

		gpuMatSolve(num_nodes, gMat, iMat, vMat);

		/*
		Transistor convergence loop
		*/
		n = 0;
		isConverged = false;
		if (num_mos == 0) isConverged = true;
		while (!isConverged && n < 1000 && num_mos > 0) {
			matCpy(vGuess, vMat, num_nodes);

			resetMat2D(gMat, num_nodes, num_nodes);
			resetMat1D(iMat, num_nodes);
			resetMat1D(vMat, num_nodes);

			linNetlistToMat(netlist, gMat, iMat, vMat);
			for (int i = 0; i < num_mos; i++) {
				MOS_toMat(&mosList[i], gMat, iMat, vGuess, num_nodes);
			}
			for (int i = 0; i < num_vdc; i++) {
				Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
			}

			gpuMatSolve(num_nodes, gMat, iMat, vMat);

			isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
			n++;
		}

		matCpy(vSweepMat[i] + 1, vMat, num_nodes);

		if (i == num_steps - 2) current_val = stop;
		else current_val += step;
	}

	swp_elem->params[0] = original_val;

	cout << "DC Sweep Solutions:\n\n" << mat2DToStr(vSweepMat, num_steps, num_nodes + 1);
	





	// store solutions,
	// allow printing when finished or save to file
	// cleanup

	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);


	return vSweepMat;
}