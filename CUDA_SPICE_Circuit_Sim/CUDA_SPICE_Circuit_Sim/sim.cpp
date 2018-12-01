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
	/*
	cout << "Passive G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
	*/
	gpuMatSolve(num_nodes, gMat, iMat, vMat);

	/*
	cout << "\nSolution 0:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
	*/

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
		/*
		printf("\nSolution %i:\n\n", n);
		cout << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
		cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
		cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
		*/
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

/*
More streamlined to reduce memory copies, mallocs, frees, etc
*/
void cuda_op(Netlist netlist) {
	float* dev_gMat = NULL;
	float* dev_vMat = NULL;
	float* dev_iMat = NULL;

	int num_nodes = netlist.netNames.size() - 1; // node 0 = GND
	int num_mos = netlist.active_elem.size();
	int num_passive = netlist.elements.size();
	int num_vdc = netlist.vdcList.size();

	Element* mosList = netlist.active_elem.data();
	Element* passives = netlist.elements.data();
	Element* vdcList = netlist.vdcList.data();

	// Setup matrices
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);
	float* vGuess = mat1D(num_nodes);

	float** gMatCpy = mat2D(num_nodes, num_nodes);
	float* iMatCpy = mat1D(num_nodes);

	// setup of passive elements g and i matrices, 
	// keep copy on CPU so we don't need to rebuild passives
	linNetlistToMat(netlist, gMat, iMat, vMat);
	mat2DCpy(gMatCpy, gMat, num_nodes, num_nodes);
	matCpy(iMatCpy, iMat, num_nodes);

	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
	}

	// setup and copy matrices to device
	setupDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);

	// passive solution
	gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);

	/*
	Test output
	*/
	/*
	copyFromDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);
	cout << "\nSolution 0:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
	*/

	// copy vMat from device to guess
	copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vGuess, dev_vMat);
	// copy copies of gmat and imat to respective mat
	// reset vMat and populate

	bool isConverged = false;
	if (num_mos == 0) isConverged = true;
	int n = 0;
	while (!isConverged && n < 1000 && num_mos > 0) {
		// copy prev. guess
		copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vGuess, dev_vMat);

		// Reset matrices & apply passive elements
		mat2DCpy(gMat, gMatCpy, num_nodes, num_nodes);
		matCpy(iMat, iMatCpy, num_nodes);
		resetMat1D(vMat, num_nodes);

		// Apply Transistor
		for (int i = 0; i < num_mos; i++) {
			MOS_toMat(&mosList[i], gMat, iMat, vGuess, num_nodes);
		}
		// Recalc VDC currents
		for (int i = 0; i < num_vdc; i++) {
			Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
		}

		copyToDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);

		// Attempt solution
		gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);

		// copy new vMat
		copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vMat, dev_vMat);

		// Measure error beteen old and new guess
		isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);

		// Iteration counter for testing
		n++;

		// print matrices for testing
		/*
		printf("\nSolution %i:\n\n", n);
		cout << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
		cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
		cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
		*/
	}

	copyFromDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, NULL, NULL);

	cout << "\nFinal Solution:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	cout << "Converged? : " << (isConverged ? "true" : "false") << "\n";


	cleanDevMats(dev_gMat, dev_iMat, dev_vMat);

	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);

	freeMat2D(gMatCpy, num_nodes);
	free(iMatCpy);
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
	// restore element parameter val
	swp_elem->params[0] = original_val;


	// store solutions,
	// allow printing when finished or save to file
	ofstream sweep_output;
	sweep_output.open("dc_sweep.txt");

	string out = mat2DToStr(vSweepMat, num_steps, num_nodes + 1);
	cout << "DC Sweep Solutions:\n\n" << name;
	sweep_output << name;
	for (int i = 1; i <= num_nodes; i++) {
		sweep_output << " " << netlist.netNames[i];
		cout << " " << netlist.netNames[i];
	}
	sweep_output << "\n" << out;
	cout << "\n" << out;

	sweep_output.close();



	// cleanup

	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);


	return vSweepMat;
}



void transient(Netlist netlist, float start, float stop, float step) {

}