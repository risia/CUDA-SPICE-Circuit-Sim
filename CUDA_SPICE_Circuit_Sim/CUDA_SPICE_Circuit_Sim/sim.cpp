#include "sim.h"

void op(Netlist* netlist) {
	int num_nodes = netlist->netNames.size() - 1; // node 0 = GND
	int num_mos = netlist->active_elem.size();
	int num_vdc = netlist->vdcList.size();
	Element* mosList = netlist->active_elem.data();
	Element* vdcList = netlist->vdcList.data();

	// Conductance Matrix
	float** gMat = mat2D(num_nodes, num_nodes);

	// Current Matrix
	float* iMat = mat1D(num_nodes);

	// Voltage Matrix (what we're solving)
	float* vMat = mat1D(num_nodes);

	// Store prev. guess for calc. and comparison
	float* vGuess = mat1D(num_nodes);

	linNetlistToMat(netlist, gMat, iMat);

	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
	}

	gpuMatSolve(num_nodes, gMat, iMat, vMat);



	// Max error between current and previous guess
	bool isConverged = false;
	if (num_mos == 0) isConverged = true;

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
		linNetlistToMat(netlist, gMat, iMat);

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

		// Iteration counter
		n++;
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
void cuda_op(Netlist* netlist) {
	float* dev_gMat = NULL;
	float* dev_vMat = NULL;
	float* dev_iMat = NULL;

	int num_nodes = netlist->netNames.size() - 1; // node 0 = GND
	int num_mos = netlist->active_elem.size();
	int num_vdc = netlist->vdcList.size();

	Element* mosList = netlist->active_elem.data();
	Element* vdcList = netlist->vdcList.data();

	// Setup matrices
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);
	float* vGuess = mat1D(num_nodes);

	float** gMatCpy = mat2D(num_nodes, num_nodes);
	float* iMatCpy = mat1D(num_nodes);

	// setup of passive elements g and i matrices, 
	// keep copy on CPU so we don't need to rebuild passives
	linNetlistToMat(netlist, gMat, iMat);
	mat2DCpy(gMatCpy, gMat, num_nodes, num_nodes);
	matCpy(iMatCpy, iMat, num_nodes);

	for (int i = 0; i < num_vdc; i++) {
		Vdc_toMat(vdcList + i, gMat, iMat, vMat, num_nodes);
	}

	// setup and copy matrices to device
	setupDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);

	// passive solution
	gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);

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

		// Iteration counter
		n++;
	}

	copyFromDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);



	// Test Output
	cout << "TEST: Original Passive G Matrix\n" << mat2DToStr(gMatCpy, num_nodes, num_nodes);

	cout << "\nFinal Solution:\n\n" << "G Matrix:\n" << mat2DToStr(gMat, num_nodes, num_nodes);
	cout << "I Matrix:\n" << mat1DToStr(iMat, num_nodes);
	cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);

	cout << "Converged? : " << (isConverged ? "true" : "false") << "\n";

	// memory cleanup
	cleanDevMats(dev_gMat, dev_iMat, dev_vMat);
	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);
	freeMat2D(gMatCpy, num_nodes);
	free(iMatCpy);
}

void dcSweep(Netlist* netlist, char* name, float start, float stop, float step) {
	// variable setup
	// store original parameter value
	// in case we need to do multiple simulations
	float original_val;
	float current_val;

	int num_nodes = netlist->netNames.size() - 1; // node 0 = GND
	int num_mos = netlist->active_elem.size();
	Element* mosList = netlist->active_elem.data();

	Element* passives = netlist->elements.data();
	int num_passive = netlist->elements.size();

	int num_vdc = netlist->vdcList.size();
	Element* vdcList = netlist->vdcList.data();

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

		linNetlistToMat(netlist, gMat, iMat);
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

			linNetlistToMat(netlist, gMat, iMat);
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


	char** names = netlist->netNames.data();
	names[0] = name;
	mat2DtoCSV(names, vSweepMat, num_steps, num_nodes + 1, "dcSweep.csv");
	names[0] = "gnd";

	// cleanup

	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);
	freeMat2D(vSweepMat, num_steps);

	//return vSweepMat;
}



void transient(Netlist* netlist, float start, float stop, float step) {
	float* dev_gMat = NULL;
	float* dev_vMat = NULL;
	float* dev_iMat = NULL;

	int num_nodes = netlist->netNames.size() - 1; // node 0 = GND
	int num_mos = netlist->active_elem.size();
	int num_vdc = netlist->vdcList.size();

	Element* mosList = netlist->active_elem.data();
	Element* vdcList = netlist->vdcList.data();

	// Setup matrices
	float** gMat = mat2D(num_nodes, num_nodes);
	float* iMat = mat1D(num_nodes);
	float* vMat = mat1D(num_nodes);
	float* vGuess = mat1D(num_nodes);
	float* vPrev = mat1D(num_nodes);

	float** gMatCpy = mat2D(num_nodes, num_nodes);
	float* iMatCpy = mat1D(num_nodes);

	float time = 0.0f;

	// Setup passive matrices
	linNetlistToMat(netlist, gMat, iMat);
	mat2DCpy(gMatCpy, gMat, num_nodes, num_nodes);
	matCpy(iMatCpy, iMat, num_nodes);

	for (int i = 0; i < num_vdc; i++) {
		VTran_toMat(vdcList + i, gMat, iMat, vMat, time, num_nodes);
	}

	setupDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);

	// Solve for t = 0s
	gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);
	copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vMat, dev_vMat);
	
	bool isConverged = false;
	if (num_mos == 0) isConverged = true;
	int n = 0;
	while (!isConverged && n < 1000 && num_mos > 0) {
		matCpy(vGuess, vMat, num_nodes);

		mat2DCpy(gMat, gMatCpy, num_nodes, num_nodes);
		matCpy(iMat, iMatCpy, num_nodes);
		resetMat1D(vMat, num_nodes);

		for (int i = 0; i < num_mos; i++) {
			MOS_toMat(&mosList[i], gMat, iMat, vGuess, num_nodes);
		}
		for (int i = 0; i < num_vdc; i++) {
			VTran_toMat(vdcList + i, gMat, iMat, vMat, time, num_nodes);
		}

		copyToDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);
		gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);
		copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vMat, dev_vMat);

		isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
		n++;
	}
	// Previous timestep voltage Matrix
	matCpy(vPrev, vMat, num_nodes);

	int skipped_steps = floor(start / step);
	if (start >(step * skipped_steps)) skipped_steps++;

	int n_steps = floor(1 + stop / step);
	if (stop > (step * n_steps)) n_steps++;
	float ** vSimMat = mat2D(n_steps - skipped_steps, num_nodes + 1);

	for (int t = 0; t < n_steps; t++) {
		// setup matices
		mat2DCpy(gMat, gMatCpy, num_nodes, num_nodes);
		matCpy(iMat, iMatCpy, num_nodes);
		resetMat1D(vMat, num_nodes);

		tranJustCToMat(netlist, gMat, iMat, vPrev, step);
		for (int i = 0; i < num_vdc; i++) {
			VTran_toMat(vdcList + i, gMat, iMat, vMat, time, num_nodes);
		}
		// Guess solution
		copyToDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);
		gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);
		copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vMat, dev_vMat);

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

			linNetlistToMat(netlist, gMat, iMat);
			tranJustCToMat(netlist, gMat, iMat, vPrev, step);
			for (int i = 0; i < num_mos; i++) {
				transientMOS_toMat(mosList + i, gMat, iMat, vGuess, vPrev, num_nodes, step);
				//MOS_toMat(mosList + i, gMat, iMat, vGuess, num_nodes);
			}
			for (int i = 0; i < num_vdc; i++) {
				VTran_toMat(vdcList + i, gMat, iMat, vMat, time, num_nodes);
			}

			copyToDevMats(num_nodes, gMat, dev_gMat, iMat, dev_iMat, vMat, dev_vMat);
			gpuDevMatSolve(num_nodes, dev_gMat, dev_iMat, dev_vMat);
			copyFromDevMats(num_nodes, NULL, NULL, NULL, NULL, vMat, dev_vMat);

			isConverged = matDiffCmp(vGuess, vMat, num_nodes, TOL);
			n++;


			//matCpy(vGuess, vMat, num_nodes);
		}
		/*
		cout << "\nTime: " << time << " Final Solution:\n";
		cout << "V Matrix:\n" << mat1DToStr(vMat, num_nodes);
		cout << "Converged? : " << (isConverged ? "true" : "false") << "after " << n << " iterations\n";
		*/


		// copy solution to output
		if (t >= skipped_steps) {
			vSimMat[t - skipped_steps][0] = time;
			matCpy(vSimMat[t - skipped_steps] + 1, vMat, num_nodes);
		}
		matCpy(vPrev, vMat, num_nodes);


		if (t == n_steps - 2) time = stop;
		else if (t == skipped_steps - 1) time = start;
		else time += step;
	}

	// store solutions,
	// save to file csv
	char** names = netlist->netNames.data();
	names[0] = "Time(s)";
	mat2DtoCSV(names, vSimMat, n_steps - skipped_steps, num_nodes + 1, "transient.csv");
	names[0] = "gnd";

	// Cleanup
	cleanDevMats(dev_gMat, dev_iMat, dev_vMat);
	freeMat2D(gMat, num_nodes);
	free(iMat);
	free(vMat);
	free(vGuess);
	freeMat2D(gMatCpy, num_nodes);
	free(iMatCpy);

	free(vPrev);
	freeMat2D(vSimMat, n_steps - skipped_steps);
}