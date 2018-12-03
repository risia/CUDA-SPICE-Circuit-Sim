#include "main.h"

int main() {

	using namespace std::chrono;

	steady_clock::time_point t1 = steady_clock::now();

	Netlist* netlist = new Netlist();
	//CUDA_Net* dev_net = new CUDA_Net();

	char* file = "C:/Users/Angelinia/Documents/CIS 565/CUDA-SPICE-Circuit-Sim/test_spi/Bigger_test.spi";
	parseNetlist(file, netlist);

	steady_clock::time_point t2 = steady_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "Netlist parser took " << time_span.count() << " seconds.\n";

	/*
	GPU Netlist stuff still broken
	Probably going to setup some other way, and in parsing step instead
	*/
	/*
	gpuNetlist(&netlist, dev_net);

	int n = dev_net->n_nodes;

	float** gMat = mat2D(n, n);
	float* iMat = mat1D(n);

	gpuNetlistToMat(dev_net, gMat, iMat);

	cout << "Passive G Matrix GPU TEST:\n" << mat2DToStr(gMat, n, n);
	cout << "I Matrix:\n" << mat1DToStr(iMat, n) << "\n\n";

	freeMat2D(gMat, n);
	free(iMat);

	freeGpuNetlist(dev_net);
	*/
	//for (int i = 0; i < 10; i++) {
	/*
		t1 = steady_clock::now();
		op(netlist);
		t2 = steady_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		cout << "Unoptimized OP Solver took " << time_span.count() << " seconds.\n";
		*/

		t1 = steady_clock::now();
		cuda_op(netlist);
		t2 = steady_clock::now();

		time_span = duration_cast<duration<double>>(t2 - t1);
		cout << "OP Solver took " << time_span.count() << " seconds.\n";
	//}

	/*
	TEST: DC Sweep
	*/
	/*
	float start = 0.0f;
	float stop = 5.0f;
	float step = 0.25f;

	char* name = "VDC@1";

	float** sweep = dcSweep(netlist, name, start, stop, step);

	int num_steps = floor((stop - start) / step);
	if (stop != (start + step * num_steps)) num_steps++;

	freeMat2D(sweep, num_steps);
	*/
	/*
	TEST: DC Sweep
	*/
	/*
	float start = 0.0f;
	float stop = 40e-9;
	float step = 0.05e-9f;

	transient(netlist, start, stop, step);
	*/
	free(netlist);
	return 0;


}