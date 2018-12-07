#include "main.h"
using namespace std::chrono;

int main(int argc, char** argv) {
	
	if (argc < 2) {
		printf("Usage: %s \"FILEPATH/NETLIST_FILE\"\n", argv[0]);
		return 1;
	}

	const char* file = argv[1];

	steady_clock::time_point t1 = steady_clock::now();

	Netlist* netlist = new Netlist();

	//char* file = "C:/Users/Angelinia/Documents/CIS 565/CUDA-SPICE-Circuit-Sim/test_spi/Bigger_test.spi";
	parseNetlist(file, netlist);

	steady_clock::time_point t2 = steady_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "Netlist parser took " << time_span.count() << " seconds.\n";


	t1 = steady_clock::now();
	cuda_op(netlist);
	t2 = steady_clock::now();

	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "OP Solver took " << time_span.count() << " seconds.\n";

	/*
	TEST: DC Sweep
	*/
	
	float start = 0;
	float stop = 5e-9;
	float step = 0.001e-9f;

	//transient(netlist, start, stop, step);


	start = 0.0f;
	stop = 1.0f;
	step = 0.01f;

	char* name = "VPulse@0";

	//dcSweep(netlist, name, start, stop, step);


	//cout << "\n*******************\n" << "CUDA Netlist Test" << "\n*******************\n";

	CUDA_Net* dev_net = new CUDA_Net();

	gpuNetlist(netlist, dev_net);

	/*
	int n = dev_net->n_nodes;

	float** gMat = mat2D(n, n);
	float* iMat = mat1D(n);
	float* vMat = mat1D(n);

	gpuNetlistToMat(dev_net, netlist, gMat, iMat, vMat);

	cout << mat2DToStr(gMat, n, n) << "\n" << mat1DToStr(iMat, n) << "\n" << mat1DToStr(vMat, n);


	free(iMat);
	free(vMat);
	freeMat2D(gMat, n);
	*/

	full_cudaOp(netlist, dev_net);
	
	free(netlist);
	free(dev_net);
	return 0;


}