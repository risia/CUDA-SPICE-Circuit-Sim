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
	CUDA_Net* dev_net = new CUDA_Net();

	//char* file = "C:/Users/Angelinia/Documents/CIS 565/CUDA-SPICE-Circuit-Sim/test_spi/Bigger_test.spi";
	parseNetlist(file, netlist);
	gpuNetlist(netlist, dev_net);

	steady_clock::time_point t2 = steady_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "\nNetlist parser took " << time_span.count() << " seconds.\n";

	cuda_op(netlist);


	t1 = steady_clock::now();

	op(netlist);

	t2 = steady_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "\nUnoptimized OP Solver took " << time_span.count() << " seconds.\n";

	t1 = steady_clock::now();

	cuda_op(netlist);

	t2 = steady_clock::now();

	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "\nOP Solver took " << time_span.count() << " seconds.\n";


	t1 = steady_clock::now();

	full_cudaOp(netlist, dev_net);

	t2 = steady_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "\nFull CUDA OP Solver took " << time_span.count() << " seconds.\n";




	/*
	TEST: Transient & DC Sweep
	*/
	
	float start = 0;
	float stop = 5e-9;
	float step = 0.01e-9f;

	t1 = steady_clock::now();

	transient(netlist, start, stop, step);

	t2 = steady_clock::now();

	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "\nTransient Solver took " << time_span.count() << " seconds.\n";

	start = 0.0f;
	stop = 1.0f;
	step = 0.01f;

	char* name = "VDC@1";

	t1 = steady_clock::now();

	dcSweep(netlist, name, start, stop, step);

	t2 = steady_clock::now();

	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "\nDC Sweep Solver took " << time_span.count() << " seconds.\n";


	
	free(netlist);
	free(dev_net);

	system("pause");
	return 0;


}