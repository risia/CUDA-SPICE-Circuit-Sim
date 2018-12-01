#include "main.h"

int main() {
	Netlist netlist;
	CUDA_Net* dev_net = new CUDA_Net();

	char* file = "C:/Users/Angelinia/Documents/CIS 565/CUDA-SPICE-Circuit-Sim/test_spi/test2.spi";
	parseNetlist(file, netlist);

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

	op(netlist);

	return 0;


}