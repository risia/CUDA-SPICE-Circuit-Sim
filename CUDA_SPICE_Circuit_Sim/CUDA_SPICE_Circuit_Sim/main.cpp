#include "main.h"

int main() {
	Netlist netlist;

	char* file = "C:/Users/Angelinia/Documents/CIS 565/CUDA-SPICE-Circuit-Sim/test_spi/Bigger_test.spi";
	parseNetlist(file, netlist);

	op(netlist);

	return 0;


}