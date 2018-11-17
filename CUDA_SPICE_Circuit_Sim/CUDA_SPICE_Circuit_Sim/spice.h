/*
Functions to parse spice netlist files

Format:

Title (first line always ignored?)
Elements
	circuit elements w/ node nets and parameter values
Commands
Outputs
.END

Each line represents a separate command or element unless a + sign is given

https://www.seas.upenn.edu/~jan/spice/spice.overview.html
*/

#pragma once

struct Resistor {
	float val;
	char* name;
	int node1;
	int node2;
};

struct Capacitor {
	float val;
	char* node1;
	char* node2;
};

struct Inductor {
	float val;
	char* node1;
	char* node2;
};

struct Transistor {
	float l; // length
	float w; // width
	char* g; // gate net
	char* s; // source
	char* d; // drain
	char* b; // bulk

};

struct Vdc {
	float val;
	char* name;
	int node_p;
	int node_n;
};

struct Idc {
	float val;
	char* name;
	int node_p;
	int node_n;
};

