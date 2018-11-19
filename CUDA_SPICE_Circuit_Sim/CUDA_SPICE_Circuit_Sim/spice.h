#pragma once
#include<iostream> 
#include<fstream>
#include <vector>

using namespace std;

#define MAX_LINE 256

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
	int model;
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

struct VCCS {
	float g;
	char* name;
	// current source nodes
	int ip;
	int in;
	// control voltage nodes
	int vp;
	int vn;
};

struct Netlist {
	vector<Resistor> rList;
	vector<Vdc> vdcList;
	vector<Idc> idcList;
	vector<VCCS> vccsList;
	vector<char*> netNames;
};

int parseNetlist(char* filepath, Netlist &netlist);
int parseElement(char* line, Netlist& netlist);
int findNode(vector<char*> &nodeList, char* name, int n);