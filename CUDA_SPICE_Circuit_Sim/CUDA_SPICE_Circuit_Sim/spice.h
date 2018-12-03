#pragma once
#include<iostream> 
#include<fstream>
#include <vector>

using namespace std;

#define MAX_LINE 256
#define TYPES "RVIGMC"
#define N_TYPES 6

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

#ifndef MODEL_STRUCT
#define MODEL_STRUCT
struct Model {
	char* name = "N";
	char type = 'n';

	float u0 = 540.0f; // 
	float tox = 14.1e-9; // oxide thickness
	float epsrox = 3.9f; // dielectric constant

	float vt0 = 0.7f; // threshold voltage
	float pclm = 0.6171774f; // CLM parameter
	float vsat = 8.0e4; // saturation velocity
	float nfactor = 1.0f; // subthreshold swing factor
	//float Ld = 0.0f;

	float CGSO = 0.0f; // gate-source overlap cap per unit W
	float CGDO = 0.0f; // gate-drain overlap cap per unit W
	float CGBO = 0.0f; // gate-bulk cap per unit L
};
#endif // !1

struct Resistor {
	float val;
	char* name;
	int node1;
	int node2;
};

struct Capacitor {
	float val;
	int node1;
	int node2;
};

struct Inductor {
	float val;
	int node1;
	int node2;
};

struct Transistor {
	float l; // length
	float w; // width
	int g; // gate net
	int d; // drain
	int s; // source
	int b; // bulk
	Model* model;
	char* name;
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

// Passive Element
struct Element {
	char type;
	char* name;
	vector<float> params;
	vector<int> nodes;
	Model* model = NULL;
};

struct Netlist {
	vector<Model*> modelList;
	vector<char*> netNames;

	// Passives
	vector<Element> elements;
	// Voltage sources (need to be applied last)
	vector<Element> vdcList;
	// Transistors
	vector<Element> active_elem;
};


int parseNetlist(const char* filepath, Netlist* netlist);

void parseNodes(int n, Netlist* netlist, Element* e, char* delim);
void parseValues(int n, int skip, Netlist* netlist, Element* e, char* delim);

int parseElement(char* line, Netlist* netlist);

int findNode(vector<char*> &nodeList, char* name, int n);
Model* findModel(vector<Model*> &modelList, char* name, int n);
float numPrefix(float num, char prefix);

void parseCmd(char* line, Netlist* netlist, ifstream* file);