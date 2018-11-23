#include "spice.h"

int parseNetlist(char* filepath, Netlist &netlist) {

	ifstream inFile(filepath);
	char oneline[MAX_LINE];


	vector<Idc>().swap(netlist.idcList);
	vector<VCCS>().swap(netlist.vccsList);
	vector<Vdc>().swap(netlist.vdcList);
	vector<Resistor>().swap(netlist.rList);
	vector<char*>().swap(netlist.netNames);

	netlist.netNames.push_back("gnd");

	while (inFile)
	{
		inFile.getline(oneline, MAX_LINE);
		if (oneline[0] == '*') continue;
		if (oneline[0] != '.') parseElement(oneline, netlist);


		//cout << oneline << endl;
	}

	/*
	// List resistors
	cout << "File Resistors:\n";
	int num_r = netlist.rList.size();

	for (int i = 0; i < num_r; i++) {
		printf(netlist.rList[i].name);
		printf(": %d %d %f Ohms\n", netlist.rList[i].node1, netlist.rList[i].node2, netlist.rList[i].val);
	}

	// List VDCS
	cout << "File VDCs:\n";
	int num_vdc = netlist.vdcList.size();

	for (int i = 0; i < num_vdc; i++) {
		printf(netlist.vdcList[i].name);
		printf(": %d %d %f V\n", netlist.vdcList[i].node_p, netlist.vdcList[i].node_n, netlist.vdcList[i].val);
	}
	*/

	inFile.close();

	return 0;
}

int parseElement(char* line, Netlist& netlist) {

	char type = line[0];

	char* token;
	char* delim = " ";

	float val;

	// parse resistor
	if (type == 'R') {
		Resistor r;

		// get name
		token = strtok(line + 1, delim);
		r.name = new char[strlen(token) + 1];
		strcpy(r.name, token);

		// node 1
		token = strtok(NULL, delim);
		r.node1 = findNode(netlist.netNames, token, netlist.netNames.size());

		// node 2
		token = strtok(NULL, delim);
		r.node2 = findNode(netlist.netNames, token, netlist.netNames.size());

		// value
		token = strtok(NULL, delim);
		val = atof(token);

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		r.val = numPrefix(val, type);

		netlist.rList.push_back(r);
	}
	// parse VDC
	else if (type == 'V') {
		Vdc v;

		// get name
		token = strtok(line + 1, delim);
		v.name = new char[strlen(token) + 1];
		strcpy(v.name, token);

		// node p
		token = strtok(NULL, delim);
		v.node_p = findNode(netlist.netNames, token, netlist.netNames.size());

		// node n
		token = strtok(NULL, delim);
		v.node_n = findNode(netlist.netNames, token, netlist.netNames.size());

		// val

		token = strtok(NULL, delim);
		if (strcmp(token, "DC") == 0) {
			token = strtok(NULL, delim);
			val = atof(token);
		}

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		v.val = numPrefix(val, type);

		netlist.vdcList.push_back(v);
	}
	// Parsse IDC
	else if (type == 'I') {
		Idc i;

		// get name
		token = strtok(line + 1, delim);
		i.name = new char[strlen(token) + 1];
		strcpy(i.name, token);

		// node p
		token = strtok(NULL, delim);
		i.node_p = findNode(netlist.netNames, token, netlist.netNames.size());

		// node n
		token = strtok(NULL, delim);
		i.node_n = findNode(netlist.netNames, token, netlist.netNames.size());

		// val

		token = strtok(NULL, delim);
		if (strcmp(token, "DC") == 0) {
			token = strtok(NULL, delim);
			val = atof(token);
		}

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		i.val = numPrefix(val, type);

		netlist.idcList.push_back(i);
	}

	else if (type == 'G') {
		VCCS iv;

		// get name
		token = strtok(line + 1, delim);
		iv.name = new char[strlen(token) + 1];
		strcpy(iv.name, token);

		// node ip
		token = strtok(NULL, delim);
		iv.ip = findNode(netlist.netNames, token, netlist.netNames.size());

		// node in
		token = strtok(NULL, delim);
		iv.in = findNode(netlist.netNames, token, netlist.netNames.size());

		// node vp
		token = strtok(NULL, delim);
		iv.vp = findNode(netlist.netNames, token, netlist.netNames.size());

		// node vn
		token = strtok(NULL, delim);
		iv.vn = findNode(netlist.netNames, token, netlist.netNames.size());

		// gain
		token = strtok(NULL, delim);
		val = atof(token);

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		iv.g = numPrefix(val, type);

		netlist.vccsList.push_back(iv);

	}

	return 0;
}

int findNode(vector<char*> &nodeList, char* name, int n) {
	int i = 0;
	for (i = 0; i < n; i++) {
		if (strcmp(name, nodeList[i]) == 0) break;
	}
	if (i < n) {
		return i;
	}
	else {
		char* new_name = new char[strlen(name) + 1];
		strcpy(new_name, name);
		nodeList.push_back(new_name);
		return n;
	}
}

float numPrefix(float num, char prefix) {
	// Applies number affix to parsed value
	// e.g. 1k = 1000
	switch (prefix) {
		//negative exponents
		case 'd':
			return num * 1e-1;
		case 'c':
			return num * 1e-2;
		case 'm':
			return num * 1e-3;
		case 'u':
			return num * 1e-6;
		case 'n':
			return num * 1e-9;
		case 'p':
			return num * 1e-12;
		case 'f':
			return num * 1e-15;
		case 'a':
			return num * 1e-18;
		case 'z':
			return num * 1e-21;
		case 'y':
			return num * 1e-24;
		// positive exponents
		case 'h':
			return num * 1e2;
		case 'k':
			return num * 1e3;
		case 'M':
			return num * 1e6;
		case 'G':
			return num * 1e9;
		case 'T':
			return num * 1e12;
		case 'P':
			return num * 1e15;
		case 'E':
			return num * 1e18;
		case 'Z':
			return num * 1e21;
		case 'Y':
			return num * 1e24;
		// default is do nothing, return original value
		default:
			return num;
	}
}