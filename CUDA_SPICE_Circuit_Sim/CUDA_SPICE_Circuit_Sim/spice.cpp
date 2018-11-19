#include "spice.h"

int parseNetlist(char* filepath, Netlist &netlist) {

	ifstream inFile(filepath);
	char oneline[MAX_LINE];


	vector<Idc>().swap(netlist.idcList);
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

	inFile.close();

	return 0;
}

int parseElement(char* line, Netlist& netlist) {

	char type = line[0];

	char* token;
	char* delim = " ";

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

		r.val = atof(token);

		netlist.rList.push_back(r);
	}
	// parse VDC
	if (type == 'V') {
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
			v.val = atof(token);
		}

		netlist.vdcList.push_back(v);
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