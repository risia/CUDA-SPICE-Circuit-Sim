#include "spice.h"

int parseNetlist(char* filepath, Netlist &netlist) {

	ifstream inFile(filepath);
	char oneline[MAX_LINE];


	vector<Idc>().swap(netlist.idcList);
	vector<VCCS>().swap(netlist.vccsList);
	vector<Vdc>().swap(netlist.vdcList);
	vector<Resistor>().swap(netlist.rList);
	vector<Transistor>().swap(netlist.mosList);
	vector<Model*>().swap(netlist.modelList);
	vector<char*>().swap(netlist.netNames);

	// Default ground node
	netlist.netNames.push_back("gnd");

	// Default test model
	Model* M_ptr = new Model();

	netlist.modelList.push_back(M_ptr);


	while (inFile)
	{
		inFile.getline(oneline, MAX_LINE);
		if (oneline[0] == '*') continue;
		if (oneline[0] != '.') parseElement(oneline, netlist);


		else cout << oneline << endl;
	}

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
	else if (type == 'M') {
		Transistor T;

		// get name
		token = strtok(line + 1, delim);
		T.name = new char[strlen(token) + 1];
		strcpy(T.name, token);

		// drain
		token = strtok(NULL, delim);
		T.d = findNode(netlist.netNames, token, netlist.netNames.size());

		// gate
		token = strtok(NULL, delim);
		T.g = findNode(netlist.netNames, token, netlist.netNames.size());

		// source
		token = strtok(NULL, delim);
		T.s = findNode(netlist.netNames, token, netlist.netNames.size());

		// bulk
		token = strtok(NULL, delim);
		T.b = findNode(netlist.netNames, token, netlist.netNames.size());

		// Model
		token = strtok(NULL, delim);
		T.model = findModel(netlist.modelList, token, netlist.modelList.size());

		// If null may need to throw error. For now:
		if (T.model == NULL) T.model = netlist.modelList[0]; // default model

		// Length
		token = strtok(NULL, delim);
		val = atof(token + 2); // skip L=

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		T.l = numPrefix(val, type);

		// Width
		token = strtok(NULL, delim);
		val = atof(token + 2); // skip W=

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		T.w = numPrefix(val, type);

		netlist.mosList.push_back(T);
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

Model* findModel(vector<Model*> &modelList, char* name, int n) {
	int i = 0;
	for (i = 0; i < n; i++) {
		if (strcmp(name, modelList[i]->name) == 0) break;
	}
	if (i < n) {
		return modelList[i];
	}
	else {
		return NULL;
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
		case 'U':
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