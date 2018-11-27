#include "spice.h"

int parseNetlist(char* filepath, Netlist &netlist) {

	ifstream inFile(filepath);
	char line[MAX_LINE];

	// Initialize element arrays
	vector<Element>().swap(netlist.elements);
	vector<Element>().swap(netlist.vdcList);
	vector<Element>().swap(netlist.active_elem);

	vector<Model*>().swap(netlist.modelList);
	vector<char*>().swap(netlist.netNames);

	// Default ground node
	netlist.netNames.push_back("gnd");

	// Default test model
	Model* M_ptr = new Model();

	netlist.modelList.push_back(M_ptr);


	while (inFile)
	{
		inFile.getline(line, MAX_LINE);
		if (line[0] == '*' || line[0] == '\0') continue;
		if (line[0] != '.') parseElement(line, netlist);


		else cout << line << endl;
	}

	inFile.close();

	return 0;
}

int parseElement(char* line, Netlist& netlist) {

	char type = line[0];

	char* token;
	char* delim = " ";

	float val;
	int node;

	Element e;

	e.type = type;

	// get name
	token = strtok(line + 1, delim);
	e.name = new char[strlen(token) + 1];
	strcpy(e.name, token);

	// node 1 (transistor drain)
	token = strtok(NULL, delim);
	node = findNode(netlist.netNames, token, netlist.netNames.size());
	e.nodes.push_back(node);

	// node 2 (transistor gate)
	token = strtok(NULL, delim);
	node = findNode(netlist.netNames, token, netlist.netNames.size());
	e.nodes.push_back(node);

	// parse resistor
	if (type == 'R') {
		// value
		token = strtok(NULL, delim);
		val = atof(token);

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		e.params.push_back(numPrefix(val, type));
	}
	// parse VDC
	else if (type == 'V') {
		// DC value
		token = strtok(NULL, delim);
		if (strcmp(token, "DC") == 0) {
			token = strtok(NULL, delim);
			val = atof(token);
		}

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		e.params.push_back(numPrefix(val, type));

		e.nodes.shrink_to_fit();
		e.params.shrink_to_fit();
		netlist.vdcList.push_back(e);
		return 0;
	}
	// Parsse IDC
	else if (type == 'I') {
		// val
		token = strtok(NULL, delim);
		if (strcmp(token, "DC") == 0) {
			token = strtok(NULL, delim);
			val = atof(token);
		}

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		e.params.push_back(numPrefix(val, type));
	}

	else if (type == 'G') {

		// node vp
		token = strtok(NULL, delim);
		node = findNode(netlist.netNames, token, netlist.netNames.size());
		e.nodes.push_back(node);

		// node vn
		token = strtok(NULL, delim);
		node = findNode(netlist.netNames, token, netlist.netNames.size());
		e.nodes.push_back(node);

		// gain
		token = strtok(NULL, delim);
		val = atof(token);

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		e.params.push_back(numPrefix(val, type));
	}
	else if (type == 'M') {
		// source
		token = strtok(NULL, delim);
		node = findNode(netlist.netNames, token, netlist.netNames.size());
		e.nodes.push_back(node);

		// bulk
		token = strtok(NULL, delim);
		node = findNode(netlist.netNames, token, netlist.netNames.size());
		e.nodes.push_back(node);

		// Model
		token = strtok(NULL, delim);
		e.model = findModel(netlist.modelList, token, netlist.modelList.size());

		// If null may need to throw error. For now:
		if (e.model == NULL) e.model = netlist.modelList[0]; // default model

		// Length
		token = strtok(NULL, delim);
		val = atof(token + 2); // skip L=

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		e.params.push_back(numPrefix(val, type));

		// Width
		token = strtok(NULL, delim);
		val = atof(token + 2); // skip W=

		// apply prefix multiplier
		type = token[strlen(token) - 1]; // reusing variable
		e.params.push_back(numPrefix(val, type));

		e.nodes.shrink_to_fit();
		e.params.shrink_to_fit();
		netlist.active_elem.push_back(e);
		return 0;
	}
	else return -1;

	e.nodes.shrink_to_fit();
	e.params.shrink_to_fit();
	netlist.elements.push_back(e);


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