#include "spice.h"

int parseNetlist(char* filepath, Netlist* netlist) {

	ifstream inFile(filepath);
	char line[MAX_LINE];

	// Initialize element arrays
	// So in the future users could run more
	// than one file per session
	vector<Element>().swap(netlist->elements);
	vector<Element>().swap(netlist->vdcList);
	vector<Element>().swap(netlist->active_elem);
	vector<Model*>().swap(netlist->modelList);
	vector<char*>().swap(netlist->netNames);

	// Default ground node
	netlist->netNames.push_back("gnd");

	// Default test models
	Model* NM_ptr = new Model(); // nmos model
	Model* PM_ptr = new Model(); // pmos model

	// PMOS Model (for testing basically same as NMOS)
	PM_ptr->name = "P";
	PM_ptr->type = 'p';
	PM_ptr->u0 = 540.0f;
	PM_ptr->vt0 = -0.7f; // just flipped this sign, really

	netlist->modelList.push_back(NM_ptr);
	netlist->modelList.push_back(PM_ptr);


	while (inFile)
	{
		inFile.getline(line, MAX_LINE);
		// Skip comments and empty lines
		if (line[0] == '*' || line[0] == '\0') continue;

		// Parse elements
		if (line[0] != '.') parseElement(line, netlist);
		// Print commands for testing for now, we'll parse later
		else cout << line << endl;
	}

	inFile.close();

	netlist->elements.shrink_to_fit();
	netlist->active_elem.shrink_to_fit();
	netlist->netNames.shrink_to_fit();
	netlist->vdcList.shrink_to_fit();
	netlist->vdcList.shrink_to_fit();

	return 0;
}

void parseNodes(int n, Netlist* netlist, Element* e, char* delim) {
	char* token;
	int node;
	for (int i = 0; i < n; i++) {
		token = strtok(NULL, delim);
		node = findNode(netlist->netNames, token, netlist->netNames.size());
		e->nodes.push_back(node);
	}
}

void parseValues(int n, int skip, Netlist* netlist, Element* e, char* delim) {
	char* token;
	float val;
	char prefix;
	for (int i = 0; i < n; i++) {
		token = strtok(NULL, delim);
		val = atof(token + skip);

		// apply prefix multiplier
		prefix = token[strlen(token) - 1];
		e->params.push_back(numPrefix(val, prefix));
	}
}

int parseElement(char* line, Netlist* netlist) {

	char type = line[0];

	// check if type allowed, return -1 if not
	bool allowed = false;
	for (int i = 0; i < N_TYPES; i++) {
		if (type == TYPES[i]) {
			allowed = true;
			break;
		}
	}
	if (allowed == false) return -1;

	char* token;
	char* delim = " ";

	float val;
	int node;

	Element e;

	e.type = type;

	// Common Parameters beginning Line
	// get name
	token = strtok(line + 1, delim);
	e.name = new char[strlen(token) + 1];
	strcpy(e.name, token);

	// Diverging parts
	// parse resistor
	if (type == 'R') {
		parseNodes(2, netlist, &e, delim);
		parseValues(1, 0, netlist, &e, delim);

		// If it's a short/ideal wire:
		if (e.params[0] == 0.0f) {
			// Make it a 0V "source" instead
			e.nodes.shrink_to_fit();
			e.params.shrink_to_fit();
			netlist->vdcList.push_back(e);
			return 0;
		}
	}
	// parse VDC
	else if (type == 'V') {
		parseNodes(2, netlist, &e, delim);
		// DC value
		token = strtok(NULL, delim);
		if (strcmp(token, "DC") == 0) {
			parseValues(1, 0, netlist, &e, delim);
		}

		e.nodes.shrink_to_fit();
		e.params.shrink_to_fit();
		netlist->vdcList.push_back(e);
		return 0;
	}
	// Parsse IDC
	else if (type == 'I') {
		parseNodes(2, netlist, &e, delim);
		// val
		token = strtok(NULL, delim);
		if (strcmp(token, "DC") == 0) {
			parseValues(1, 0, netlist, &e, delim);
		}
	}

	else if (type == 'G') {
		// nodes
		parseNodes(4, netlist, &e, delim);
		// gain
		parseValues(1, 0, netlist, &e, delim);
	}
	else if (type == 'M') {
		parseNodes(4, netlist, &e, delim);

		// Model
		token = strtok(NULL, delim);
		e.model = findModel(netlist->modelList, token, netlist->modelList.size());

		// If null may need to throw error. For now:
		if (e.model == NULL) e.model = netlist->modelList[0]; // default model

		// Length & Width
		parseValues(2, 2, netlist, &e, delim);

		e.nodes.shrink_to_fit();
		e.params.shrink_to_fit();
		netlist->active_elem.push_back(e);
		return 0;
	}

	e.nodes.shrink_to_fit();
	e.params.shrink_to_fit();
	netlist->elements.push_back(e);


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