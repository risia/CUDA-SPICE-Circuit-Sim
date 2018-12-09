#include "spice.h"

int parseNetlist(const char* filepath, Netlist* netlist) {

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
		if (line[0] == '.') {
			cout << line << endl;
			parseCmd(line, netlist, &inFile);
		}
		// Parse elements
		else parseElement(line, netlist);
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
		if (token == NULL) e->params.push_back(0.0f);

		else {
			val = atof(token + skip);

			// apply prefix multiplier
			prefix = token[strlen(token) - 1];
			e->params.push_back(numPrefix(val, prefix));
		}
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
	char* delim = " ()";

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
	// parse Voltage sources
	else if (type == 'V') {
		parseNodes(2, netlist, &e, delim);
		// DC value
		token = strlwr(strtok(NULL, delim));
		if (strcmp(token, "dc") == 0) {
			parseValues(1, 0, netlist, &e, delim);
		}
		else if (strcmp(token, "sin") == 0) {
			e.type = 'S'; // change type so identify for transient
			parseValues(6, 0, netlist, &e, delim);
		}
		else if (strcmp(token, "pulse") == 0) {
			e.type = 'P';
			e.params.push_back(0.0f); // DC val placeholder
			parseValues(7, 0, netlist, &e, delim);

			// check if actual DC value specified
			token = strlwr(strtok(NULL, delim));
			if (strcmp(token, "dc") == 0) {
				parseValues(1, 0, netlist, &e, delim);
			}
			e.params[0] = e.params.back();
			e.params.pop_back();
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
		token = strlwr(strtok(NULL, delim));
		if (strcmp(token, "dc") == 0) {
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

		//cout << "Transistor " << e.name << " model: " << e.model->name << "\n";

		// If null may need to throw error. For now:
		if (e.model == NULL) {
			e.model = netlist->modelList[0]; // default model
			cout << "\nMissing model for MOSFET named:" << e.name << "\n";
		}

		// Length & Width
		parseValues(2, 2, netlist, &e, delim);

		e.nodes.shrink_to_fit();
		e.params.shrink_to_fit();
		netlist->active_elem.push_back(e);


		// Cgb0
		if (fabs(e.model->CGBO) > 0.0f) {
			Element C;
			C.type = 'C';
			C.nodes.push_back(e.nodes[1]);
			C.nodes.push_back(e.nodes[3]);
			C.params.push_back(e.model->CGBO * e.params[0]);
			netlist->elements.push_back(C);
		}
		// Cgs0
		if (fabs(e.model->CGSO) > 0.0f) {
			Element C;
			C.type = 'C';
			C.nodes.push_back(e.nodes[1]);
			C.nodes.push_back(e.nodes[2]);
			C.params.push_back(e.model->CGSO * e.params[1]);
			netlist->elements.push_back(C);
		}
		// Cgd0
		if (fabs(e.model->CGDO) > 0.0f) {
			Element C;
			C.type = 'C';
			C.nodes.push_back(e.nodes[1]);
			C.nodes.push_back(e.nodes[0]);
			C.params.push_back(e.model->CGDO * e.params[1]);
			netlist->elements.push_back(C);
		}

		return 0;
	}
	else if (type == 'C') {
		parseNodes(2, netlist, &e, delim);
		parseValues(1, 0, netlist, &e, delim);
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



void parseCmd(char* line, Netlist* netlist, ifstream* file) {
	char* token;

	// make copy of line,
	// because strtok corrupt data
	char copy[MAX_LINE];
	strcpy(copy, line);

	char* delim = " ()=";

	token = strlwr(strtok(line + 1, delim));

	// parse includes file
	if (strcmp(token, "include") == 0) {
		token = strtok(copy + 9, "\"\'");
		cout << "Filestring: " << token << "\n";
		parseNetlist(token, netlist);
	}

	// The glorious model file
	else if (strcmp(token, "model") == 0) {
		Model* M_ptr = new Model();
		netlist->modelList.push_back(M_ptr);
		// format:
		// .model name type level parameters...
		// '+' indicates continuation onto next line

		// NAME
		token = strtok(NULL, delim);
		M_ptr->name = new char[strlen(token) + 1];
		strcpy(M_ptr->name, token);

		// TYPE
		token = strlwr(strtok(NULL, delim));
		if (strcmp(token, "nmos") == 0) M_ptr->type = 'n';
		else if (strcmp(token, "pmos") == 0) M_ptr->type = 'p';

		float val;
		char peek = file->peek();

		while (peek == '\n' && file) {
			file->getline(line, MAX_LINE);
			peek = file->peek();
		}

		while (peek == '+') {
			file->getline(line, MAX_LINE);
			// parse parameters
			token = strtok(line + 1, delim);
			while (token != NULL) {
				token = strlwr(token);
				if (strcmp(token, "toxe") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->tox = atof(token);

					//cout << std::scientific << M_ptr->tox << "\n";
				}
				else if (strcmp(token, "vth0") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->vt0 = atof(token);

					//cout << std::scientific << M_ptr->vt0 << "\n";
				}
				else if (strcmp(token, "u0") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->u0 = atof(token);

					//cout << std::scientific << M_ptr->u0 << "\n";
				}
				else if (strcmp(token, "pclm") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->pclm = atof(token);

					//cout << std::scientific << M_ptr->pclm << "\n";
				}
				else if (strcmp(token, "vsat") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->vsat = atof(token);

					//cout << std::scientific << M_ptr->vsat << "\n";
				}
				else if (strcmp(token, "nfactor") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->nfactor = atof(token);

					//cout << std::scientific << M_ptr->nfactor << "\n";
				}
				else if (strcmp(token, "cgso") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->CGSO = atof(token);

					//cout << std::scientific << M_ptr->CGSO << "\n";
				}
				else if (strcmp(token, "cgdo") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->CGDO = atof(token);

					//cout << std::scientific << M_ptr->CGDO << "\n";
				}
				else if (strcmp(token, "cgbo") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->CGBO = atof(token);

					//cout << std::scientific << M_ptr->CGBO << "\n";
				}
				else if (strcmp(token, "epsrox") == 0) {
					//cout << "Parameter: " << token << " = ";

					token = strtok(NULL, delim);
					M_ptr->epsrox = atof(token);

					//cout << std::scientific << M_ptr->epsrox << "\n";
				}
				
				token = strtok(NULL, delim);
			}

			// check next line;
			peek = file->peek();

			while (peek == '\n' && file) {
				file->getline(line, MAX_LINE);
				peek = file->peek();
			}
		}
	}
}