#pragma once

#include "spice.h"

 /* 
 
 starting with element list of just resistors to test
 setting up matrices and solving

 */

Resistor* testRList1() {
	Resistor r1;
	r1.name = "R1";
	r1.val = 1000.0f;
	r1.node1 = 1;
	r1.node2 = 2;

	Resistor r2;
	r2.name = "R2";
	r2.val = 1000.0f;
	r2.node1 = 1;
	r2.node2 = 2;

	Resistor r3;
	r3.name = "R3";
	r3.val = 1000.0f;
	r3.node1 = 0;
	r3.node2 = 2;

	Resistor r4;
	r4.name = "R4";
	r4.val = 1000.0f;
	r4.node1 = 3;
	r4.node2 = 2;

	Resistor r5;
	r5.name = "R5";
	r5.val = 1000.0f;
	r5.node1 = 0;
	r5.node2 = 3;

	Resistor* rList = (Resistor*)malloc(5  * sizeof(Resistor));
	rList[0] = r1;
	rList[1] = r2;
	rList[2] = r3;
	rList[3] = r4;
	rList[4] = r5;

	return rList;
}

Resistor* testRList2() {

	Resistor r1;
	r1.name = "R1";
	r1.val = 1.0f;
	r1.node1 = 1;
	r1.node2 = 0;

	Resistor r2;
	r2.name = "R2";
	r2.val = 1.0f;
	r2.node1 = 1;
	r2.node2 = 0;

	Resistor r3;
	r3.name = "R3";
	r3.val = 1.0f;
	r3.node1 = 1;
	r3.node2 = 2;

	Resistor r4;
	r4.name = "R4";
	r4.val = 1.0f;
	r4.node1 = 3;
	r4.node2 = 4;

	Resistor r5;
	r5.name = "R5";
	r5.val = 1.0f;
	r5.node1 = 0;
	r5.node2 = 4;

	Resistor r6;
	r6.name = "R6";
	r6.val = 1.0f;
	r6.node1 = 0;
	r6.node2 = 4;

	Resistor* rList = (Resistor*)malloc(6 * sizeof(Resistor));
	rList[0] = r1;
	rList[1] = r2;
	rList[2] = r3;
	rList[3] = r4;
	rList[4] = r5;
	rList[5] = r6;

	return rList;
}

Vdc* testVList1() {
	Vdc v0;
	v0.name = "VDC0";
	v0.val = 1;
	v0.node_n = 0;
	v0.node_p = 1;

	Vdc* vList = (Vdc*)malloc(sizeof(Vdc));
	vList[0] = v0;

	return vList;
}

Vdc* testVList2() {
	Vdc v0;
	v0.name = "VDC0";
	v0.val = 1;
	v0.node_n = 2;
	v0.node_p = 3;

	Vdc* vList = (Vdc*)malloc(sizeof(Vdc));
	vList[0] = v0;

	return vList;
}

Idc* testIList1() {
	Idc i0;
	i0.name = "IDC0";
	i0.val = 0.001;
	i0.node_n = 0;
	i0.node_p = 2;

	Idc* iList = (Idc*)malloc(sizeof(Idc));
	iList[0] = i0;

	return iList;
}
