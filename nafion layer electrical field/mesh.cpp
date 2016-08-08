#include "mesh.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>

mesh::mesh(int fm, double fdr0, double fdr, int fn, double fdz0, double fdz) :
	m(fm), dr0(fdr0), dr(fdr), n(fn), dz0(fdz0), dz(fdz), R(n, m), RR(n, m), Z(n, m), ZZ(n, m)
{
	
}

mesh::~mesh()
{
	
}


void mesh::PrintMesh(string ex_file_name)
{
	ofstream myfile1(ex_file_name + " mesh-R.txt");
	ofstream myfile2(ex_file_name + " mesh-Z.txt");
	ofstream myfile3(ex_file_name + " mesh parameters.txt");

	if (myfile1.is_open() && myfile2.is_open()) {

		myfile1 << R;
		myfile2 << Z;

		myfile1.close();
		myfile2.close();
	}
	else {
		cout << " unable to open mesh data file" << endl;
		exit(EXIT_FAILURE);
	}

	myfile3 << "number of R nodes: " << m
		<< "\ndr0, cm: " << dr0
		<< "\ndr, cm: " << dr
		<< "number of z nodes: " << n
		<< "\ndz0, cm: " << dz0
		<< "\ndz, cm: " << dz;
}


const vector<long> mesh::GetMeshSize() const
{
	return vector<long> {m, n};
}