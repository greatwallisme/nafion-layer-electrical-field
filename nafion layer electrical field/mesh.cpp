#include "mesh.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>

mesh::mesh(int fm, double fdr, int fn, double fdz) :
	m(fm), dr(fdr), n(fn), dz(fdz), R(n, m), RR(n, m), Z(n, m), ZZ(n, m)
{
	for (long i = 0; i < m; ++i) {
		for (long j = 0; j < n; ++j) {
			R(j, i) = i*dr;
			Z(j, i) = j*dz;
			RR(j, i) = R(j, i) + 0.5*dr;
			ZZ(j, i) = Z(j, i) + 0.5*dz;
		}
	}
}

mesh::~mesh()
{
	
}


void mesh::PrintMesh(std::string ex_file_name)
{
	std::ofstream myfile1(ex_file_name + " mesh-R.txt");
	std::ofstream myfile2(ex_file_name + " mesh-Z.txt");
	std::ofstream myfile3(ex_file_name + " mesh parameters.txt");

	if (myfile1.is_open() && myfile2.is_open()) {

		myfile1 << R;
		myfile2 << Z;

		myfile1.close();
		myfile2.close();
	}
	else {
		std::cout << " unable to open mesh data file" << std::endl;
		exit(EXIT_FAILURE);
	}

	myfile3 << "number of R nodes: " << m
		<< "\ndr, cm: " << dr
		<< "\nnumber of z nodes: " << n
		<< "\ndz, cm: " << dz;
}


const std::vector<long> mesh::GetMeshSize() const
{
	return std::vector<long> {m, n};
}