#include "mesh.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>

mesh::mesh(int fm, double fdr0, double fdr, int fn, double fdz0, double fdz) :
	m(fm), dr0(fdr0), dr(fdr), n(fn), dz0(fdz0), dz(fdz),
	Cren(n, m), Creo(n, m), Cprn(n, m), Cpro(n, m), Cann(n, m), Cano(n, m), Ccan(n, m), Ccao(n, m), Ptln(n, m), Ptlo(n, m),
	R(n, m), RR(n, m), Z(n, m), ZZ(n, m)
{
	
}

mesh::~mesh()
{
	
}

void mesh::Ca2Cb(string Ca_name, string Cb_name)
{
	Eigen::MatrixXd* Ca, *Cb;
	if (Ca_name == "Cren") {
		Ca = &Cren;
	}
	else if (Ca_name == "Creo") {
		Ca = &Creo;
	}
	else if (Ca_name == "Cprn") {
		Ca = &Cprn;
	}
	else if (Ca_name == "Cpro") {
		Ca = &Cpro;
	}
	if (Ca_name == "Cann") {
		Ca = &Cann;
	}
	else if (Ca_name == "Cano") {
		Ca = &Cano;
	}
	else if (Ca_name == "Ccan") {
		Ca = &Ccan;
	}
	else if (Ca_name == "Ccao") {
		Ca = &Ccao;
	}
	else if (Ca_name == "Ptln") {
		Ca = &Ptln;
	}
	else if (Ca_name == "Ptlo") {
		Ca = &Ptlo;
	}
	else {
		std::cout << "Ca2Cb cannot find Ca: " << Ca_name << endl;
		exit(EXIT_FAILURE);

	}

	if (Cb_name == "Cren") {
		Cb = &Cren;
	}
	else if (Cb_name == "Creo") {
		Cb = &Creo;
	}
	else if (Cb_name == "Cprn") {
		Cb = &Cprn;
	}
	else if (Cb_name == "Cpro") {
		Cb = &Cpro;
	}
	if (Cb_name == "Cann") {
		Cb = &Cann;
	}
	else if (Cb_name == "Cano") {
		Cb = &Cano;
	}
	else if (Cb_name == "Ccan") {
		Cb = &Ccan;
	}
	else if (Cb_name == "Ccao") {
		Cb = &Ccao;
	}
	else if (Cb_name == "Ptln") {
		Cb = &Ptln;
	}
	else if (Cb_name == "Ptlo") {
		Cb = &Ptlo;
	}
	else {
		std::cout << "Ca2Cb cannot find Cb: " << Cb_name << endl;
		exit(EXIT_FAILURE);
	}

	*Cb = *Ca;
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

void mesh::PrintConcentration(string target_concentration_name, string ex_file_name)
{
	Eigen::MatrixXd* C;
	if (target_concentration_name == "Cren") {
		C = &Cren;
	}
	else if (target_concentration_name == "Creo") {
		C = &Creo;
	}
	else if (target_concentration_name == "Cprn") {
		C = &Cprn;
	}
	else if (target_concentration_name == "Cpro") {
		C = &Cpro;
	}
	if (target_concentration_name == "Cann") {
		C = &Cann;
	}
	else if (target_concentration_name == "Cano") {
		C = &Cano;
	}
	else if (target_concentration_name == "Ccan") {
		C = &Ccan;
	}
	else if (target_concentration_name == "Ccao") {
		C = &Ccao;
	}
	else if (target_concentration_name == "Ptln") {
		C = &Ptln;
	}
	else if (target_concentration_name == "Ptlo") {
		C = &Ptlo;
	}
	else {
		std::cout << "print_concentration cannot find target_concentration_name: " << target_concentration_name << endl;
		exit(EXIT_FAILURE);
	}

	ofstream myfile(ex_file_name + ".txt");
	if (myfile.is_open()) {
		myfile << C;
		myfile.close();
	}
	else {
		std::cout << "print_concentration cannot open " << target_concentration_name << ".txt" << endl;
	}
}

const vector<long> mesh::GetMeshSize() const
{
	return vector<long> {m, n};
}