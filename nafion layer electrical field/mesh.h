#pragma once
#include <string>
#include <Eigen/Dense>
#include <vector>

using namespace std;

class mesh
{
	friend class solver;
public:
	mesh(int fm, double fdr0, double fdr, int fn, double fdz0, double fdz);
	~mesh();

	// Pass Cren to Creo
	void Ca2Cb(string Ca_name, string Cb_name);
	// Export the mesh grid
	void PrintMesh(string ex_file_name);
	// Export the concentrations
	void PrintConcentration(string target_concentration_name, string ex_file_name);

	const vector<unsigned long> GetMeshSize() const;

	const unsigned long Getmxn() const {return m*n;}

private:
	// define x and y interval
	const unsigned long m; // r axis node number
	const double dr0; // , cm
	const double dr; // , cm
	const unsigned long n; // z axis node number
	const double dz; // , cm
	const double dz0; // , cm

	// mesh define
	Eigen::MatrixXd Cren;
	Eigen::MatrixXd Creo;
	Eigen::MatrixXd Cprn;
	Eigen::MatrixXd Cpro;
	Eigen::MatrixXd Cann;
	Eigen::MatrixXd Cano;
	Eigen::MatrixXd Ccan;
	Eigen::MatrixXd Ccao;
	Eigen::MatrixXd Ptln; //potential field new
	Eigen::MatrixXd Ptlo; // potential field old
	Eigen::MatrixXd R; // r node for plot graph
	Eigen::MatrixXd Z; // z node for plot graph
	Eigen::MatrixXd RR; // the r position of the centre of each box
	Eigen::MatrixXd ZZ; // the z position of the centre of each box
};