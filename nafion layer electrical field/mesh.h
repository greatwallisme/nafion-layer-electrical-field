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

	// Export the mesh grid
	void PrintMesh(string ex_file_name);

	const vector<long> GetMeshSize() const;

	const long Getmxn() const {return m*n;}

	const long m; // r axis node number

	const long n; // z axis node number

private:
	// define x and y interval
	
	const double dr0; // , cm
	const double dr; // , cm
	const double dz; // , cm
	const double dz0; // , cm

	// mesh define
	Eigen::MatrixXd R; // r node for plot graph
	Eigen::MatrixXd Z; // z node for plot graph
	Eigen::MatrixXd RR; // the r position of the centre of each box
	Eigen::MatrixXd ZZ; // the z position of the centre of each box
};