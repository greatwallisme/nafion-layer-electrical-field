#pragma once
#include <string>
#include <Eigen/Dense>
#include <vector>


class mesh
{
	friend class solver;
public:
	mesh(int fm, double fdr, int fn, double fdz);
	~mesh();

	// Export the mesh grid
	void PrintMesh(std::string ex_file_name);

	const std::vector<long> GetMeshSize() const;

	const long Getmxn() const {return m*n;}

	const long m; // r axis node number

	const long n; // z axis node number

private:
	// define x and y interval
	const double dr; // , cm
	const double dz; // , cm
	

	// mesh define
	Eigen::MatrixXd R; // r node for plot graph
	Eigen::MatrixXd Z; // z node for plot graph
	Eigen::MatrixXd RR; // the r position of the centre of each box
	Eigen::MatrixXd ZZ; // the z position of the centre of each box
};