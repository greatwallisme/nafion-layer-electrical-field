#pragma once
#include <Eigen/Sparse>
#include "mesh.h"
typedef Eigen::SparseVector<double> SpVectorXd;
typedef Eigen::SparseMatrix<double> SpMatrixXd;

class Ion
{
public:
	Ion(double fD, int fZ, double C, mesh& Mesh);
	const double D;
	const int Z;
	double Cinitial;

	Eigen::MatrixXd DenseN; // for ions species, it stores the concentration values; for potential, it stores the potential values

	void PrintDense(string fileName);
};

class IonSystem
{
public:
	IonSystem(double fEpsilon_r, double Epsilon_0, Ion& fReactant, Ion& fProduct, Ion& fSupportCation, Ion& fSupportAnion, Ion& fImmobileCharge = Ion(0.0, 0, 0.0, mesh(1, 1, 1, 1, 1, 1)), double fCImmobileCharge = 0.0);

	const Ion& Reactant; // diffusion coefficient of the reactant
	const Ion& Product; // diffusion coefficient of the product
	const Ion& SupportCation; // diffusion coefficient of the supporting cation
	const Ion& SupportAnion; // diffusion coefficient of the supporting anion
	const Ion& ImmobileCharge; //immobile charge, default: D = 0, z = 0

	const double ReciprocalEpsilon_rEpsilon_0;
	const double CImmobileCharge; // concentration of immobile charge ions
	const double CxZImmobileCharge;

private:
	const double Epsilon_r;
	const double Epsilon_0;
};

