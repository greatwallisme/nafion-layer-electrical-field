#pragma once
#include <Eigen/Sparse>
#include "mesh.h"
#include <map>
#include "EnumStruct.h"
typedef Eigen::SparseVector<double> SpVectorXd;
typedef Eigen::SparseMatrix<double> SpMatrixXd;

class Ion
{
public:
	Ion(double fD, int fZ, double Density, mesh& Mesh);
	const double D;
	const int Z;
	const double Cinitial;

	Eigen::MatrixXd DensityN; // for ions species, it stores the concentration values; for potential, it stores the potential values

	void PrintDense(std::string fileName);
};

class IonSystem
{
public:
	IonSystem(double fEpsilon_r, double Epsilon_0, std::map<SpeciesEnum::Species, Ion*>& fIons, Ion& fPotential, Ion& fImmobileCharge = Ion(0.0, 0, 0.0, mesh(1, 1, 1, 1)));
	//IonSystem(double fEpsilon_r, double Epsilon_0, Ion& fReactant, Ion& fProduct, Ion& fSupportCation, Ion& fSupportAnion, Ion& fPotential, Ion& fImmobileCharge = Ion(0.0, 0, 0.0, mesh(1, 1, 1, 1)));

	//Ion& Reactant; // diffusion coefficient of the reactant
	//Ion& Product; // diffusion coefficient of the product
	//Ion& SupportCation; // diffusion coefficient of the supporting cation
	//Ion& SupportAnion; // diffusion coefficient of the supporting anion
	Ion& ImmobileCharge; //immobile charge, default: D = 0, z = 0
	Ion& Potential;
	const double ReciprocalEpsilon_rEpsilon_0;
	const double CxZImmobileCharge;
	int IonNum;
	
	Ion& operator[](SpeciesEnum::Species i) const { return *(Ions[i]); }
private:
	const double Epsilon_r;
	const double Epsilon_0;
	std::map<SpeciesEnum::Species, Ion*>& Ions;
};

