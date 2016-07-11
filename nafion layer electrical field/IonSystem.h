#pragma once
#include <Eigen/Sparse>
typedef Eigen::SparseVector<double> SpVectorXd;
typedef Eigen::SparseMatrix<double> SpMatrixXd;

class IonSystem
{
public:
	IonSystem(double fEpsilon_r, double Epsilon_0, Ion& fReactant, Ion& fProduct, Ion& fSupportCation, Ion& fSupportAnion, Ion& fImmobileCharge = Ion(0.0, 0), double fCImmobileCharge = 0.0);
	const Ion& GetReactant() const { return Reactant; }
	const Ion& GetProduct() const { return Product; }
	const Ion& GetSupportCation() const { return SupportCation; }
	const Ion& GetSupportAnion() const { return SupportAnion; }
	const Ion& GetImmobileCharge() const { return ImmobileCharge; }
	const double GetReciprocalEpsilon_rEpsilon_0() const { return ReciprocalEpsilon_rEpsilon_0; }
	const double GetCxZImmobileCharge() const { return CxZImmobileCharge; }

private:
	const Ion& Reactant; // diffusion coefficient of the reactant
	const Ion& Product; // diffusion coefficient of the product
	const Ion& SupportCation; // diffusion coefficient of the supporting cation
	const Ion& SupportAnion; // diffusion coefficient of the supporting anion
	const Ion& ImmobileCharge; //immobile charge, default: D = 0, z = 0
	const double ReciprocalEpsilon_rEpsilon_0;
	const double Epsilon_r;
	const double Epsilon_0;
	const double CImmobileCharge; // concentration of immobile charge ions
	double CxZImmobileCharge;
};

class Ion
{
public:
	Ion(double fD, int fZ);
	double GetD() const { return D; }
	double GetZ() const { return Z; }
private:
	const double D;
	const double Z;
};