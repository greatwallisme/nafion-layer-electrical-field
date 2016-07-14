#pragma once
#include "mesh.h"
#include "IonSystem.h"
#include "potential_signal.h"
#include <Eigen/Sparse>
#include "thermodynamics.h"

typedef Eigen::SparseMatrix<double> SpMatrixXd;
typedef Eigen::SparseVector<double> SpVectorXd;
typedef Eigen::Triplet<double> Td;

class solver
{
public:
	solver(mesh& fmembrane, mesh& fsolution, const IonSystem& fmembraneIons, const IonSystem& fsolutionIons, PotentialSignal& fSignal, const nernst_equation& fThermo);
	void initialise();
	
private:
	SpMatrixXd MatrixA; // Ax = b for Fc, each element inside is initialised as 0
	Eigen::VectorXd arrayb; // Ax = b for Fc; each element inside is initialised as 0
	Eigen::VectorXd dX; // the change of x in an iteration
	Eigen::VectorXd X; // the calculated results;
	Eigen::VectorXd F; // values of Newton-Raphson functions

	EquationCoefficient MemEquationCoefficient; // first three elements: diffusion, last four elements: migration
	EquationCoefficient SolEquationCoefficient; // first three elements: diffusion, last four elements: migration

	void CalculateF();
	void GeoCoefficientA(mesh& phase, Eigen::MatrixXd& GeoCoeffA) const;
	void GeoCoefficientB(mesh& phase, Eigen::MatrixXd& GeoCoeffB) const;
	void initialiseX();

	mesh& membrane;
	mesh& solution;
	PotentialSignal& Signal;
	const nernst_equation& Thermo;
	const IonSystem& membraneIons;
	const IonSystem& solutionIons;
	const unsigned long MatrixLen;
};

class EquationCoefficient
{
	friend class solver;
public:
	EquationCoefficient(const IonSystem& fIons, const mesh& phase, const PotentialSignal& fSignal, const nernst_equation& fThermo);
	void CalculateCoeff(Eigen::MatrixXd& GeoCoeffA, Eigen::MatrixXd GeoCoeffB);

private:
	Eigen::MatrixXd CoeffProductA;
	Eigen::MatrixXd CoeffProductB;
	Eigen::MatrixXd CoeffReactantA;
	Eigen::MatrixXd CoeffReactantB;
	Eigen::MatrixXd CoeffAnionA;
	Eigen::MatrixXd CoeffAnionB;
	Eigen::MatrixXd CoeffCationA;
	Eigen::MatrixXd CoeffCationB;

	Eigen::MatrixXd CoeffPotentialA;
	Eigen::MatrixXd CoeffPotentialB;

	const IonSystem& Ions;
	const PotentialSignal& Signal;
	const double F_R_T;
};