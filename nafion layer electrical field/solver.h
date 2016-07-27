#pragma once
#include "mesh.h"
#include "IonSystem.h"
#include "potential_signal.h"
#include <Eigen/Sparse>
#include "thermodynamics.h"

typedef Eigen::SparseMatrix<double> SpMatrixXd;
typedef Eigen::SparseVector<double> SpVectorXd;
typedef Eigen::Triplet<double> Td;

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

class solver
{
public:
	solver(mesh& fmembrane, mesh& fsolution,
		const IonSystem& fmembraneIons, const IonSystem& fsolutionIons,
		PotentialSignal& fSignal,
		const nernst_equation& fThermo,
		const ElectrodeReaction& fElecR,
		const InterfaceReaction& fCationTransR,
		const InterfaceReaction& fProductTransR,
		const InterfaceReaction& fReactantTransR);
	void initialise();

private:
	SpMatrixXd MatrixA; // Ax = b for Fc, each element inside is initialised as 0
	Eigen::VectorXd arrayb; // Ax = b for Fc; each element inside is initialised as 0
	Eigen::VectorXd dX; // the change of x in an iteration
	Eigen::VectorXd X; // the calculated results;
	Eigen::VectorXd F; // values of Newton-Raphson functions

	EquationCoefficient MemEquationCoefficient; // first three elements: diffusion, last four elements: migration
	EquationCoefficient SolEquationCoefficient; // first three elements: diffusion, last four elements: migration

	enum Boundary { bulk, bottom, top, left, right, right_bottom, left_bottom_corner, right_bottom_corner, left_upper_corner, right_upper_corner }; // add enum struct for boundary
	enum Species {Reactant, Product, Anion, Cation, Potential};

	void CalculateF();
	void GeoCoefficientA(mesh& phase, Eigen::MatrixXd& GeoCoeffA) const;
	void GeoCoefficientB(mesh& phase, Eigen::MatrixXd& GeoCoeffB) const;
	void initialiseX();

	double BulkMTEquation(unsigned long i, unsigned long j,
		double Xj_i, double Xjp1_i, double Xjm1_i, double Xj_ip1, double Xj_im1,
		double Xpot_j_i, double Xpot_jp1_i, double Xpot_jm1_i, double Xpot_j_ip1, double Xpot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn);

	//Derivatives
	void MembraneMTDerivativeInit(vector<Tt>& MatrixAlist, unsigned long i, unsigned long j, unsigned long j_i, unsigned long jp1_i, unsigned long jm1_i, unsigned long j_ip1, unsigned long j_im1,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn, Boundary boundary, Species species) const;

	void SolutionMTDerivativeInit(vector<Tt>& MatrixAlist, unsigned long i, unsigned long j, unsigned long j_i, unsigned long jp1_i, unsigned long jm1_i, unsigned long j_ip1, unsigned long j_im1,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn, Boundary boundary, Species species) const;

	double BulkPotEquation(unsigned long i, unsigned long j,
		double Xrea_j_i, double Xpro_j_i, double Xani_j_i, double Xcat_j_i,
		double Xpot_j_i, double Xpot_jp1_i, double Xpot_jm1_i, double Xpot_j_ip1, double Xpot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I);

	void initialiseMatrixA();

	mesh& membrane;
	mesh& solution;
	PotentialSignal& Signal;
	const nernst_equation& Thermo;
	const ElectrodeReaction& ElecR;
	const IonSystem& membraneIons;
	const IonSystem& solutionIons;
	const unsigned long MatrixLen;
	const InterfaceReaction& CationTransR;
	const InterfaceReaction& ProductTransR;
	const InterfaceReaction& ReactantTransR;
};