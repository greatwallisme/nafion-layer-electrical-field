#pragma once
#include "mesh.h"
#include "IonSystem.h"
#include "potential_signal.h"
#include <Eigen/Sparse>
#include "thermodynamics.h"
#include "EquationCoefficient.h"
#include "SolverIndex.h"

typedef Eigen::SparseMatrix<double> SpMatrixXd;
typedef Eigen::SparseVector<double> SpVectorXd;
typedef SpMatrixXd::InnerIterator InnerIterator;
typedef Eigen::Triplet<double> Tt;

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
	void solve();

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

	double BulkMTEquation(unsigned long i, unsigned long j,
		double Xj_i, double Xjp1_i, double Xjm1_i, double Xj_ip1, double Xj_im1,
		double Xpot_j_i, double Xpot_jp1_i, double Xpot_jm1_i, double Xpot_j_ip1, double Xpot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn);

	//Derivatives
	void MembraneMTDerivative(unsigned long i, unsigned long j, unsigned long j_i, unsigned long jp1_i, unsigned long jm1_i, unsigned long j_ip1, unsigned long j_im1,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn, BoundaryEnum::Boundary boundary, SpeciesEnum::Species species, void (*Assign)(Tt));

	void SolutionMTDerivative(unsigned long i, unsigned long j, unsigned long j_i, unsigned long jp1_i, unsigned long jm1_i, unsigned long j_ip1, unsigned long j_im1,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn, BoundaryEnum::Boundary boundary, SpeciesEnum::Species species, void(*Assign)(Tt));

	double BulkPotEquation(unsigned long i, unsigned long j,
		double Xrea_j_i, double Xpro_j_i, double Xani_j_i, double Xcat_j_i,
		double Xpot_j_i, double Xpot_jp1_i, double Xpot_jm1_i, double Xpot_j_ip1, double Xpot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I);

	void SolutionPotDerivative(unsigned long i, unsigned long j, unsigned long rea_j_i, unsigned long pro_j_i, unsigned long ani_j_i, unsigned long cat_j_i,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I, BoundaryEnum::Boundary boundary, void(*Assign)(Tt));

	void MembranePotDerivative(unsigned long i, unsigned long j, unsigned long rea_j_i, unsigned long pro_j_i, unsigned long cat_j_i,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I, BoundaryEnum::Boundary boundary, void(*Assign)(Tt));

	void initialiseMatrixA(void(*Assign)(Tt));

	void UpdateMatrixA();

	void LockedPushBack(Tt triplet);
	void LockedIndexAssign(Tt triplet);

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

	const OneDIndex& Index1d;
	const TwoDIndex& Index2d;
	vector<Tt> MatrixAlist;

	omp_lock_t writeLock;
	unsigned long MatrixAAssignIndex;

	
};

