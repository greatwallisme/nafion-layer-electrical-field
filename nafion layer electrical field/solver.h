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

	void SolutionPotDerivativeInit( vector<Tt>& MatrixAlist, unsigned long i, unsigned long j, unsigned long rea_j_i, unsigned long pro_j_i, unsigned long ani_j_i, unsigned long cat_j_i,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I, Boundary boundary) const;

	void MembranePotDerivativeInit(vector<Tt>& MatrixAlist, unsigned long i, unsigned long j, unsigned long rea_j_i, unsigned long pro_j_i, unsigned long cat_j_i,
		unsigned long pot_j_i, unsigned long pot_jp1_i, unsigned long pot_jm1_i, unsigned long pot_j_ip1, unsigned long pot_j_im1,
		const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I, Boundary boundary) const;

	void initialiseMatrixA();

	void UpdateMatrixA();
	void UpdateMemDerivative(TwoDIndex::Species species, unsigned long i, unsigned long j, InnerIterator& it);
	void UpdateSolDerivative(TwoDIndex::Species species, unsigned long i, unsigned long j, InnerIterator& it);

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
};