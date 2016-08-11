#include "solver.h"
#include <vector>
#include <iostream>
#include "Errors.h"
#include <Eigen/SparseCholesky>

solver::solver(mesh& fmembrane, mesh& fsolution, IonSystem& fmembraneIons, IonSystem& fsolutionIons, 
	PotentialSignal& fSignal, const nernst_equation& fThermo, const ElectrodeReaction& fElecR, 
	const InterfaceReaction& fCationTransR, const InterfaceReaction& fProductTransR, const InterfaceReaction& fReactantTransR) :
	membrane(fmembrane), solution(fsolution), membraneIons(fmembraneIons), solutionIons(fsolutionIons), Signal(fSignal), Thermo(fThermo), 
	ElecR(fElecR), CationTransR(fCationTransR), ProductTransR(fProductTransR), ReactantTransR(fReactantTransR),
	MatrixLen(fmembrane.m*fmembrane.n*4 + fsolution.m*fsolution.n*5),
	MatrixA(MatrixLen, MatrixLen),
	arrayb(MatrixLen),
	dX(MatrixLen),
	X(MatrixLen),
	F(MatrixLen),
	MemEquationCoefficient(fmembraneIons, fmembrane, fSignal, fThermo),
	SolEquationCoefficient(fsolutionIons, fsolution, fSignal, fThermo),
	Index1d(OneDIndex(fmembrane, fsolution)),
	Index2d(TwoDIndex(fmembrane, fsolution))
{
	MatrixAlist.reserve((membrane.Getmxn() * 4 + solution.Getmxn() * 5) * 10);
	MatrixAAssignIndex = 0;
}

void solver::initialise()
{
	/*
	A0 = -1/(dZ1*dZ2)
	A1 = 1/[dZ1(dZ1 + dZ2)], A2 = 1/[dZ2(dZ1 + dZ2)];
	A1_ = 1/[(dZ1 + dZ2)^2], A2_ = 1/[dZ2(dZ1 + dZ2)], A3_ = 1/[dZ1(dZ1 + dZ2)]

	B0 = -1/(dR1*dR2)
	B1 = 1/[dR1(dR1 + dR2)] - 1/R*1/(dR1 + dR2), B2 = 1/[dR2(dR1 + dR2)] + 1/R*1/(dR1 + dR2)
	B1_ = 1/[(dR1 + dR2)^2], B2_ = 1/[dR2(dR1 + dR2)] + 1/[R(dR1 + dR2)], B3_ = 1/[dR1(dR1 + dR2)] - 1/[R(dR1 + dR2)]
	*/

	Eigen::MatrixXd GeoCoeffMemA(7, membrane.n);
	Eigen::MatrixXd GeoCoeffSolA(7, solution.n);
	Eigen::MatrixXd GeoCoeffMemB(7, membrane.m);
	Eigen::MatrixXd GeoCoeffSolB(7, solution.m);
	// membrane
	GeoCoefficientA(membrane, GeoCoeffMemA);
	GeoCoefficientB(membrane, GeoCoeffMemB);
	// solution
	GeoCoefficientA(solution, GeoCoeffSolA);
	GeoCoefficientB(solution, GeoCoeffSolB);

	// initialise equation coefficients
	MemEquationCoefficient.CalculateCoeff(GeoCoeffMemA, GeoCoeffMemB);
	SolEquationCoefficient.CalculateCoeff(GeoCoeffSolA, GeoCoeffSolB);

	// initialise variable vector X
	initialiseX();

	// initialise MatrixA
	initialiseMatrixA(&solver::LockedPushBack);
	MatrixAlist.shrink_to_fit();

}

void solver::solve()
{
	/*
	Newto-Raphson method.
	1. F(Xn)
	2. F'(Xn)dX = -F(Xn)
	3. Xn+1 = Xn + dX
	*/

	do {
		CalculateF();
		Eigen::SimplicialLLT<SpMatrixXd> dXSolver;
		dXSolver.compute(MatrixA);
		if (dXSolver.info() != Eigen::Success) {
			// decomposition failed
			cout << "Decomposition of MatrixA Failed";
			throw(DecompositionFailed());
		}
		dX = dXSolver.solve(-F);
		X = X + dX;
		UpdateMatrixA();
	} while ((dX.array() / X.array()).maxCoeff() > 0.001);

	SaveDensity();
}

void solver::GeoCoefficientA(mesh& phase, Eigen::MatrixXd& GeoCoeffA) const
{
	double dZ1, dZ2;
	for (long i = 0; i < phase.n; ++i) {
		if (i == 0) {
			dZ1 = 0.5*(phase.dz0 + phase.dz);
			dZ2 = 0.5*(phase.dz0 + phase.dz);
		}
		else if (i == 1) {
			dZ1 = 0.5*(phase.dz0 + phase.dz);
			dZ2 = phase.dz;
		}
		else {
			dZ1 = phase.dz;
			dZ2 = phase.dz;
		}

		GeoCoeffA(0, i) = -1 / (dZ1*dZ2);
		GeoCoeffA(1, i) = 1 / (dZ1*(dZ1 + dZ2));
		GeoCoeffA(2, i) = 1 / (dZ2*(dZ1 + dZ2));
		GeoCoeffA(3, i) = -1 / (dZ1*dZ2);
		GeoCoeffA(4, i) = 1 / ((dZ1 + dZ2)*(dZ1 + dZ2));
		GeoCoeffA(5, i) = 1 / (dZ2*(dZ1 + dZ2));
		GeoCoeffA(6, i) = i, 1 / (dZ1*(dZ1 + dZ2));
	}
}

void solver::GeoCoefficientB(mesh& phase, Eigen::MatrixXd& GeoCoeffB) const
{
	double dR1, dR2, R;
	for (long i = 0; i < phase.m; ++i) {
		R = phase.RR(0, i);
		if (i == 0) {
			dR1 = 0.5*(phase.dr0 + phase.dr);
			dR2 = 0.5*(phase.dr0 + phase.dr);
		}
		else if (i == 1) {
			dR1 = 0.5*(phase.dr0 + phase.dr);
			dR2 = phase.dr;
		}
		else{
			dR1 = phase.dr;
			dR2 = phase.dr;
		}

		GeoCoeffB(0, i) = -1 / (dR1*dR2);
		GeoCoeffB(1, i) = 1 / (dR1*(dR1 + dR2)) - 1 / R * 1 / (dR1 + dR2);
		GeoCoeffB(2, i) = 1 / (dR2*(dR1 + dR2)) + 1 / R * 1 / (dR1 + dR2);
		GeoCoeffB(3, i) = -1 / (dR1*dR2);
		GeoCoeffB(4, i) = 1 / ((dR1 + dR2) * (dR1 + dR2));
		GeoCoeffB(5, i) = 1 / (dR2*(dR1 + dR2)) + 1 / (R*(dR1 + dR2));
		GeoCoeffB(6, i) = 1 / (dR1*(dR1 + dR2)) - 1 / (R*(dR1 + dR2));
	}
}

void solver::initialiseX()
{
#pragma omp parallel for
	for (long i = 0; i < membrane.m; ++i) {
		for (long j = 0; j < membrane.n; ++j) {

			//Reactant
			X(Index1d(SpeciesEnum::mReactant, j, i)) = membraneIons.Reactant.DensityN(j, i);
			//Product
			X(Index1d(SpeciesEnum::mProduct, j, i)) = membraneIons.Product.DensityN(j, i);
			//Cation
			X(Index1d(SpeciesEnum::mCation, j, i)) = membraneIons.SupportCation.DensityN(j, i);
			//Potential
			X(Index1d(SpeciesEnum::mPotential, j, i)) = membraneIons.Potential.DensityN(j, i);
		}
	}

#pragma omp parallel for
	for (long i = 0; i < solution.m; ++i) {
		for (long j = 0; j < solution.n; ++j) {
			//Reactant
			X(Index1d(SpeciesEnum::sReactant, j, i)) = solutionIons.Reactant.DensityN(j, i);
			//Product
			X(Index1d(SpeciesEnum::sProduct, j, i)) = solutionIons.Product.DensityN(j, i);
			//Anion
			X(Index1d(SpeciesEnum::sAnion, j, i)) = solutionIons.SupportAnion.DensityN(j, i);
			//Cation
			X(Index1d(SpeciesEnum::sCation, j, i)) = solutionIons.SupportCation.DensityN(j, i);
			//Potential
			X(Index1d(SpeciesEnum::sPotential, j, i)) = solutionIons.Potential.DensityN(j, i);
		}
	}
}

void solver::CalculateF()
{
	// define reference
	auto& M = MemEquationCoefficient;
	auto& S = SolEquationCoefficient;
	auto& MI = membraneIons;
	auto& SI = solutionIons;
	//Reactant index
	long rea_j_i(0UL), rea_jm1_i(0UL), rea_jp1_i(0UL), rea_j_im1(0UL), rea_j_ip1(0UL);
	//Product index
	long pro_j_i(0UL), pro_jm1_i(0UL), pro_jp1_i(0UL), pro_j_im1(0UL), pro_j_ip1(0UL);
	//Anion index
	long ani_j_i(0UL), ani_jm1_i(0UL), ani_jp1_i(0UL), ani_j_im1(0UL), ani_j_ip1(0UL);
	//Cation index
	long cat_j_i(0UL), cat_jm1_i(0UL), cat_jp1_i(0UL), cat_j_im1(0UL), cat_j_ip1(0UL);
	//Potential index
	long pot_j_i(0UL), pot_jm1_i(0UL), pot_jp1_i(0UL), pot_j_im1(0UL), pot_j_ip1(0UL);

	long mxn = membrane.Getmxn();

	// Calculate membrane
#pragma omp parallel for private(rea_j_i, rea_jm1_i, rea_jp1_i, rea_j_im1, rea_j_ip1, pro_j_i, pro_jm1_i, pro_jp1_i, pro_j_im1, pro_j_ip1, cat_j_i, cat_jm1_i, cat_jp1_i, cat_j_im1, cat_j_ip1, pot_j_i, pot_jm1_i, pot_jp1_i, pot_j_im1, pot_j_ip1)
	for (long i = 0; i < membrane.m - 1; ++i) {
		for (long j = 0; j < membrane.n - 1; ++j) {
			// Reactant index
			rea_j_i = membrane.n*i + j;
			rea_jm1_i = rea_j_i - 1;
			rea_jp1_i = rea_j_i + 1;
			rea_j_im1 = rea_j_i - membrane.n;
			rea_j_ip1 = rea_j_i + membrane.n;
			//Product index
			pro_j_i = rea_j_i + mxn;
			pro_jm1_i = pro_j_i - 1;
			pro_jp1_i = pro_j_i + 1;
			pro_j_im1 = pro_j_i - membrane.n;
			pro_j_ip1 = pro_j_i + membrane.n;
			//Cation index
			cat_j_i = ani_j_i + mxn;
			cat_jm1_i = cat_j_i - 1;;
			cat_jp1_i = cat_j_i + 1;
			cat_j_im1 = cat_j_i - membrane.n;
			cat_j_ip1 = cat_j_i + membrane.n;
			//Potential index
			pot_j_i = cat_j_i + mxn;
			pot_jm1_i = pot_j_i - 1;
			pot_jp1_i = pot_j_i + 1;
			pot_j_im1 = pot_j_i - membrane.n;
			pot_j_ip1 = pot_j_i + membrane.n;

			if (i > 1 && j > 1 && i < membrane.m - 1 && i < membrane.n - 1) {
				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_jm1_i), X(rea_j_ip1), X(rea_j_im1), 
							   X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_jm1_i), X(pro_j_ip1), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_jm1_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffPotentialA, M.CoeffPotentialB, MI);

			}
			else if (j == 0 && i > 1 && i < membrane.m - 1) {
				
				double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
				double kf = ElecR.kf(DrivingPotential);
				double kb = ElecR.kb(DrivingPotential);
				double reactionRate = kf*X(pro_j_i) - kb*X(rea_j_i); // production is O, reatant is R

				// Reactant:
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_j_i), X(rea_j_ip1), X(rea_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				F(rea_j_i) += reactionRate / membrane.dz*Signal.dt;
				// Product:
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_j_i), X(pro_j_ip1), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				F(pro_j_i) -= reactionRate / membrane.dz*Signal.dt;
				// Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_j_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				// Potential
				F(pot_j_i) = Signal.AppliedPotential() - X(pot_j_i) - ElecR.E_formal - DrivingPotential;
			}
			else if (i == 0 && j > 0 && j < membrane.n - 1) {
				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_jm1_i), X(rea_j_ip1), X(rea_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_jm1_i), X(pro_j_ip1), X(pro_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_jm1_i), X(cat_j_ip1), X(cat_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffPotentialA, M.CoeffPotentialB, MI);
			}
			else if (j == membrane.n - 1 && i > 0 && i < membrane.m - 1) {
				// index in the solution phase
				// Reactant index
				long srea_jp1_i = solution.n*i + 4 * membrane.Getmxn();
				//Product index
				long spro_jp1_i = srea_jp1_i + mxn;
				//Cation index
				long scat_jp1_i = spro_jp1_i + 2*mxn;
				//Potential index
				long spot_jp1_i = scat_jp1_i + mxn;

				double dE = X(spot_jp1_i) - X(pot_j_i);
				//Cation transfer rate
				double kf = CationTransR.kf(dE);
				double kb = CationTransR.kb(dE);
				double CationTransRate = kf*X(scat_jp1_i) - kb*X(cat_j_i);
				//Product transfer rate
				kf = ProductTransR.kf(dE);
				kb = ProductTransR.kb(dE);
				double ProductTransRate = kf*X(spro_jp1_i) - kb*X(pro_j_i);
				//Reactant transfer rate
				kf = ReactantTransR.kf(0);
				kb = ReactantTransR.kb(0);
				double ReactantTransRate = kf*X(srea_jp1_i) - kb*X(rea_j_i);

				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_j_i), X(rea_jm1_i), X(rea_j_ip1), X(rea_j_im1),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				F(rea_j_i) += ReactantTransRate / membrane.dz*Signal.dt;
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_j_i), X(pro_jm1_i), X(pro_j_ip1), X(pro_j_im1),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				F(pro_j_i) += ProductTransRate / membrane.dz*Signal.dt;
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_j_i), X(cat_jm1_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				F(cat_j_i) += CationTransRate/membrane.dz*Signal.dt;
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), 
					X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffPotentialA, M.CoeffPotentialB, MI);


			}
			else if (i == membrane.m - 1 && j > 0 && j < membrane.n - 1) {
				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_jm1_i), X(rea_j_i), X(rea_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_jm1_i), X(pro_j_i), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_jm1_i), X(cat_j_i), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffPotentialA, M.CoeffPotentialB, MI);
			}
			else if (i == 0 && j == 0) {
				
				double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
				double kf = ElecR.kf(DrivingPotential);
				double kb = ElecR.kb(DrivingPotential);
				double reactionRate = kf*X(pro_j_i) - kb*X(rea_j_i); // production is O, reatant is R
				// Reactant:
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_j_i), X(rea_j_ip1), X(rea_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);

				F(rea_j_i) += reactionRate / membrane.dz*Signal.dt;
				// Product:
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_j_i), X(pro_j_ip1), X(pro_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);

				F(pro_j_i) -= reactionRate / membrane.dz*Signal.dt;
				// Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_j_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				// Potential
				F(pot_j_i) = Signal.AppliedPotential() - X(pot_j_i) - ElecR.E_formal - DrivingPotential;
			}
			else if (i == membrane.m - 1 && j == 0) {
				// Reactant:
				double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
				double kf = ElecR.kf(DrivingPotential);
				double kb = ElecR.kb(DrivingPotential);
				double reactionRate = kf*X(pro_j_i) - kb*X(rea_j_i); // production is O, reatant is R

				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_j_i), X(rea_j_i), X(rea_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_i), X(pot_j_im1), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				F(rea_j_i) += reactionRate / membrane.dz*Signal.dt;
				// Product:
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_j_i), X(pro_j_i), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_i), X(pot_j_im1), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				F(pro_j_i) -= reactionRate / membrane.dz*Signal.dt;
				// Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_j_i), X(cat_j_i), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_i), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				// Potential
				F(pot_j_i) = Signal.AppliedPotential() - X(pot_j_i) - ElecR.E_formal - DrivingPotential;
			}
			else if (j == membrane.n - 1 && i == 0) {
				long srea_jp1_i = solution.n*i + 4 * membrane.Getmxn();
				//Product index
				long spro_jp1_i = srea_jp1_i + mxn;
				//Cation index
				long scat_jp1_i = spro_jp1_i + 2*mxn;
				//Potential index
				long spot_jp1_i = scat_jp1_i + mxn;

				double dE = X(spot_jp1_i) - X(pot_j_i);
				//Cation transfer rate
				double kf = CationTransR.kf(dE);
				double kb = CationTransR.kb(dE);
				double CationTransRate = kf*X(scat_jp1_i) - kb*X(cat_j_i);
				//Product transfer rate
				kf = ProductTransR.kf(dE);
				kb = ProductTransR.kb(dE);
				double ProductTransRate = kf*X(spro_jp1_i) - kb*X(pro_j_i);
				//Reactant transfer rate
				kf = ReactantTransR.kf(0);
				kb = ReactantTransR.kb(0);
				double ReactantTransRate = kf*X(srea_jp1_i) - kb*X(rea_j_i);

				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_j_i), X(rea_jm1_i), X(rea_j_ip1), X(rea_j_i),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				F(rea_j_i) += ReactantTransRate / membrane.dz*Signal.dt;
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_j_i), X(pro_jm1_i), X(pro_j_ip1), X(pro_j_i),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				F(pro_j_i) += ProductTransRate / membrane.dz*Signal.dt;
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_j_i), X(cat_jm1_i), X(cat_j_ip1), X(cat_j_i),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				F(cat_j_i) += CationTransRate / membrane.dz*Signal.dt;
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i),
					X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), M.CoeffPotentialA, M.CoeffPotentialB, MI);
			}
			else if (j == membrane.n - 1 && i == membrane.m - 1) {
				long srea_jp1_i = solution.n*i + 4 * membrane.Getmxn();
				//Product index
				long spro_jp1_i = srea_jp1_i + mxn;
				//Cation index
				long scat_jp1_i = spro_jp1_i + mxn;
				//Potential index
				long spot_jp1_i = scat_jp1_i + mxn;

				double dE = X(spot_jp1_i) - X(pot_j_i);
				//Cation transfer rate
				double kf = CationTransR.kf(dE);
				double kb = CationTransR.kb(dE);
				double CationTransRate = kf*X(scat_jp1_i) - kb*X(cat_j_i);
				//Product transfer rate
				kf = ProductTransR.kf(dE);
				kb = ProductTransR.kb(dE);
				double ProductTransRate = kf*X(spro_jp1_i) - kb*X(pro_j_i);
				//Reactant transfer rate
				kf = ReactantTransR.kf(0);
				kb = ReactantTransR.kb(0);
				double ReactantTransRate = kf*X(srea_jp1_i) - kb*X(rea_j_i);

				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_j_i), X(rea_jm1_i), X(rea_j_i), X(rea_j_im1),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffReactantA, M.CoeffReactantB, membraneIons.Reactant.DensityN);
				F(rea_j_i) += ReactantTransRate / membrane.dz*Signal.dt;
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_j_i), X(pro_jm1_i), X(pro_j_i), X(pro_j_im1),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffProductA, M.CoeffProductB, membraneIons.Product.DensityN);
				F(pro_j_i) += ProductTransRate / membrane.dz*Signal.dt;
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_j_i), X(cat_jm1_i), X(cat_j_i), X(cat_j_im1),
					X(pot_j_i), X(pot_j_i), X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffCationA, M.CoeffCationB, membraneIons.SupportCation.DensityN);
				F(cat_j_i) += CationTransRate / membrane.dz*Signal.dt;
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i),
					X(pot_jm1_i), X(pot_j_i), X(pot_j_im1), M.CoeffPotentialA, M.CoeffPotentialB, MI);
			}
		}
	}

	mxn = solution.Getmxn();
	// Solution
#pragma omp parallel for private(rea_j_i, rea_jm1_i, rea_jp1_i, rea_j_im1, rea_j_ip1, pro_j_i, pro_jm1_i, pro_jp1_i, pro_j_im1, pro_j_ip1, ani_j_i, ani_jm1_i, ani_jp1_i, ani_j_im1, ani_j_ip1,cat_j_i, cat_jm1_i, cat_jp1_i, cat_j_im1, cat_j_ip1, pot_j_i, pot_jm1_i, pot_jp1_i, pot_j_im1, pot_j_ip1)
	for (long i = 0; i < solution.m - 1; ++i) {
		for (long j = 0; j < solution.n - 1; ++j) {

			// Reactant index
			rea_j_i = solution.n*i + j + 4*membrane.Getmxn();
			rea_jm1_i = rea_j_i - 1;
			rea_jp1_i = rea_j_i + 1;
			rea_j_im1 = rea_j_i - solution.n;
			rea_j_ip1 = rea_j_i + solution.n;
			//Product index
			pro_j_i = rea_j_i + mxn;
			pro_jm1_i = pro_j_i - 1;
			pro_jp1_i = pro_j_i + 1;
			pro_j_im1 = pro_j_i - solution.n;
			pro_j_ip1 = pro_j_i + solution.n;
			//Anion index
			ani_j_i = pro_j_i + mxn;
			ani_jm1_i = ani_j_i - 1;
			ani_jp1_i = ani_j_i + 1;
			ani_j_im1 = ani_j_i - solution.n;
			ani_j_ip1 = ani_j_i + solution.n;
			//Cation index
			cat_j_i = ani_j_i + mxn;
			cat_jm1_i = cat_j_i - 1;;
			cat_jp1_i = cat_j_i + 1;
			cat_j_im1 = cat_j_i - solution.n;
			cat_j_ip1 = cat_j_i + solution.n;
			//Potential index
			pot_j_i = cat_j_i + mxn;
			pot_jm1_i = pot_j_i - 1;
			pot_jp1_i = pot_j_i + 1;
			pot_j_im1 = pot_j_i - solution.n;
			pot_j_ip1 = pot_j_i + solution.n;

			if (i > 0 && i < solution.m - 1 && j > 0 && j < solution.n - 1) {
				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_jm1_i), X(rea_j_ip1), X(rea_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffReactantA, S.CoeffReactantB, SI.Reactant.DensityN);
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_jm1_i), X(pro_j_ip1), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffProductA, S.CoeffProductB, SI.Product.DensityN);
				//Anion
				F(ani_j_i) = BulkMTEquation(i, j, X(ani_j_i), X(ani_jp1_i), X(ani_jm1_i), X(ani_j_ip1), X(ani_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffAnionA, S.CoeffAnionB, SI.SupportAnion.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_jm1_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffCationA, S.CoeffCationB, SI.SupportCation.DensityN);
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffPotentialA, S.CoeffPotentialB, SI);
			}
			else if (i > 0 && i < solution.m - 1 && j == solution.n - 1) {
				//Reactant
				F(rea_j_i) = X(rea_j_i) - solutionIons.Reactant.Cinitial;
				//Product
				F(pro_j_i) = X(pro_j_i) - solutionIons.Product.Cinitial;
				//Anion
				F(ani_j_i) = X(ani_j_i) - solutionIons.SupportAnion.Cinitial;
				//Cation
				F(cat_j_i) = X(cat_j_i) - solutionIons.SupportCation.Cinitial;
				//Potential
				F(pot_j_i) = X(pot_j_i);
			}
			else if (i == 0 && j > 0 && j < solution.n - 1) {
				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_jm1_i), X(rea_j_ip1), X(rea_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), S.CoeffReactantA, S.CoeffReactantB, SI.Reactant.DensityN);
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_jm1_i), X(pro_j_ip1), X(pro_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), S.CoeffProductA, S.CoeffProductB, SI.Product.DensityN);
				//Anion
				F(ani_j_i) = BulkMTEquation(i, j, X(ani_j_i), X(ani_jp1_i), X(ani_jm1_i), X(ani_j_ip1), X(ani_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), S.CoeffAnionA, S.CoeffAnionB, SI.SupportAnion.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_jm1_i), X(cat_j_ip1), X(cat_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), S.CoeffCationA, S.CoeffCationB, SI.SupportCation.DensityN);
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_jm1_i), X(pot_j_ip1), X(pot_j_i), S.CoeffPotentialA, S.CoeffPotentialB, SI);
			}
			else if (i == solution.m - 1 && j > 0 && j < solution.n - 1) {
				//Reactant
				F(rea_j_i) = X(rea_j_i) - solutionIons.Reactant.Cinitial;
				//Product
				F(pro_j_i) = X(pro_j_i) - solutionIons.Product.Cinitial;
				//Anion
				F(ani_j_i) = X(ani_j_i) - solutionIons.SupportAnion.Cinitial;
				//Cation
				F(cat_j_i) = X(cat_j_i) - solutionIons.SupportCation.Cinitial;
				//Potential
				F(pot_j_i) = X(pot_j_i);
			}
			else if (i == 0 && j == solution.n - 1) {
				//Reactant
				F(rea_j_i) = X(rea_j_i) - solutionIons.Reactant.Cinitial;
				//Product
				F(pro_j_i) = X(pro_j_i) - solutionIons.Product.Cinitial;
				//Anion
				F(ani_j_i) = X(ani_j_i) - solutionIons.SupportAnion.Cinitial;
				//Cation
				F(cat_j_i) = X(cat_j_i) - solutionIons.SupportCation.Cinitial;
				//Potential
				F(pot_j_i) = X(pot_j_i);
			}
			else if (i == solution.m - 1 && j == solution.n - 1) {
				//Reactant
				F(rea_j_i) = X(rea_j_i) - solutionIons.Reactant.Cinitial;
				//Product
				F(pro_j_i) = X(pro_j_i) - solutionIons.Product.Cinitial;
				//Anion
				F(ani_j_i) = X(ani_j_i) - solutionIons.SupportAnion.Cinitial;
				//Cation
				F(cat_j_i) = X(cat_j_i) - solutionIons.SupportCation.Cinitial;
				//Potential
				F(pot_j_i) = X(pot_j_i);
			}
			else if (i > 0 && i < membrane.m && j == 0) {

				// index in the membrane phase
				// Reactant index
				long mrea_jm1_i = membrane.n*i + membrane.n - 1;
				//Product index
				long mpro_jm1_i = mrea_jm1_i + membrane.Getmxn();
				//Cation index
				long mcat_jm1_i = mpro_jm1_i + membrane.Getmxn();
				//Potential index
				long mpot_jm1_i = mcat_jm1_i + membrane.Getmxn();

				double dE = X(pot_j_i) - X(mpot_jm1_i);
				//Cation transfer rate
				double kf = CationTransR.kf(dE);
				double kb = CationTransR.kb(dE);
				double CationTransRate = kf*X(cat_j_i) - kb*X(mcat_jm1_i);
				//Product transfer rate
				kf = ProductTransR.kf(dE);
				kb = ProductTransR.kb(dE);
				double ProductTransRate = kf*X(pro_j_i) - kb*X(mpro_jm1_i);
				//Reactant transfer rate
				kf = ReactantTransR.kf(0);
				kb = ReactantTransR.kb(0);
				double ReactantTransRate = kf*X(rea_j_i) - kb*X(mrea_jm1_i);

				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_j_i), X(rea_j_ip1), X(rea_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffReactantA, S.CoeffReactantB, SI.Reactant.DensityN);
				F(rea_j_i) -= ReactantTransRate / solution.dz0*Signal.dt;
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_j_i), X(pro_j_ip1), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffProductA, S.CoeffProductB, SI.Product.DensityN);
				F(pro_j_i) -= ProductTransRate / solution.dz0*Signal.dt;
				//Anion
				F(ani_j_i) = BulkMTEquation(i, j, X(ani_j_i), X(ani_jp1_i), X(ani_j_i), X(ani_j_ip1), X(ani_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffAnionA, S.CoeffAnionB, SI.SupportAnion.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_j_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffCationA, S.CoeffCationB, SI.SupportCation.DensityN);
				F(cat_j_i) -= CationTransRate / solution.dz0*Signal.dt;
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffPotentialA, S.CoeffPotentialB, SI);
			}
			else if (j == 0 && i > membrane.m - 1 && i < solution.m - 1) {
				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_j_i), X(rea_j_ip1), X(rea_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffReactantA, S.CoeffReactantB, SI.Reactant.DensityN);
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_j_i), X(pro_j_ip1), X(pro_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffProductA, S.CoeffProductB, SI.Product.DensityN);
				//Anion
				F(ani_j_i) = BulkMTEquation(i, j, X(ani_j_i), X(ani_jp1_i), X(ani_j_i), X(ani_j_ip1), X(ani_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffAnionA, S.CoeffAnionB, SI.SupportAnion.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_j_i), X(cat_j_ip1), X(cat_j_im1),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffCationA, S.CoeffCationB, SI.SupportCation.DensityN);
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_im1), S.CoeffPotentialA, S.CoeffPotentialB, SI);
			}
			else if (i == 0 && j == 0) {
				// index in the membrane phase
				// Reactant index
				long mrea_jm1_i = membrane.n*i + membrane.n - 1;
				//Product index
				long mpro_jm1_i = mrea_jm1_i + membrane.Getmxn();
				//Cation index
				long mcat_jm1_i = mpro_jm1_i + membrane.Getmxn();
				//Potential index
				long mpot_jm1_i = mcat_jm1_i + membrane.Getmxn();

				double dE = X(pot_j_i) - X(mpot_jm1_i);
				//Cation transfer rate
				double kf = CationTransR.kf(dE);
				double kb = CationTransR.kb(dE);
				double CationTransRate = kf*X(cat_j_i) - kb*X(mcat_jm1_i);
				//Product transfer rate
				kf = ProductTransR.kf(dE);
				kb = ProductTransR.kb(dE);
				double ProductTransRate = kf*X(pro_j_i) - kb*X(mpro_jm1_i);
				//Reactant transfer rate
				kf = ReactantTransR.kf(0);
				kb = ReactantTransR.kb(0);
				double ReactantTransRate = kf*X(rea_j_i) - kb*X(mrea_jm1_i);

				//Reactant
				F(rea_j_i) = BulkMTEquation(i, j, X(rea_j_i), X(rea_jp1_i), X(rea_j_i), X(rea_j_ip1), X(rea_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), S.CoeffReactantA, S.CoeffReactantB, SI.Reactant.DensityN);
				F(rea_j_i) -= ReactantTransRate / solution.dz0*Signal.dt;
				//Product
				F(pro_j_i) = BulkMTEquation(i, j, X(pro_j_i), X(pro_jp1_i), X(pro_j_i), X(pro_j_ip1), X(pro_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), S.CoeffProductA, S.CoeffProductB, SI.Product.DensityN);
				F(pro_j_i) -= ProductTransRate / solution.dz0*Signal.dt;
				//Anion
				F(ani_j_i) = BulkMTEquation(i, j, X(ani_j_i), X(ani_jp1_i), X(ani_j_i), X(ani_j_ip1), X(ani_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), S.CoeffAnionA, S.CoeffAnionB, SI.SupportAnion.DensityN);
				//Cation
				F(cat_j_i) = BulkMTEquation(i, j, X(cat_j_i), X(cat_jp1_i), X(cat_j_i), X(cat_j_ip1), X(cat_j_i),
					X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), S.CoeffCationA, S.CoeffCationB, SI.SupportCation.DensityN);
				F(cat_j_i) -= CationTransRate / solution.dz0*Signal.dt;
				//Potential
				F(pot_j_i) = BulkPotEquation(i, j, X(rea_j_i), X(pro_j_i), X(ani_j_i), X(cat_j_i), X(pot_j_i), X(pot_jp1_i), X(pot_j_i), X(pot_j_ip1), X(pot_j_i), S.CoeffPotentialA, S.CoeffPotentialB, SI);
			}
			else if (i == solution.m - 1 && j == 0) {
				//Reactant
				F(rea_j_i) = X(rea_j_i) - solutionIons.Reactant.Cinitial;
				//Product
				F(pro_j_i) = X(pro_j_i) - solutionIons.Product.Cinitial;
				//Anion
				F(ani_j_i) = X(ani_j_i) - solutionIons.SupportAnion.Cinitial;
				//Cation
				F(cat_j_i) = X(cat_j_i) - solutionIons.SupportCation.Cinitial;
				//Potential
				F(pot_j_i) = X(pot_j_i);
			}
			
			
		}
	}
}

void solver::initialiseMatrixA(void (solver::*Assign)(Tt))
{
	//Reactant index
	long rea_j_i(0UL), rea_jm1_i(0UL), rea_jp1_i(0UL), rea_j_im1(0UL), rea_j_ip1(0UL);
	//Product index
	long pro_j_i(0UL), pro_jm1_i(0UL), pro_jp1_i(0UL), pro_j_im1(0UL), pro_j_ip1(0UL);
	//Anion index
	long ani_j_i(0UL), ani_jm1_i(0UL), ani_jp1_i(0UL), ani_j_im1(0UL), ani_j_ip1(0UL);
	//Cation index
	long cat_j_i(0UL), cat_jm1_i(0UL), cat_jp1_i(0UL), cat_j_im1(0UL), cat_j_ip1(0UL);
	//Potential index
	long pot_j_i(0UL), pot_jm1_i(0UL), pot_jp1_i(0UL), pot_j_im1(0UL), pot_j_ip1(0UL);
	
	omp_init_lock(&writeLock);
	//Calculate membrane
	for (long i = 0; i < membrane.m - 1; ++i) {
		for (long j = 0; j < membrane.n - 1; ++j) {
			// Reactant index
			rea_j_i = Index1d(SpeciesEnum::mReactant, j, i);
			rea_jm1_i = Index1d(SpeciesEnum::mReactant, j - 1, i);
			rea_jp1_i = Index1d(SpeciesEnum::mReactant, j + 1, i);
			rea_j_im1 = Index1d(SpeciesEnum::mReactant, j, i - 1);
			rea_j_ip1 = Index1d(SpeciesEnum::mReactant, j, i + 1);
			//Product index
			pro_j_i = Index1d(SpeciesEnum::mProduct, j, i);
			pro_jm1_i = Index1d(SpeciesEnum::mProduct, j - 1, i);
			pro_jp1_i = Index1d(SpeciesEnum::mProduct, j + 1, i);
			pro_j_im1 = Index1d(SpeciesEnum::mProduct, j, i - 1);
			pro_j_ip1 = Index1d(SpeciesEnum::mProduct, j, i + 1);
			//Cation index
			cat_j_i = Index1d(SpeciesEnum::mCation, j, i); 
			cat_jm1_i = Index1d(SpeciesEnum::mCation, j - 1, i);
			cat_jp1_i = Index1d(SpeciesEnum::mCation, j + 1, i);
			cat_j_im1 = Index1d(SpeciesEnum::mCation, j, i - 1);
			cat_j_ip1 = Index1d(SpeciesEnum::mCation, j, i + 1);
			//Potential index
			pot_j_i = Index1d(SpeciesEnum::mPotential, j, i);
			pot_jm1_i = Index1d(SpeciesEnum::mPotential, j - 1, i);
			pot_jp1_i = Index1d(SpeciesEnum::mPotential, j + 1, i);
			pot_j_im1 = Index1d(SpeciesEnum::mPotential, j, i - 1);
			pot_j_ip1 = Index1d(SpeciesEnum::mPotential, j, i + 1);
		

			if (i != 0 && j != 0 && i != membrane.m - 1 && j != membrane.n - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_jm1_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::bulk, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_jm1_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::bulk, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_jm1_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::bulk, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::bulk, Assign);
			}
			else if (j == 0 && i != 0 && i != membrane.m - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::bottom, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::bottom, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::bottom, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::bottom, Assign);
			}
			else if (j == membrane.n - 1 && i != 0 && i != membrane.m - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_j_i, rea_jm1_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::top, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_j_i, pro_jm1_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::top, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
										MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::top, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffProductB, membraneIons, BoundaryEnum::top, Assign);
			}
			else if (i == 0 && j != 0 && j != membrane.n - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_jm1_i, rea_j_ip1, rea_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::left, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_jm1_i, pro_j_ip1, pro_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::left, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_jm1_i, cat_j_ip1, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::left, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::left, Assign);
			}
			else if (i == membrane.n - 1 && j != 0 && j != membrane.n - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_jm1_i, rea_j_i, rea_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::right, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_jm1_i, pro_j_i, pro_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::right, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_jm1_i, cat_j_i, cat_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::right, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::right, Assign);
			}
			else if (i == 0 && j == 0) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_ip1, rea_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_ip1, pro_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_ip1, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::left_bottom_corner, Assign);
			}
			else if (j == 0 && i == membrane.m - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_i, rea_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_i, pro_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_i, cat_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::right_bottom_corner, Assign);
			}
			else if (i == 0 && j == membrane.n - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_j_i, rea_jm1_i, rea_j_ip1, rea_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_j_i, pro_jm1_i, pro_j_ip1, pro_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_ip1, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::left_upper_corner, Assign);
			}
			else if (j == membrane.n - 1 && i == membrane.m - 1) {
				// Reactant
				MembraneMTDerivative(i, j, rea_j_i, rea_j_i, rea_jm1_i, rea_j_i, rea_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffReactantA, MemEquationCoefficient.CoeffReactantB, membraneIons.Reactant.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::mReactant, Assign);
				// Product
				MembraneMTDerivative(i, j, pro_j_i, pro_j_i, pro_jm1_i, pro_j_i, pro_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffProductA, MemEquationCoefficient.CoeffProductB, membraneIons.Product.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::mProduct, Assign);
				// Cation
				MembraneMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_i, cat_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffCationA, MemEquationCoefficient.CoeffCationB, membraneIons.SupportCation.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::mCation, Assign);
				// Potential
				MembranePotDerivative(i, j, rea_j_i, pro_j_i, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					MemEquationCoefficient.CoeffPotentialA, MemEquationCoefficient.CoeffPotentialB, membraneIons, BoundaryEnum::right_upper_corner, Assign);
			}

		}
	}

	//Calculate solution
	for (long i = 0; i < solution.m - 1; ++i) {
		for (long j = 0; j < solution.n - 1; ++j) {

			// Reactant index
			rea_j_i = Index1d(SpeciesEnum::sReactant, j, i);
			rea_jm1_i = Index1d(SpeciesEnum::sReactant, j - 1, i);
			rea_jp1_i = Index1d(SpeciesEnum::sReactant, j + 1, i);
			rea_j_im1 = Index1d(SpeciesEnum::sReactant, j, i - 1);
			rea_j_ip1 = Index1d(SpeciesEnum::sReactant, j, i + 1);
			//Product index
			pro_j_i = Index1d(SpeciesEnum::sProduct, j, i);
			pro_jm1_i = Index1d(SpeciesEnum::sProduct, j - 1, i);
			pro_jp1_i = Index1d(SpeciesEnum::sProduct, j + 1, i);
			pro_j_im1 = Index1d(SpeciesEnum::sProduct, j, i - 1);
			pro_j_ip1 = Index1d(SpeciesEnum::sProduct, j, i + 1);
			//Anion index
			ani_j_i = Index1d(SpeciesEnum::sAnion, j, i);
			ani_jm1_i = Index1d(SpeciesEnum::sAnion, j - 1, i);
			ani_jp1_i = Index1d(SpeciesEnum::sAnion, j + 1, i);
			ani_j_im1 = Index1d(SpeciesEnum::sAnion, j, i - 1);
			ani_j_ip1 = Index1d(SpeciesEnum::sAnion, j, i + 1);
			//Cation index
			cat_j_i = Index1d(SpeciesEnum::sCation, j, i);
			cat_jm1_i = Index1d(SpeciesEnum::sCation, j - 1, i);
			cat_jp1_i = Index1d(SpeciesEnum::sCation, j + 1, i);
			cat_j_im1 = Index1d(SpeciesEnum::sCation, j, i - 1);
			cat_j_ip1 = Index1d(SpeciesEnum::sCation, j, i + 1);
			//Potential index
			pot_j_i = Index1d(SpeciesEnum::sPotential, j, i);
			pot_jm1_i = Index1d(SpeciesEnum::sPotential, j - 1, i);
			pot_jp1_i = Index1d(SpeciesEnum::sPotential, j + 1, i);
			pot_j_im1 = Index1d(SpeciesEnum::sPotential, j, i - 1);
			pot_j_ip1 = Index1d(SpeciesEnum::sPotential, j, i + 1);

			if (i > 0 && i < solution.m - 1 && j > 0 && j < solution.n - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_jm1_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::bulk, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_jm1_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::bulk, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_jm1_i, ani_j_ip1, ani_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::bulk, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_jm1_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::bulk, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::bulk, Assign);
			}
			else if (i > 0 && i < solution.m - 1 && j == solution.n - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_j_i, rea_jm1_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::top, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_j_i, pro_jm1_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::top, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_j_i, ani_jm1_i, ani_j_ip1, ani_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::top, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::top, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::top, Assign);
			}
			else if (i == 0 && j > 0 && j < solution.n - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_jm1_i, rea_j_ip1, rea_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::left, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_jm1_i, pro_j_ip1, pro_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::left, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_jm1_i, ani_j_ip1, ani_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::left, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_ip1, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::left, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::left, Assign);
			}
			else if (i == solution.m - 1 && j > 0 && j < solution.n - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_jm1_i, rea_j_i, rea_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::right, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_jm1_i, pro_j_i, pro_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::right, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_jm1_i, ani_j_i, ani_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::right, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_jm1_i, cat_j_i, cat_j_im1, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::right, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::right, Assign);
			}
			else if (i == 0 && j == solution.n - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_j_i, rea_jm1_i, rea_j_ip1, rea_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_j_i, pro_jm1_i, pro_j_ip1, pro_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_j_i, ani_jm1_i, ani_j_ip1, ani_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_ip1, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::left_upper_corner, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::left_upper_corner, Assign);
			}
			else if (i = solution.m - 1 && j == solution.n - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_j_i, rea_jm1_i, rea_j_i, rea_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_j_i, pro_jm1_i, pro_j_i, pro_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_j_i, ani_jm1_i, ani_j_i, ani_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_j_i, cat_jm1_i, cat_j_i, cat_j_im1, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::right_upper_corner, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_j_i, pot_jm1_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::right_upper_corner, Assign);
			}
			else if (i > 0 && i < membrane.m && j == 0) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::bottom, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::bottom, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_j_i, ani_j_ip1, ani_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::bottom, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::bottom, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::bottom, Assign);
			}
			else if (j == 0 && i > membrane.m - 1 && i < solution.m - 1) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_ip1, rea_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::right_bottom, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_ip1, pro_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::right_bottom, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_j_i, ani_j_ip1, ani_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::right_bottom, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_ip1, cat_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::right_bottom, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::right_bottom, Assign);
			}
			else if (i == 0 && j == 0) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_ip1, rea_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_ip1, pro_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_j_i, ani_j_ip1, ani_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_ip1, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::left_bottom_corner, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_ip1, pot_j_i,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::left_bottom_corner, Assign);
			}
			else if (i == solution.m - 1 && j == 0) {
				//Reactant
				SolutionMTDerivative(i, j, rea_j_i, rea_jp1_i, rea_j_i, rea_j_i, rea_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffReactantA, SolEquationCoefficient.CoeffReactantB, solutionIons.Reactant.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::sReactant, Assign);
				//Product
				SolutionMTDerivative(i, j, pro_j_i, pro_jp1_i, pro_j_i, pro_j_i, pro_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffProductA, SolEquationCoefficient.CoeffProductB, solutionIons.Product.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::sProduct, Assign);
				//Anion
				SolutionMTDerivative(i, j, ani_j_i, ani_jp1_i, ani_j_i, ani_j_i, ani_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffAnionA, SolEquationCoefficient.CoeffAnionB, solutionIons.SupportAnion.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::sAnion, Assign);
				//Cation
				SolutionMTDerivative(i, j, cat_j_i, cat_jp1_i, cat_j_i, cat_j_i, cat_j_im1, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffCationA, SolEquationCoefficient.CoeffCationB, solutionIons.SupportCation.DensityN, BoundaryEnum::right_bottom_corner, SpeciesEnum::sCation, Assign);
				//Potential
				SolutionPotDerivative(i, j, rea_j_i, pro_j_i, ani_j_i, cat_j_i, pot_j_i, pot_jp1_i, pot_j_i, pot_j_i, pot_j_im1,
					SolEquationCoefficient.CoeffPotentialA, SolEquationCoefficient.CoeffPotentialB, solutionIons, BoundaryEnum::right_bottom_corner, Assign);
			}
		}
	}

	omp_destroy_lock(&writeLock);

	// initialise MatrixA
	MatrixA.setFromTriplets(MatrixAlist.begin(), MatrixAlist.end());
}

void solver::UpdateMatrixA()
{
	MatrixAAssignIndex = 0;
	initialiseMatrixA(&solver::LockedIndexAssign);
}

inline double solver::BulkMTEquation(long i, long j, double Xj_i, double Xjp1_i, double Xjm1_i, double Xj_ip1, double Xj_im1,
	double Xpot_j_i, double Xpot_jp1_i, double Xpot_jm1_i, double Xpot_j_ip1, double Xpot_j_im1,
	const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn)
{
	/*
	Example Equation:
	F(rea_j_i) =
		M.CoeffReactantA(1, j) * X(rea_jm1_i) + M.CoeffReactantA(2, j) * X(rea_jp1_i)
		+ M.CoeffReactantB(1, i) * X(rea_j_im1) + M.CoeffReactantB(2, i) * X(rea_j_ip1)
		+ (M.CoeffReactantA(0, j) + M.CoeffReactantB(0, i))*X(rea_j_i)

		+ M.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

		+ (M.CoeffReactantA(5, j) * X(pot_jp1_i) + M.CoeffReactantA(6, j) * X(pot_jm1_i)
			+ M.CoeffReactantB(5, i) * X(pot_j_ip1) + M.CoeffReactantB(6, i) * X(pot_j_im1)
			+ (M.CoeffReactantA(3, j) + M.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

		+ M.CoeffReactantB(4, i) * (X(rea_j_ip1) - X(rea_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
		+ membrane.Cren(j, i);
		*/

	return
		CA(1, j)*Xjm1_i + CA(2, j)*Xjp1_i
		+ CB(1, i)*Xj_im1 + CB(2, i)*Xj_ip1
		+ (CA(0, j) + CB(0, i))*Xj_i

		+ CA(4, j)*(Xjp1_i - Xjm1_i)*(Xpot_jp1_i - Xpot_jm1_i)

		+ (CA(5, j)*Xpot_jp1_i + CA(6, j)*Xpot_jm1_i 
		+ CB(5, i)*Xpot_j_ip1 + CB(6, i)*Xpot_j_im1 
		+ (CA(3, j) + CB(3, i))*Xpot_j_i)*Xj_i

		+ CB(4, i)*(Xj_ip1 - Xj_im1)*(Xpot_j_ip1 - Xpot_j_im1)
		+ Cn(j, i);
}

void solver::MembraneMTDerivative(long i, long j, long j_i, long jp1_i, long jm1_i, long j_ip1, long j_im1,	long pot_j_i, long pot_jp1_i, long pot_jm1_i, long pot_j_ip1, long pot_j_im1,
	const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn, BoundaryEnum::Boundary boundary, SpeciesEnum::Species species, void (solver::*Assign)(Tt))
{
	if (species > SpeciesEnum::mCation) {
		cout << "Improper species for MembraneMTDerivative";
		throw(ImproperSpecies());
	}

	double Djm1_i = CA(1, j) - CA(4, j)*(X(pot_jp1_i) - X(pot_jm1_i));
	double Djp1_i = CA(2, j) + CA(4, j)*(X(pot_jp1_i) - X(pot_jm1_i));
	double Dj_im1 = CB(1, i) - CB(4, i)*(X(pot_j_ip1) - X(pot_j_im1));
	double Dj_ip1 = CB(2, i) + CB(4, i)*(X(pot_j_ip1) - X(pot_j_im1));
	double Dj_i = (CA(0, j) + CB(0, i)) + (CA(5, j)*X(pot_jp1_i) + CA(6, j)*X(pot_jm1_i) + CB(5, i)*X(pot_j_ip1) + CB(6, i)*X(pot_j_im1) + (CA(3, j) + CB(3, i))*X(pot_j_i));
	double Dpot_jm1_i = -CA(4, j)*(X(jp1_i) - X(jm1_i)) + CA(6, j)*X(j_i);
	double Dpot_jp1_i = CA(4, j)*(X(jp1_i) - X(jm1_i)) + CA(5, j)*X(j_i);
	double Dpot_j_im1 = CB(6, i)*X(j_i) - CB(4, i)*(X(j_ip1) - X(j_im1));
	double Dpot_j_ip1 = CB(5, i)*X(j_i) + CB(4, i)*(X(j_ip1) - X(j_im1));
	double Dpot_j_i = CB(3, i)*X(j_i);


	switch (boundary)
	{
	case BoundaryEnum::bulk: {
		(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
		(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
		(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
		(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
		(this->*Assign)(Tt(j_i, j_i, Dj_i));
		(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i));
		}
		break;
	case BoundaryEnum::bottom: {
		switch (species)
		{
		case SpeciesEnum::mReactant: {
			double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
			double kf = ElecR.kf(DrivingPotential);
			double kb = ElecR.kb(DrivingPotential);
			double erDpot_jp1_i = kf*ElecR.minusAlfaNF_R_T*ElecR.DrivingPotentialCoeff / membrane.dz;
			double erDpot_j_i = -Dpot_jp1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i + erDpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + erDpot_j_i));

			(this->*Assign)(Tt(j_i, j_i + membrane.Getmxn(), kf / membrane.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::mProduct: {
			double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
			double kf = ElecR.kf(DrivingPotential);
			double kb = ElecR.kb(DrivingPotential);
			double erDpot_jp1_i = kf*ElecR.minusAlfaNF_R_T*ElecR.DrivingPotentialCoeff / membrane.dz;
			double erDpot_j_i = -Dpot_jp1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i - erDpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i - erDpot_j_i));

			(this->*Assign)(Tt(j_i, j_i - membrane.Getmxn(), -kf / membrane.dz*Signal.dt));
		}	
			break;
		case SpeciesEnum::mCation: {
			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i));
		}
			break;
		default: {
			std::cout << "No " << species;
			throw(ImproperSpecies());
		}
			break;
		}
	}	
		break;
	case BoundaryEnum::top: {
		switch (species)
		{
		case SpeciesEnum::mReactant: {
			//Reactant index
			long srea_jp1_i = Index1d(SpeciesEnum::sReactant, 0, i);
			//Reactant transfer rate
			double kf = ReactantTransR.kf(0);
			double kb = ReactantTransR.kb(0);
			double ReactantTransRate = kf*X(srea_jp1_i) - kb*X(j_i);

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i));

			(this->*Assign)(Tt(j_i, srea_jp1_i, kf / membrane.dz*Signal.dt));
		}
									 break;
		case SpeciesEnum::mProduct: {
			//Potential index
			long spot_jp1_i = Index1d(SpeciesEnum::sPotential, 0, i);
			double dE = X(spot_jp1_i) - X(pot_j_i);
			//Product index
			long spro_jp1_i = Index1d(SpeciesEnum::sProduct, 0, i);
			//Product transfer rate
			double kf = ProductTransR.kf(dE);
			double kb = ProductTransR.kb(dE);
			double ProductTransRate = kf*X(spro_jp1_i) - kb*X(j_i);
			double inDspot_jp1_i = ProductTransR.minusAlfaNF_R_T*kf*X(spro_jp1_i) - ProductTransR.AlfaMinusOneNF_R_T*kb*X(j_i);
			double inDpot_j_i = -inDspot_jp1_i;

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + inDpot_j_i / membrane.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, spro_jp1_i, kf / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, spot_jp1_i, inDspot_jp1_i / membrane.dz*Signal.dt));
		}
									break;
		case SpeciesEnum::mCation: {
			//Potential index
			long spot_jp1_i = Index1d(SpeciesEnum::sPotential, 0, i);
			double dE = X(spot_jp1_i) - X(pot_j_i);
			//Cation index
			long scat_jp1_i = Index1d(SpeciesEnum::sCation, 0, i);
			//Cation transfer rate
			double kf = CationTransR.kf(dE);
			double kb = CationTransR.kb(dE);
			double CationTransRate = kf*X(scat_jp1_i) - kb*X(j_i);
			double inDspot_jp1_i = CationTransR.minusAlfaNF_R_T*kf*X(scat_jp1_i) - CationTransR.AlfaMinusOneNF_R_T*kb*X(j_i);
			double inDpot_j_i = -inDspot_jp1_i;

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + inDpot_j_i / membrane.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, scat_jp1_i, kf / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, spot_jp1_i, inDspot_jp1_i / membrane.dz*Signal.dt));
		}
			break;
		default: {
			std::cout << "miss membrane phase (" << i << ", " << j << ")\n";
			throw(MissPhase());
		}
			break;
		}
	}
		break;
	case BoundaryEnum::left: {
		(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
		(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
		(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
		(this->*Assign)(Tt(j_i, j_i, Dj_i + Dj_im1));
		(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_j_im1));
	}
		break;
	case BoundaryEnum::right: {
		(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
		(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
		(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
		(this->*Assign)(Tt(j_i, j_i, Dj_i + Dj_ip1));
		(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_j_ip1));
	}
		break;
	case BoundaryEnum::left_bottom_corner: {
		switch (species)
		{
		case SpeciesEnum::mReactant: {
			double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
			double kf = ElecR.kf(DrivingPotential);
			double kb = ElecR.kb(DrivingPotential);
			double erDpot_jp1_i = kf*ElecR.minusAlfaNF_R_T*ElecR.DrivingPotentialCoeff / membrane.dz;
			double erDpot_j_i = -Dpot_jp1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i + erDpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1 + erDpot_j_i));

			(this->*Assign)(Tt(j_i, j_i + membrane.Getmxn(), kf / membrane.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::mProduct: {
			double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
			double kf = ElecR.kf(DrivingPotential);
			double kb = ElecR.kb(DrivingPotential);
			double erDpot_jp1_i = kf*ElecR.minusAlfaNF_R_T*ElecR.DrivingPotentialCoeff / membrane.dz;
			double erDpot_j_i = -Dpot_jp1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1 + kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i - erDpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1 - erDpot_j_i));

			(this->*Assign)(Tt(j_i, j_i - membrane.Getmxn(), -kf / membrane.dz*Signal.dt));
			}
			break;
		case SpeciesEnum::mCation: {
			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1));
			}
			break;
		default: {
			std::cout << "No " << species;
			throw(ImproperSpecies());
			}
			break;
		}
	}
		
		break;
	case BoundaryEnum::right_bottom_corner: {
		switch (species)
		{
		case SpeciesEnum::mReactant: {
			double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
			double kf = ElecR.kf(DrivingPotential);
			double kb = ElecR.kb(DrivingPotential);
			double erDpot_jp1_i = kf*ElecR.minusAlfaNF_R_T*ElecR.DrivingPotentialCoeff / membrane.dz;
			double erDpot_j_i = -Dpot_jp1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_ip1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i + erDpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_ip1 + erDpot_j_i));

			(this->*Assign)(Tt(j_i, j_i + membrane.Getmxn(), kf / membrane.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::mProduct: {
			double DrivingPotential = ElecR.DrivingPotential((X(pot_jp1_i) - X(pot_j_i)) / membrane.dz);
			double kf = ElecR.kf(DrivingPotential);
			double kb = ElecR.kb(DrivingPotential);
			double erDpot_jp1_i = kf*ElecR.minusAlfaNF_R_T*ElecR.DrivingPotentialCoeff / membrane.dz;
			double erDpot_j_i = -Dpot_jp1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_ip1 + kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i - erDpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_ip1 - erDpot_j_i));

			(this->*Assign)(Tt(j_i, j_i - membrane.Getmxn(), -kf / membrane.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::mCation: {
			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_ip1));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_ip1));
		}
			break;
		default: {
			std::cout << "No " << species;
			throw(ImproperSpecies());
		}
			break;
		}
	}
		break;
	case BoundaryEnum::left_upper_corner: {
		switch (species)
		{
		case SpeciesEnum::mReactant: {
			//Reactant index
			long srea_jp1_i = solution.n*i + 4 * membrane.Getmxn();
			//Reactant transfer rate
			double kf = ReactantTransR.kf(0);
			double kb = ReactantTransR.kb(0);
			double ReactantTransRate = kf*X(srea_jp1_i) - kb*X(j_i);

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i + Dj_im1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_im1));

			(this->*Assign)(Tt(j_i, srea_jp1_i, kf / membrane.dz*Signal.dt));
		}	
			break;
		case SpeciesEnum::mProduct: {
			//Potential index
			long spot_jp1_i = Index1d(SpeciesEnum::sPotential, 0, i);
			double dE = X(spot_jp1_i) - X(pot_j_i);
			//Product index
			long spro_jp1_i = Index1d(SpeciesEnum::sProduct, 0, i);
			//Product transfer rate
			double kf = ProductTransR.kf(dE);
			double kb = ProductTransR.kb(dE);
			double ProductTransRate = kf*X(spro_jp1_i) - kb*X(j_i);
			double inDspot_jp1_i = ProductTransR.minusAlfaNF_R_T*kf*X(spro_jp1_i) - ProductTransR.AlfaMinusOneNF_R_T*kb*X(j_i);
			double inDpot_j_i = -inDspot_jp1_i;

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i + Dj_im1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_im1 + inDpot_j_i / membrane.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, spro_jp1_i, kf / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, spot_jp1_i, inDspot_jp1_i / membrane.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::mCation: {
			//Potential index
			long spot_jp1_i = Index1d(SpeciesEnum::sPotential, 0, i);
			double dE = X(spot_jp1_i) - X(pot_j_i);
			//Cation index
			long scat_jp1_i = Index1d(SpeciesEnum::sCation, 0, i);
			//Cation transfer rate
			double kf = CationTransR.kf(dE);
			double kb = CationTransR.kb(dE);
			double CationTransRate = kf*X(scat_jp1_i) - kb*X(j_i);
			double inDspot_jp1_i = CationTransR.minusAlfaNF_R_T*kf*X(scat_jp1_i) - CationTransR.AlfaMinusOneNF_R_T*kb*X(j_i);
			double inDpot_j_i = -inDspot_jp1_i;

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i + Dj_im1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_im1 + inDpot_j_i / membrane.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, scat_jp1_i, kf / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, spot_jp1_i, inDspot_jp1_i / membrane.dz*Signal.dt));
		}
			break;
		default: {
			std::cout << "No " << species;
			throw(ImproperSpecies());
		}
			break;
		}
	}
		break;
	case BoundaryEnum::right_upper_corner: {
		switch (species)
		{
		case SpeciesEnum::mReactant: {
			//Reactant index
			long srea_jp1_i = Index1d(SpeciesEnum::sReactant, 0, i);
			//Reactant transfer rate
			double kf = ReactantTransR.kf(0);
			double kb = ReactantTransR.kb(0);
			double ReactantTransRate = kf*X(srea_jp1_i) - kb*X(j_i);

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i + Dj_ip1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_ip1));

			(this->*Assign)(Tt(j_i, srea_jp1_i, kf / membrane.dz*Signal.dt));
		}	
			break;
		case SpeciesEnum::mProduct: {
			//Potential index
			long spot_jp1_i = Index1d(SpeciesEnum::sPotential, 0, i);
			double dE = X(spot_jp1_i) - X(pot_j_i);
			//Product index
			long spro_jp1_i = Index1d(SpeciesEnum::sProduct, 0, i);
			//Product transfer rate
			double kf = ProductTransR.kf(dE);
			double kb = ProductTransR.kb(dE);
			double ProductTransRate = kf*X(spro_jp1_i) - kb*X(j_i);
			double inDspot_jp1_i = ProductTransR.minusAlfaNF_R_T*kf*X(spro_jp1_i) - ProductTransR.AlfaMinusOneNF_R_T*kb*X(j_i);
			double inDpot_j_i = -inDspot_jp1_i;

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i + Dj_ip1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_ip1 + inDpot_j_i / membrane.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, spro_jp1_i, kf / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, spot_jp1_i, inDspot_jp1_i / membrane.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::mCation: {
			//Potential index
			long spot_jp1_i = Index1d(SpeciesEnum::sPotential, 0, i);
			double dE = X(spot_jp1_i) - X(pot_j_i);
			//Cation index
			long scat_jp1_i = Index1d(SpeciesEnum::sCation, 0, i);
			//Cation transfer rate
			double kf = CationTransR.kf(dE);
			double kb = CationTransR.kb(dE);
			double CationTransRate = kf*X(scat_jp1_i) - kb*X(j_i);
			double inDspot_jp1_i = CationTransR.minusAlfaNF_R_T*kf*X(scat_jp1_i) - CationTransR.AlfaMinusOneNF_R_T*kb*X(j_i);
			double inDpot_j_i = -inDspot_jp1_i;

			(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djp1_i + Dj_ip1 - kb / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_ip1 + inDpot_j_i / membrane.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, scat_jp1_i, kf / membrane.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, spot_jp1_i, inDspot_jp1_i / membrane.dz*Signal.dt));
		}
			break;
		default: {
			std::cout << "No " << species;
			throw(ImproperSpecies());
		}
			break;
		}
	}
		break;
	default: {
		std::cout << "miss membrane phase (" << i << ", " << j << ")\n";
		throw(MissPhase());
	}
		break;
	}
}

void solver::SolutionMTDerivative(long i, long j, long j_i, long jp1_i, long jm1_i, long j_ip1, long j_im1,	long pot_j_i, long pot_jp1_i, long pot_jm1_i, long pot_j_ip1, long pot_j_im1,
	const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const Eigen::MatrixXd& Cn, BoundaryEnum::Boundary boundary, SpeciesEnum::Species species, void (solver::*Assign)(Tt))
{
	if (species < SpeciesEnum::sReactant) {
		cout << "Improper species for SolutionMTDerivative";
		throw(ImproperSpecies());
	}
	
	double Djm1_i = CA(1, j) - CA(4, j)*(X(pot_jp1_i) - X(pot_jm1_i));
	double Djp1_i = CA(2, j) + CA(4, j)*(X(pot_jp1_i) - X(pot_jm1_i));
	double Dj_im1 = CB(1, i) - CB(4, i)*(X(pot_j_ip1) - X(pot_j_im1));
	double Dj_ip1 = CB(2, i) + CB(4, i)*(X(pot_j_ip1) - X(pot_j_im1));
	double Dj_i = (CA(0, j) + CB(0, i)) + (CA(5, j)*X(pot_jp1_i) + CA(6, j)*X(pot_jm1_i) + CB(5, i)*X(pot_j_ip1) + CB(6, i)*X(pot_j_im1) + (CA(3, j) + CB(3, i))*X(pot_j_i));
	double Dpot_jm1_i = -CA(4, j)*(X(jp1_i) - X(jm1_i)) + CA(6, j)*X(j_i);
	double Dpot_jp1_i = CA(4, j)*(X(jp1_i) - X(jm1_i)) + CA(5, j)*X(j_i);
	double Dpot_j_im1 = CB(6, i)*X(j_i) - CB(4, i)*(X(j_ip1) - X(j_im1));
	double Dpot_j_ip1 = CB(5, i)*X(j_i) + CB(4, i)*(X(j_ip1) - X(j_im1));
	double Dpot_j_i = CB(3, i)*X(j_i);
	
	switch (boundary)
	{
	case BoundaryEnum::bulk: {
		(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
		(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
		(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
		(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
		(this->*Assign)(Tt(j_i, j_i, Dj_i));
		(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i));
	}
		break;
	case BoundaryEnum::bottom: {
		switch (species)
		{
		case SpeciesEnum::sReactant: {
			long mrea_jm1_i = membrane.n*i + membrane.n - 1;
			//Reactant transfer rate
			double kf = ReactantTransR.kf(0);
			double kb = ReactantTransR.kb(0);
			double ReactantTransRate = kf*X(j_i) - kb*X(mrea_jm1_i);

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i - kf / solution.dz * Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i));

			(this->*Assign)(Tt(j_i, mrea_jm1_i, kb / solution.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::sProduct: {
			long mpro_jm1_i = membrane.n*i + membrane.n - 1 + membrane.Getmxn();
			//Potential index
			long mpot_jm1_i = membrane.n*i + membrane.n - 1 + 3 * membrane.Getmxn();
			double dE = X(pot_j_i) - X(mpot_jm1_i);
			double kf = ProductTransR.kf(dE);
			double kb = ProductTransR.kb(dE);
			double trDmpot_jm1_i = ProductTransR.minusAlfaNF_R_T*kf*X(j_i) - ProductTransR.AlfaMinusOneNF_R_T*kb*X(mpro_jm1_i);
			double trDpot_j_i = -trDmpot_jm1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i - kf / solution.dz * Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + trDpot_j_i / solution.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, mpot_jm1_i, trDmpot_jm1_i / solution.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, mpro_jm1_i, kb / solution.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::sAnion: {
			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i));
		}
			break;
		case SpeciesEnum::sCation: {
			//Potential index
			long mcat_jm1_i = membrane.n*i + membrane.n - 1 + 2 * membrane.Getmxn();
			long mpot_jm1_i = membrane.n*i + membrane.n - 1 + 3 * membrane.Getmxn();
			double dE = X(pot_j_i) - X(mpot_jm1_i);
			//Cation transfer rate
			double kf = CationTransR.kf(dE);
			double kb = CationTransR.kb(dE);
			double trDmpot_jm1_i = CationTransR.minusAlfaNF_R_T*kf*X(j_i) - CationTransR.AlfaMinusOneNF_R_T*kb*X(mcat_jm1_i);
			double trDpot_j_i = -trDmpot_jm1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i - kf / solution.dz * Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + trDpot_j_i / solution.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, mpot_jm1_i, trDmpot_jm1_i / solution.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, mcat_jm1_i, kb / solution.dz*Signal.dt));
		}
			break;
		default: {
			cout << " Improper Spcecies for SolutionMTDerivative Function";
			throw(ImproperSpecies());
		}
			break;
		}
	}
		break;
	case BoundaryEnum::top: {
		(this->*Assign)(Tt(j_i, j_i, 1));
	}
		break;
	case BoundaryEnum::left: {
		(this->*Assign)(Tt(j_i, jm1_i, Djm1_i));
		(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
		(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
		(this->*Assign)(Tt(j_i, j_i, Dj_i + Dj_im1));
		(this->*Assign)(Tt(j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_j_im1));
	}
		break;
	case BoundaryEnum::right: {
		(this->*Assign)(Tt(j_i, j_i, 1));
	}
		break;
	case BoundaryEnum::right_bottom: {
		(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
		(this->*Assign)(Tt(j_i, j_im1, Dj_im1));
		(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
		(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i));
		(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i));
	}
		break;
	case BoundaryEnum::left_bottom_corner: {
		switch (species)
		{
		case SpeciesEnum::sReactant: {
			long mrea_jm1_i = membrane.n*i + membrane.n - 1;
			//Reactant transfer rate
			double kf = ReactantTransR.kf(0);
			double kb = ReactantTransR.kb(0);
			double ReactantTransRate = kf*X(j_i) - kb*X(mrea_jm1_i);

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1 - kf / solution.dz * Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1));

			(this->*Assign)(Tt(j_i, mrea_jm1_i, kb / solution.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::sProduct: {
			long mpro_jm1_i = membrane.n*i + membrane.n - 1 + membrane.Getmxn();
			//Potential index
			long mpot_jm1_i = membrane.n*i + membrane.n - 1 + 3 * membrane.Getmxn();
			double dE = X(pot_j_i) - X(mpot_jm1_i);
			double kf = ProductTransR.kf(dE);
			double kb = ProductTransR.kb(dE);
			double trDmpot_jm1_i = ProductTransR.minusAlfaNF_R_T*kf*X(j_i) - ProductTransR.AlfaMinusOneNF_R_T*kb*X(mpro_jm1_i);
			double trDpot_j_i = -trDmpot_jm1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1 - kf / solution.dz * Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1 + trDpot_j_i / solution.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, mpot_jm1_i, trDmpot_jm1_i / solution.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, mpro_jm1_i, kb / solution.dz*Signal.dt));
		}
			break;
		case SpeciesEnum::sAnion: {
			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1));
		}
			break;
		case SpeciesEnum::sCation: {
			//Potential index
			long mcat_jm1_i = membrane.n*i + membrane.n - 1 + 2 * membrane.Getmxn();
			long mpot_jm1_i = membrane.n*i + membrane.n - 1 + 3 * membrane.Getmxn();
			double dE = X(pot_j_i) - X(mpot_jm1_i);
			//Cation transfer rate
			double kf = CationTransR.kf(dE);
			double kb = CationTransR.kb(dE);
			double trDmpot_jm1_i = CationTransR.minusAlfaNF_R_T*kf*X(j_i) - CationTransR.AlfaMinusOneNF_R_T*kb*X(mcat_jm1_i);
			double trDpot_j_i = -trDmpot_jm1_i;

			(this->*Assign)(Tt(j_i, jp1_i, Djp1_i));
			(this->*Assign)(Tt(j_i, j_ip1, Dj_ip1));
			(this->*Assign)(Tt(j_i, j_i, Dj_i + Djm1_i + Dj_im1 - kf / solution.dz * Signal.dt));
			(this->*Assign)(Tt(j_i, pot_jp1_i, Dpot_jp1_i));
			(this->*Assign)(Tt(j_i, pot_j_ip1, Dpot_j_ip1));
			(this->*Assign)(Tt(j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1 + trDpot_j_i / solution.dz*Signal.dt));

			(this->*Assign)(Tt(j_i, mpot_jm1_i, trDmpot_jm1_i / solution.dz*Signal.dt));
			(this->*Assign)(Tt(j_i, mcat_jm1_i, kb / solution.dz*Signal.dt));
		}
			break;
		default: {
			cout << " Improper Spcecies for SolutionMTDerivative Function";
			throw(ImproperSpecies());
		}
			break;
		}
	}
		break;
	case BoundaryEnum::right_bottom_corner: {
		(this->*Assign)(Tt(j_i, j_i, 1));
	}
		break;
	case BoundaryEnum::left_upper_corner: {
		(this->*Assign)(Tt(j_i, j_i, 1));
	}
		break;
	case BoundaryEnum::right_upper_corner: {
		(this->*Assign)(Tt(j_i, j_i, 1));
	}
		break;
	default: {
		std::cout << "miss solution phase (" << i << ", " << j << ")\n";
		throw(MissPhase());
	}
		break;
	}
}

inline double solver::BulkPotEquation(long i, long j, double Xrea_j_i, double Xpro_j_i, double Xani_j_i, double Xcat_j_i, double Xpot_j_i, double Xpot_jp1_i, double Xpot_jm1_i, double Xpot_j_ip1, double Xpot_j_im1,
	const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I)
{
	/*
	Example Equation:
	F(pot_j_i) =
		M.CoeffPotentialA(1, j) * X(pot_jm1_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
		+ M.CoeffPotentialB(1, i) * X(pot_j_im1) + M.CoeffPotentialB(2, i) * X(pot_j_ip1)
		+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
		+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
			+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
			+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	*/

	return
		CA(1, j)*Xpot_jm1_i + CA(2, j)*Xpot_jp1_i
		+ CB(1, i)*Xpot_j_im1 + CB(2, i)*Xpot_j_ip1
		+ (CA(0, j) + CB(0, i))*Xpot_j_i
		+ (I.Reactant.Z*Xrea_j_i + I.Product.Z*Xpro_j_i 
		+ I.SupportAnion.Z*Xani_j_i + I.SupportCation.Z*Xcat_j_i 
		+ I.CxZImmobileCharge)*I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;

}

void solver:: SolutionPotDerivative(long i, long j, long rea_j_i, long pro_j_i, long ani_j_i, long cat_j_i,	long pot_j_i, long pot_jp1_i, long pot_jm1_i, long pot_j_ip1, long pot_j_im1,
	const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I, BoundaryEnum::Boundary boundary, void (solver::*Assign)(Tt))
{
	double Dpot_jm1_i = CA(1, j);
	double Dpot_jp1_i = CA(2, j);
	double Dpot_j_im1 = CB(1, i);
	double Dpot_j_ip1 = CB(2, j);
	double Dpot_j_i = CA(0, j) + CB(0, i);
	double Drea_j_i = I.Reactant.Z*I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	double Dpro_j_i = I.Product.Z*I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	double Dani_j_i = I.SupportAnion.Z*I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	double Dcat_j_i = I.SupportCation.Z* I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	
	switch (boundary)
	{
	case BoundaryEnum::bulk: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, ani_j_i, Dani_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::bottom: {
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, ani_j_i, Dani_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::top: {
		(this->*Assign)(Tt(pot_j_i, pot_j_i, 1));
	}
		break;
	case BoundaryEnum::left: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, ani_j_i, Dani_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::right: {
		(this->*Assign)(Tt(pot_j_i, pot_j_i, 1));
	}
		break;
	case BoundaryEnum::right_bottom: {
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, ani_j_i, Dani_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::left_bottom_corner: {
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_jm1_i + Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, ani_j_i, Dani_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::right_bottom_corner: {
		(this->*Assign)(Tt(pot_j_i, pot_j_i, 1));
	}
		break;
	case BoundaryEnum::left_upper_corner: {
		(this->*Assign)(Tt(pot_j_i, pot_j_i, 1));
	}
		break;
	case BoundaryEnum::right_upper_corner: {
		(this->*Assign)(Tt(pot_j_i, pot_j_i, 1));
	}
		break;
	default: {
		std::cout << "miss solution phase (" << i << ", " << j << ")\n";
		throw(MissPhase());
	}
		break;
	}
}

void solver::MembranePotDerivative(long i, long j, long rea_j_i, long pro_j_i, long cat_j_i, long pot_j_i, long pot_jp1_i, long pot_jm1_i, long pot_j_ip1, long pot_j_im1,
	const Eigen::MatrixXd& CA, const Eigen::MatrixXd& CB, const IonSystem& I, BoundaryEnum::Boundary boundary, void (solver::*Assign)(Tt))
{
	double Dpot_jm1_i = CA(1, j);
	double Dpot_jp1_i = CA(2, j);
	double Dpot_j_im1 = CB(1, i);
	double Dpot_j_ip1 = CB(2, j);
	double Dpot_j_i = CA(0, j) + CB(0, i);
	double Drea_j_i = I.Reactant.Z*I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	double Dpro_j_i = I.Product.Z*I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
	double Dcat_j_i = I.SupportCation.Z* I.ReciprocalEpsilon_rEpsilon_0*Thermo.F;

	switch (boundary)
	{
	case BoundaryEnum::bulk: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::bottom: {
		double Dpot_j_i = ElecR.DrivingPotentialCoeff / membrane.dz - 1;
		double Dpot_jp1_i = -ElecR.DrivingPotentialCoeff / membrane.dz;

		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
	}
		break;
	case BoundaryEnum::top: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::left: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::right: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::left_bottom_corner: {
		double Dpot_j_i = ElecR.DrivingPotentialCoeff / membrane.dz - 1;
		double Dpot_jp1_i = -ElecR.DrivingPotentialCoeff / membrane.dz;

		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
	}
		break;
	case BoundaryEnum::right_bottom_corner: {
		double Dpot_j_i = ElecR.DrivingPotentialCoeff / membrane.dz - 1;
		double Dpot_jp1_i = -ElecR.DrivingPotentialCoeff / membrane.dz;

		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i));
		(this->*Assign)(Tt(pot_j_i, pot_jp1_i, Dpot_jp1_i));
	}
		break;
	case BoundaryEnum::left_upper_corner: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_ip1, Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	case BoundaryEnum::right_upper_corner: {
		(this->*Assign)(Tt(pot_j_i, pot_jm1_i, Dpot_jm1_i));
		(this->*Assign)(Tt(pot_j_i, pot_j_im1, Dpot_j_im1));
		(this->*Assign)(Tt(pot_j_i, pot_j_i, Dpot_j_i + Dpot_jp1_i + Dpot_j_ip1));
		(this->*Assign)(Tt(pot_j_i, rea_j_i, Drea_j_i));
		(this->*Assign)(Tt(pot_j_i, pro_j_i, Dpro_j_i));
		(this->*Assign)(Tt(pot_j_i, cat_j_i, Dcat_j_i));
	}
		break;
	default: {
		std::cout << "miss membrane phase (" << i << ", " << j << ")\n";
		throw(MissPhase());
	}
		break;
	}
}

inline void solver::LockedPushBack(Tt triplet)
{
	omp_set_lock(&writeLock);
	MatrixAlist.push_back(triplet);
	omp_unset_lock(&writeLock);
}

inline void solver::LockedIndexAssign(Tt triplet)
{
	omp_set_lock(&writeLock);
	long index = MatrixAAssignIndex++;
	omp_unset_lock(&writeLock);
	MatrixAlist[index] = triplet;
}

void solver::SaveDensity()
{
	const auto& index = Index1d;
	const auto& x = X;

	auto SD = [&index, &x](SpeciesEnum::Species species, Eigen::MatrixXd& DensityN, const long m, const long n) {
#pragma omp parallel for
		for (long i = 0; i < m; ++i) {
			for (long j = 0; j < n; ++j) {
				DensityN(j, i) = x(index(species, j, i));
			}
		}
	};
	// membrane
	// mReactant
	SD(SpeciesEnum::mReactant, membraneIons.Reactant.DensityN, membrane.m, membrane.n);
	// mProduct
	SD(SpeciesEnum::mProduct, membraneIons.Product.DensityN, membrane.m, membrane.n);
	// mCation
	SD(SpeciesEnum::mCation, membraneIons.SupportCation.DensityN, membrane.m, membrane.n);
	// mPotential
	SD(SpeciesEnum::mPotential, membraneIons.Potential.DensityN, membrane.m, membrane.n);

	// solution
	// sReactant
	SD(SpeciesEnum::sReactant, solutionIons.Reactant.DensityN, solution.m, solution.n);
	// sProduct
	SD(SpeciesEnum::sProduct, solutionIons.Product.DensityN, solution.m, solution.n);
	// sCation
	SD(SpeciesEnum::sCation, solutionIons.SupportCation.DensityN, solution.m, solution.n);
	// sAnion
	SD(SpeciesEnum::sAnion, solutionIons.SupportAnion.DensityN, solution.m, solution.n);
	// sPotential
	SD(SpeciesEnum::sPotential, solutionIons.Potential.DensityN, solution.m, solution.n);
}

double solver::FaradaicCurrent() const
{
	double DrivingPotential(0.0), kf(0.0), kb(0.0), reactionRate(0.0), I(0.0);

#pragma omp parallel for private(DrivingPotential, kf, kb, reactionRate) reduction(+ : I)
	for (long i = 0; i < membrane.m; ++i) {
		DrivingPotential = ElecR.DrivingPotential((membraneIons.Potential.DensityN(1, i) - membraneIons.Potential.DensityN(0, i)) / membrane.dz);
		kf = ElecR.kf(DrivingPotential);
		kb = ElecR.kb(DrivingPotential);
		reactionRate = kf*membraneIons.Product.DensityN(0, i) - kb*membraneIons.Reactant.DensityN(0, i);

		I += reactionRate*membrane.RR(0, i);
	}

	I *= 6.28*(membraneIons.Reactant.Z - membraneIons.Product.Z)*Thermo.F / 1000 * membrane.dr;
	
	return I;
}