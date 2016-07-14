#include "solver.h"
#include <vector>

typedef Eigen::Triplet<double> Tt;

solver::solver(mesh& fmembrane, mesh& fsolution, const IonSystem& fmembraneIons, const IonSystem& fsolutionIons, PotentialSignal& fSignal, const nernst_equation& fThermo) :
	membrane(fmembrane), solution(fsolution), membraneIons(fmembraneIons), solutionIons(fsolutionIons), Signal(fSignal), Thermo(fThermo),
	MatrixLen((fmembrane.m*fmembrane.n + fsolution.m*fsolution.n)*5),
	MatrixA(MatrixLen, MatrixLen),
	arrayb(MatrixLen),
	dX(MatrixLen),
	X(MatrixLen),
	F(MatrixLen),
	MemEquationCoefficient(fmembraneIons, fmembrane, fSignal, fThermo),
	SolEquationCoefficient(fsolutionIons, fsolution, fSignal, fThermo)
{}

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
}

void solver::GeoCoefficientA(mesh& phase, Eigen::MatrixXd& GeoCoeffA) const
{
	double dZ1, dZ2;
	for (unsigned i = 0; i < phase.n; ++i) {
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
	double dR1, double dR2, R;
	for (unsigned long i = 0; i < phase.m; ++i) {
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
	unsigned long nxiplusj = 0UL;
	unsigned long mxn = membrane.Getmxn();

#pragma omp parallel for private(nxiplusj)
	for (unsigned long int i = 0; i < membrane.m; ++i) {
		for (unsigned long j = 0; j < membrane.n; ++j) {

			nxiplusj = membrane.n*i + j;
			//Reactant
			X(nxiplusj) = membrane.Cren(j, i);
			//Product
			X(nxiplusj + mxn) = membrane.Cprn(j, i);
			//Anion
			X(nxiplusj + 2 * mxn) = membrane.Cann(j, i);
			//Cation
			X(nxiplusj + 3 * mxn) = membrane.Ccan(j, i);
			//Potential
			X(nxiplusj + 4 * mxn) = membrane.Ptln(j, i);
		}
	}

	unsigned long int mxnx5 = 5 * mxn;
	mxn = solution.Getmxn();

#pragma omp parallel for private(nxiplusj)
	for (unsigned long i = 0; i < solution.m; ++i) {
		for (unsigned long j = 0; j < solution.n; ++j) {
			
			nxiplusj = solution.n*i + j;
			//Reactant
			X(nxiplusj + mxnx5) = solution.Cren(j, i);
			//Product
			X(nxiplusj + 1 * mxn + mxnx5) = solution.Cprn(j, i);
			//Anion
			X(nxiplusj + 2 * mxn + mxnx5) = solution.Cann(j, i);
			//Cation
			X(nxiplusj + 3 * mxn + mxnx5) = solution.Ccan(j, i);
			//Potential
			X(nxiplusj + 4 * mxn + mxnx5) = solution.Ptln(j, i);
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
	unsigned long rea_j_i = 0UL;
	unsigned long rea_jm1_i = 0UL;
	unsigned long rea_jp1_i = 0UL;
	unsigned long rea_j_im1 = 0UL;
	unsigned long rea_j_ip1 = 0UL;
	//Product index
	unsigned long pro_j_i = 0UL;
	unsigned long pro_jm1_i = 0UL;
	unsigned long pro_jp1_i = 0UL;
	unsigned long pro_j_im1 = 0UL;
	unsigned long pro_j_ip1 = 0UL;
	//Anion index
	unsigned long ani_j_i = 0UL;
	unsigned long ani_jm1_i = 0UL;
	unsigned long ani_jp1_i = 0UL;
	unsigned long ani_j_im1 = 0UL;
	unsigned long ani_j_ip1 = 0UL;
	//Cation index
	unsigned long cat_j_i = 0UL;
	unsigned long cat_jm1_i = 0UL;
	unsigned long cat_jp1_i = 0UL;
	unsigned long cat_j_im1 = 0UL;
	unsigned long cat_j_ip1 = 0UL;
	//Potential index
	unsigned long pot_j_i = 0UL;
	unsigned long pot_jm1_i = 0UL;
	unsigned long pot_jp1_i = 0UL;
	unsigned long pot_j_im1 = 0UL;
	unsigned long pot_j_ip1 = 0UL;

	unsigned long mxn = membrane.Getmxn();

	// Calculate membrane bulk
#pragma omp parallel for private(rea_j_i, rea_jm1_i, rea_jp1_i, rea_j_im1, rea_j_ip1, pro_j_i, pro_jm1_i, pro_jp1_i, pro_j_im1, pro_j_ip1, ani_j_i, ani_jm1_i, ani_jp1_i, ani_j_im1, ani_j_ip1,cat_j_i, cat_jm1_i, cat_jp1_i, cat_j_im1, cat_j_ip1, pot_j_i, pot_jm1_i, pot_jp1_i, pot_j_im1, pot_j_ip1)
	for (unsigned long i = 1; i < membrane.m - 1; ++i) {
		for (unsigned long j = 1; j < membrane.n - 1; ++j) {
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
			//Anion index
			ani_j_i = pro_j_i + mxn;
			ani_jm1_i = ani_j_i - 1;
			ani_jp1_i = ani_j_i + 1;
			ani_j_im1 = ani_j_i - membrane.n;
			ani_j_ip1 = ani_j_i + membrane.n;
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
				//Product
				F(pro_j_i) =
					M.CoeffProductA(1, j) * X(pro_jm1_i) + M.CoeffProductA(2, j) * X(pro_jp1_i)
					+ M.CoeffProductB(1, i) * X(pro_j_im1) + M.CoeffProductB(2, i) * X(pro_j_ip1)
					+ (M.CoeffProductA(0, j) + M.CoeffProductB(0, i))*X(pro_j_i)

					+ M.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffProductA(5, j) * X(pot_jp1_i) + M.CoeffProductA(6, j) * X(pot_jm1_i)
						+ M.CoeffProductB(5, i) * X(pot_j_ip1) + M.CoeffProductB(6, i) * X(pot_j_im1)
						+ (M.CoeffProductA(3, j) + M.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

					+ M.CoeffProductB(4, i) * (X(pro_j_ip1) - X(pro_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Cprn(j, i);
				//Anion
				F(ani_j_i) =
					M.CoeffAnionA(1, j) * X(ani_jm1_i) + M.CoeffAnionA(2, j) * X(ani_jp1_i)
					+ M.CoeffAnionB(1, i) * X(ani_j_im1) + M.CoeffAnionB(2, i) * X(ani_j_ip1)
					+ (M.CoeffAnionA(0, j) + M.CoeffAnionB(0, i))*X(ani_j_i)

					+ M.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffAnionA(5, j) * X(pot_jp1_i) + M.CoeffAnionA(6, j) * X(pot_jm1_i)
						+ M.CoeffAnionB(5, i) * X(pot_j_ip1) + M.CoeffAnionB(6, i) * X(pot_j_im1)
						+ (M.CoeffAnionA(3, j) + M.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

					+ M.CoeffAnionB(4, i) * (X(ani_j_ip1) - X(ani_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Cann(j, i);
				//Cation
				F(cat_j_i) =
					M.CoeffCationA(1, j) * X(cat_jm1_i) + M.CoeffCationA(2, j) * X(cat_jp1_i)
					+ M.CoeffCationB(1, i) * X(cat_j_im1) + M.CoeffCationB(2, i) * X(cat_j_ip1)
					+ (M.CoeffCationA(0, j) + M.CoeffCationB(0, i))*X(cat_j_i)

					+ M.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffCationA(5, j) * X(pot_jp1_i) + M.CoeffCationA(6, j) * X(pot_jm1_i)
						+ M.CoeffCationB(5, i) * X(pot_j_ip1) + M.CoeffCationB(6, i) * X(pot_j_im1)
						+ (M.CoeffCationA(3, j) + M.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

					+ M.CoeffCationB(4, i) * (X(cat_j_ip1) - X(cat_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Ccan(j, i);
				//Potential
				F(pot_j_i) =
					M.CoeffPotentialA(1, j) * X(pot_jm1_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
					+ M.CoeffPotentialB(1, i) * X(pot_j_im1) + M.CoeffPotentialB(2, i) * X(pot_j_ip1)
					+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
					+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
						+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
						+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
			}
			else if (j == 0 && i > 1 && i < membrane.m - 1) {
				// Reactant:
				double reactionRate = X(rea_j_i)*Thermo.k0*exp(-Thermo.alfa * Thermo.F_R_T*(Signal.AppliedPotential() - Thermo.E_formal - X(pot_j_i)))
					- X(pro_j_i)*Thermo.k0*exp((Thermo.alfa - 1)*Thermo.F_R_T*(Signal.AppliedPotential() - Thermo.E_formal - X(pot_j_i)));
				F(rea_j_i) =
					M.CoeffReactantA(1, j) * X(rea_j_i) + M.CoeffReactantA(2, j) * X(rea_jp1_i)
					+ M.CoeffReactantB(1, i) * X(rea_j_im1) + M.CoeffReactantB(2, i) * X(rea_j_ip1)
					+ (M.CoeffReactantA(0, j) + M.CoeffReactantB(0, i))*X(rea_j_i)

					+ M.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffReactantA(5, j) * X(pot_jp1_i) + M.CoeffReactantA(6, j) * X(pot_j_i)
						+ M.CoeffReactantB(5, i) * X(pot_j_ip1) + M.CoeffReactantB(6, i) * X(pot_j_im1)
						+ (M.CoeffReactantA(3, j) + M.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

					+ M.CoeffReactantB(4, i) * (X(rea_j_ip1) - X(rea_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Cren(j, i)
					- reactionRate;
				// Product:
				F(pro_j_i) =
					M.CoeffProductA(1, j) * X(pro_j_i) + M.CoeffProductA(2, j) * X(pro_jp1_i)
					+ M.CoeffProductB(1, i) * X(pro_j_im1) + M.CoeffProductB(2, i) * X(pro_j_ip1)
					+ (M.CoeffProductA(0, j) + M.CoeffProductB(0, i))*X(pro_j_i)

					+ M.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffProductA(5, j) * X(pot_jp1_i) + M.CoeffProductA(6, j) * X(pot_j_i)
						+ M.CoeffProductB(5, i) * X(pot_j_ip1) + M.CoeffProductB(6, i) * X(pot_j_im1)
						+ (M.CoeffProductA(3, j) + M.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

					+ M.CoeffProductB(4, i) * (X(pro_j_ip1) - X(pro_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Cprn(j, i)
					+ reactionRate;
				// Anion
				F(ani_j_i) =
					M.CoeffAnionA(1, j) * X(ani_j_i) + M.CoeffAnionA(2, j) * X(ani_jp1_i)
					+ M.CoeffAnionB(1, i) * X(ani_j_im1) + M.CoeffAnionB(2, i) * X(ani_j_ip1)
					+ (M.CoeffAnionA(0, j) + M.CoeffAnionB(0, i))*X(ani_j_i)

					+ M.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffAnionA(5, j) * X(pot_jp1_i) + M.CoeffAnionA(6, j) * X(pot_j_i)
						+ M.CoeffAnionB(5, i) * X(pot_j_ip1) + M.CoeffAnionB(6, i) * X(pot_j_im1)
						+ (M.CoeffAnionA(3, j) + M.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

					+ M.CoeffAnionB(4, i) * (X(ani_j_ip1) - X(ani_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Cann(j, i);
				// Cation
				F(cat_j_i) =
					F(cat_j_i) =
					M.CoeffCationA(1, j) * X(cat_j_i) + M.CoeffCationA(2, j) * X(cat_jp1_i)
					+ M.CoeffCationB(1, i) * X(cat_j_im1) + M.CoeffCationB(2, i) * X(cat_j_ip1)
					+ (M.CoeffCationA(0, j) + M.CoeffCationB(0, i))*X(cat_j_i)

					+ M.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffCationA(5, j) * X(pot_jp1_i) + M.CoeffCationA(6, j) * X(pot_j_i)
						+ M.CoeffCationB(5, i) * X(pot_j_ip1) + M.CoeffCationB(6, i) * X(pot_j_im1)
						+ (M.CoeffCationA(3, j) + M.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

					+ M.CoeffCationB(4, i) * (X(cat_j_ip1) - X(cat_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
					+ membrane.Ccan(j, i);
				// Potential
				F(pot_j_i) =
					F(pot_j_i) =
					M.CoeffPotentialA(1, j) * X(pot_j_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
					+ M.CoeffPotentialB(1, i) * X(pot_j_im1) + M.CoeffPotentialB(2, i) * X(pot_j_ip1)
					+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
					+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
						+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
						+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
			}
			else if (i == 0 && j > 0 && j < membrane.n - 1) {
				//Reactant
				F(rea_j_i) =
					M.CoeffReactantA(1, j) * X(rea_jm1_i) + M.CoeffReactantA(2, j) * X(rea_jp1_i)
					+ M.CoeffReactantB(1, i) * X(rea_j_i) + M.CoeffReactantB(2, i) * X(rea_j_ip1)
					+ (M.CoeffReactantA(0, j) + M.CoeffReactantB(0, i))*X(rea_j_i)

					+ M.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffReactantA(5, j) * X(pot_jp1_i) + M.CoeffReactantA(6, j) * X(pot_jm1_i)
						+ M.CoeffReactantB(5, i) * X(pot_j_ip1) + M.CoeffReactantB(6, i) * X(pot_j_i)
						+ (M.CoeffReactantA(3, j) + M.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

					+ M.CoeffReactantB(4, i) * (X(rea_j_ip1) - X(rea_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Cren(j, i);
				//Product
				F(pro_j_i) =
					M.CoeffProductA(1, j) * X(pro_jm1_i) + M.CoeffProductA(2, j) * X(pro_jp1_i)
					+ M.CoeffProductB(1, i) * X(pro_j_i) + M.CoeffProductB(2, i) * X(pro_j_ip1)
					+ (M.CoeffProductA(0, j) + M.CoeffProductB(0, i))*X(pro_j_i)

					+ M.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffProductA(5, j) * X(pot_jp1_i) + M.CoeffProductA(6, j) * X(pot_jm1_i)
						+ M.CoeffProductB(5, i) * X(pot_j_ip1) + M.CoeffProductB(6, i) * X(pot_j_i)
						+ (M.CoeffProductA(3, j) + M.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

					+ M.CoeffProductB(4, i) * (X(pro_j_ip1) - X(pro_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Cprn(j, i);
				//Anion
				F(ani_j_i) =
					M.CoeffAnionA(1, j) * X(ani_jm1_i) + M.CoeffAnionA(2, j) * X(ani_jp1_i)
					+ M.CoeffAnionB(1, i) * X(ani_j_i) + M.CoeffAnionB(2, i) * X(ani_j_ip1)
					+ (M.CoeffAnionA(0, j) + M.CoeffAnionB(0, i))*X(ani_j_i)

					+ M.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffAnionA(5, j) * X(pot_jp1_i) + M.CoeffAnionA(6, j) * X(pot_jm1_i)
						+ M.CoeffAnionB(5, i) * X(pot_j_ip1) + M.CoeffAnionB(6, i) * X(pot_j_i)
						+ (M.CoeffAnionA(3, j) + M.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

					+ M.CoeffAnionB(4, i) * (X(ani_j_ip1) - X(ani_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Cann(j, i);
				//Cation
				F(cat_j_i) =
					M.CoeffCationA(1, j) * X(cat_jm1_i) + M.CoeffCationA(2, j) * X(cat_jp1_i)
					+ M.CoeffCationB(1, i) * X(cat_j_i) + M.CoeffCationB(2, i) * X(cat_j_ip1)
					+ (M.CoeffCationA(0, j) + M.CoeffCationB(0, i))*X(cat_j_i)

					+ M.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffCationA(5, j) * X(pot_jp1_i) + M.CoeffCationA(6, j) * X(pot_jm1_i)
						+ M.CoeffCationB(5, i) * X(pot_j_ip1) + M.CoeffCationB(6, i) * X(pot_j_i)
						+ (M.CoeffCationA(3, j) + M.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

					+ M.CoeffCationB(4, i) * (X(cat_j_ip1) - X(cat_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Ccan(j, i);
				//Potential
				F(pot_j_i) =
					M.CoeffPotentialA(1, j) * X(pot_jm1_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
					+ M.CoeffPotentialB(1, i) * X(pot_j_i) + M.CoeffPotentialB(2, i) * X(pot_j_ip1)
					+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
					+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
						+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
						+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
			}
			else if (j == membrane.n - 1 && i > 0 && i < membrane.m - 1) {

			}
			else if (i == membrane.m - 1 && j > 0 && j < membrane.n - 1) {
				//Reactant
				F(rea_j_i) =
					M.CoeffReactantA(1, j) * X(rea_jm1_i) + M.CoeffReactantA(2, j) * X(rea_jp1_i)
					+ M.CoeffReactantB(1, i) * X(rea_j_im1) + M.CoeffReactantB(2, i) * X(rea_j_i)
					+ (M.CoeffReactantA(0, j) + M.CoeffReactantB(0, i))*X(rea_j_i)

					+ M.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffReactantA(5, j) * X(pot_jp1_i) + M.CoeffReactantA(6, j) * X(pot_jm1_i)
						+ M.CoeffReactantB(5, i) * X(pot_j_i) + M.CoeffReactantB(6, i) * X(pot_j_im1)
						+ (M.CoeffReactantA(3, j) + M.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

					+ M.CoeffReactantB(4, i) * (X(rea_j_i) - X(rea_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Cren(j, i);
				//Product
				F(pro_j_i) =
					M.CoeffProductA(1, j) * X(pro_jm1_i) + M.CoeffProductA(2, j) * X(pro_jp1_i)
					+ M.CoeffProductB(1, i) * X(pro_j_im1) + M.CoeffProductB(2, i) * X(pro_j_i)
					+ (M.CoeffProductA(0, j) + M.CoeffProductB(0, i))*X(pro_j_i)

					+ M.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffProductA(5, j) * X(pot_jp1_i) + M.CoeffProductA(6, j) * X(pot_jm1_i)
						+ M.CoeffProductB(5, i) * X(pot_j_i) + M.CoeffProductB(6, i) * X(pot_j_im1)
						+ (M.CoeffProductA(3, j) + M.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

					+ M.CoeffProductB(4, i) * (X(pro_j_i) - X(pro_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Cprn(j, i);
				//Anion
				F(ani_j_i) =
					M.CoeffAnionA(1, j) * X(ani_jm1_i) + M.CoeffAnionA(2, j) * X(ani_jp1_i)
					+ M.CoeffAnionB(1, i) * X(ani_j_im1) + M.CoeffAnionB(2, i) * X(ani_j_i)
					+ (M.CoeffAnionA(0, j) + M.CoeffAnionB(0, i))*X(ani_j_i)

					+ M.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffAnionA(5, j) * X(pot_jp1_i) + M.CoeffAnionA(6, j) * X(pot_jm1_i)
						+ M.CoeffAnionB(5, i) * X(pot_j_i) + M.CoeffAnionB(6, i) * X(pot_j_im1)
						+ (M.CoeffAnionA(3, j) + M.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

					+ M.CoeffAnionB(4, i) * (X(ani_j_i) - X(ani_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Cann(j, i);
				//Cation
				F(cat_j_i) =
					M.CoeffCationA(1, j) * X(cat_jm1_i) + M.CoeffCationA(2, j) * X(cat_jp1_i)
					+ M.CoeffCationB(1, i) * X(cat_j_im1) + M.CoeffCationB(2, i) * X(cat_j_i)
					+ (M.CoeffCationA(0, j) + M.CoeffCationB(0, i))*X(cat_j_i)

					+ M.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

					+ (M.CoeffCationA(5, j) * X(pot_jp1_i) + M.CoeffCationA(6, j) * X(pot_jm1_i)
						+ M.CoeffCationB(5, i) * X(pot_j_i) + M.CoeffCationB(6, i) * X(pot_j_im1)
						+ (M.CoeffCationA(3, j) + M.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

					+ M.CoeffCationB(4, i) * (X(cat_j_i) - X(cat_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Ccan(j, i);
				//Potential
				F(pot_j_i) =
					M.CoeffPotentialA(1, j) * X(pot_jm1_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
					+ M.CoeffPotentialB(1, i) * X(pot_j_im1) + M.CoeffPotentialB(2, i) * X(pot_j_i)
					+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
					+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
						+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
						+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
			}
			else if (i == 0 && j == 0) {
				// Reactant:
				double reactionRate = X(rea_j_i)*Thermo.k0*exp(-Thermo.alfa * Thermo.F_R_T*(Signal.AppliedPotential() - Thermo.E_formal - X(pot_j_i)))
					- X(pro_j_i)*Thermo.k0*exp((Thermo.alfa - 1)*Thermo.F_R_T*(Signal.AppliedPotential() - Thermo.E_formal - X(pot_j_i)));

				F(rea_j_i) =
					M.CoeffReactantA(1, j) * X(rea_j_i) + M.CoeffReactantA(2, j) * X(rea_jp1_i)
					+ M.CoeffReactantB(1, i) * X(rea_j_i) + M.CoeffReactantB(2, i) * X(rea_j_ip1)
					+ (M.CoeffReactantA(0, j) + M.CoeffReactantB(0, i))*X(rea_j_i)

					+ M.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffReactantA(5, j) * X(pot_jp1_i) + M.CoeffReactantA(6, j) * X(pot_j_i)
						+ M.CoeffReactantB(5, i) * X(pot_j_ip1) + M.CoeffReactantB(6, i) * X(pot_j_i)
						+ (M.CoeffReactantA(3, j) + M.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

					+ M.CoeffReactantB(4, i) * (X(rea_j_ip1) - X(rea_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Cren(j, i)
					- reactionRate;
				// Product:
				F(pro_j_i) =
					M.CoeffProductA(1, j) * X(pro_j_i) + M.CoeffProductA(2, j) * X(pro_jp1_i)
					+ M.CoeffProductB(1, i) * X(pro_j_i) + M.CoeffProductB(2, i) * X(pro_j_ip1)
					+ (M.CoeffProductA(0, j) + M.CoeffProductB(0, i))*X(pro_j_i)

					+ M.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffProductA(5, j) * X(pot_jp1_i) + M.CoeffProductA(6, j) * X(pot_j_i)
						+ M.CoeffProductB(5, i) * X(pot_j_ip1) + M.CoeffProductB(6, i) * X(pot_j_i)
						+ (M.CoeffProductA(3, j) + M.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

					+ M.CoeffProductB(4, i) * (X(pro_j_ip1) - X(pro_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Cprn(j, i)
					+ reactionRate;
				// Anion
				F(ani_j_i) =
					M.CoeffAnionA(1, j) * X(ani_j_i) + M.CoeffAnionA(2, j) * X(ani_jp1_i)
					+ M.CoeffAnionB(1, i) * X(ani_j_i) + M.CoeffAnionB(2, i) * X(ani_j_ip1)
					+ (M.CoeffAnionA(0, j) + M.CoeffAnionB(0, i))*X(ani_j_i)

					+ M.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffAnionA(5, j) * X(pot_jp1_i) + M.CoeffAnionA(6, j) * X(pot_j_i)
						+ M.CoeffAnionB(5, i) * X(pot_j_ip1) + M.CoeffAnionB(6, i) * X(pot_j_i)
						+ (M.CoeffAnionA(3, j) + M.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

					+ M.CoeffAnionB(4, i) * (X(ani_j_ip1) - X(ani_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Cann(j, i);
				// Cation
				F(cat_j_i) =
					F(cat_j_i) =
					M.CoeffCationA(1, j) * X(cat_j_i) + M.CoeffCationA(2, j) * X(cat_jp1_i)
					+ M.CoeffCationB(1, i) * X(cat_j_i) + M.CoeffCationB(2, i) * X(cat_j_ip1)
					+ (M.CoeffCationA(0, j) + M.CoeffCationB(0, i))*X(cat_j_i)

					+ M.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffCationA(5, j) * X(pot_jp1_i) + M.CoeffCationA(6, j) * X(pot_j_i)
						+ M.CoeffCationB(5, i) * X(pot_j_ip1) + M.CoeffCationB(6, i) * X(pot_j_i)
						+ (M.CoeffCationA(3, j) + M.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

					+ M.CoeffCationB(4, i) * (X(cat_j_ip1) - X(cat_j_i))*(X(pot_j_ip1) - X(pot_j_i))
					+ membrane.Ccan(j, i);
				// Potential
				F(pot_j_i) =
					F(pot_j_i) =
					M.CoeffPotentialA(1, j) * X(pot_j_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
					+ M.CoeffPotentialB(1, i) * X(pot_j_i) + M.CoeffPotentialB(2, i) * X(pot_j_ip1)
					+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
					+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
						+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
						+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
			}
			else if (i == membrane.m - 1 && j == 0) {
				// Reactant:
				double reactionRate = X(rea_j_i)*Thermo.k0*exp(-Thermo.alfa * Thermo.F_R_T*(Signal.AppliedPotential() - Thermo.E_formal - X(pot_j_i)))
					- X(pro_j_i)*Thermo.k0*exp((Thermo.alfa - 1)*Thermo.F_R_T*(Signal.AppliedPotential() - Thermo.E_formal - X(pot_j_i)));
				F(rea_j_i) =
					M.CoeffReactantA(1, j) * X(rea_j_i) + M.CoeffReactantA(2, j) * X(rea_jp1_i)
					+ M.CoeffReactantB(1, i) * X(rea_j_im1) + M.CoeffReactantB(2, i) * X(rea_j_i)
					+ (M.CoeffReactantA(0, j) + M.CoeffReactantB(0, i))*X(rea_j_i)

					+ M.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffReactantA(5, j) * X(pot_jp1_i) + M.CoeffReactantA(6, j) * X(pot_j_i)
						+ M.CoeffReactantB(5, i) * X(pot_j_i) + M.CoeffReactantB(6, i) * X(pot_j_im1)
						+ (M.CoeffReactantA(3, j) + M.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

					+ M.CoeffReactantB(4, i) * (X(rea_j_i) - X(rea_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Cren(j, i)
					- reactionRate;
				// Product:
				F(pro_j_i) =
					M.CoeffProductA(1, j) * X(pro_j_i) + M.CoeffProductA(2, j) * X(pro_jp1_i)
					+ M.CoeffProductB(1, i) * X(pro_j_im1) + M.CoeffProductB(2, i) * X(pro_j_i)
					+ (M.CoeffProductA(0, j) + M.CoeffProductB(0, i))*X(pro_j_i)

					+ M.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffProductA(5, j) * X(pot_jp1_i) + M.CoeffProductA(6, j) * X(pot_j_i)
						+ M.CoeffProductB(5, i) * X(pot_j_i) + M.CoeffProductB(6, i) * X(pot_j_im1)
						+ (M.CoeffProductA(3, j) + M.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

					+ M.CoeffProductB(4, i) * (X(pro_j_i) - X(pro_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Cprn(j, i)
					+ reactionRate;
				// Anion
				F(ani_j_i) =
					M.CoeffAnionA(1, j) * X(ani_j_i) + M.CoeffAnionA(2, j) * X(ani_jp1_i)
					+ M.CoeffAnionB(1, i) * X(ani_j_im1) + M.CoeffAnionB(2, i) * X(ani_j_i)
					+ (M.CoeffAnionA(0, j) + M.CoeffAnionB(0, i))*X(ani_j_i)

					+ M.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffAnionA(5, j) * X(pot_jp1_i) + M.CoeffAnionA(6, j) * X(pot_j_i)
						+ M.CoeffAnionB(5, i) * X(pot_j_i) + M.CoeffAnionB(6, i) * X(pot_j_im1)
						+ (M.CoeffAnionA(3, j) + M.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

					+ M.CoeffAnionB(4, i) * (X(ani_j_i) - X(ani_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Cann(j, i);
				// Cation
				F(cat_j_i) =
					F(cat_j_i) =
					M.CoeffCationA(1, j) * X(cat_j_i) + M.CoeffCationA(2, j) * X(cat_jp1_i)
					+ M.CoeffCationB(1, i) * X(cat_j_im1) + M.CoeffCationB(2, i) * X(cat_j_i)
					+ (M.CoeffCationA(0, j) + M.CoeffCationB(0, i))*X(cat_j_i)

					+ M.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_j_i))*(X(pot_jp1_i) - X(pot_j_i))

					+ (M.CoeffCationA(5, j) * X(pot_jp1_i) + M.CoeffCationA(6, j) * X(pot_j_i)
						+ M.CoeffCationB(5, i) * X(pot_j_i) + M.CoeffCationB(6, i) * X(pot_j_im1)
						+ (M.CoeffCationA(3, j) + M.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

					+ M.CoeffCationB(4, i) * (X(cat_j_i) - X(cat_j_im1))*(X(pot_j_i) - X(pot_j_im1))
					+ membrane.Ccan(j, i);
				// Potential
				F(pot_j_i) =
					F(pot_j_i) =
					M.CoeffPotentialA(1, j) * X(pot_j_i) + M.CoeffPotentialA(2, j) * X(pot_jp1_i)
					+ M.CoeffPotentialB(1, i) * X(pot_j_im1) + M.CoeffPotentialB(2, i) * X(pot_j_i)
					+ (M.CoeffPotentialA(0, j) + M.CoeffPotentialB(0, i)) * X(pot_j_i)
					+ (MI.Reactant.Z*X(rea_j_i) + MI.Product.Z*X(pro_j_i)
						+ MI.SupportAnion.Z*X(ani_j_i) + MI.SupportCation.Z*X(cat_j_i)
						+ MI.CxZImmobileCharge)*MI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
			}
		}
	}

	mxn = solution.Getmxn();
	// Solution bulk
	for (unsigned long i = 1; i < solution.m - 1; ++i) {
		for (unsigned long j = 1; j < solution.n - 1; ++j) {

			// Reactant index
			rea_j_i = solution.n*i + j;
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

			//Reactant
			F(rea_j_i) =
				S.CoeffReactantA(1, j) * X(rea_jm1_i) + S.CoeffReactantA(2, j) * X(rea_jp1_i)
				+ S.CoeffReactantB(1, i) * X(rea_j_im1) + S.CoeffReactantB(2, i) * X(rea_j_ip1)
				+ (S.CoeffReactantA(0, j) + S.CoeffReactantB(0, i))*X(rea_j_i)

				+ S.CoeffReactantA(4, j)*(X(rea_jp1_i) - X(rea_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

				+ (S.CoeffReactantA(5, j) * X(pot_jp1_i) + S.CoeffReactantA(6, j) * X(pot_jm1_i)
					+ S.CoeffReactantB(5, i) * X(pot_j_ip1) + S.CoeffReactantB(6, i) * X(pot_j_im1)
					+ (S.CoeffReactantA(3, j) + S.CoeffReactantB(3, i))*X(pot_j_i))*X(rea_j_i)

				+ S.CoeffReactantB(4, i) * (X(rea_j_ip1) - X(rea_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
				+ solution.Cren(j, i);
			//Product
			F(pro_j_i) =
				S.CoeffProductA(1, j) * X(pro_jm1_i) + S.CoeffProductA(2, j) * X(pro_jp1_i)
				+ S.CoeffProductB(1, i) * X(pro_j_im1) + S.CoeffProductB(2, i) * X(pro_j_ip1)
				+ (S.CoeffProductA(0, j) + S.CoeffProductB(0, i))*X(pro_j_i)

				+ S.CoeffProductA(4, j)*(X(pro_jp1_i) - X(pro_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

				+ (S.CoeffProductA(5, j) * X(pot_jp1_i) + S.CoeffProductA(6, j) * X(pot_jm1_i)
					+ S.CoeffProductB(5, i) * X(pot_j_ip1) + S.CoeffProductB(6, i) * X(pot_j_im1)
					+ (S.CoeffProductA(3, j) + S.CoeffProductB(3, i))*X(pot_j_i))*X(pro_j_i)

				+ S.CoeffProductB(4, i) * (X(pro_j_ip1) - X(pro_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
				+ solution.Cprn(j, i);
			//Anion
			F(ani_j_i) =
				S.CoeffAnionA(1, j) * X(ani_jm1_i) + S.CoeffAnionA(2, j) * X(ani_jp1_i)
				+ S.CoeffAnionB(1, i) * X(ani_j_im1) + S.CoeffAnionB(2, i) * X(ani_j_ip1)
				+ (S.CoeffAnionA(0, j) + S.CoeffAnionB(0, i))*X(ani_j_i)

				+ S.CoeffAnionA(4, j)*(X(ani_jp1_i) - X(ani_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

				+ (S.CoeffAnionA(5, j) * X(pot_jp1_i) + S.CoeffAnionA(6, j) * X(pot_jm1_i)
					+ S.CoeffAnionB(5, i) * X(pot_j_ip1) + S.CoeffAnionB(6, i) * X(pot_j_im1)
					+ (S.CoeffAnionA(3, j) + S.CoeffAnionB(3, i))*X(pot_j_i))*X(ani_j_i)

				+ S.CoeffAnionB(4, i) * (X(ani_j_ip1) - X(ani_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
				+ solution.Cann(j, i);
			//Cation
			F(cat_j_i) =
				S.CoeffCationA(1, j) * X(cat_jm1_i) + S.CoeffCationA(2, j) * X(cat_jp1_i)
				+ S.CoeffCationB(1, i) * X(cat_j_im1) + S.CoeffCationB(2, i) * X(cat_j_ip1)
				+ (S.CoeffCationA(0, j) + S.CoeffCationB(0, i))*X(cat_j_i)

				+ S.CoeffCationA(4, j)*(X(cat_jp1_i) - X(cat_jm1_i))*(X(pot_jp1_i) - X(pot_jm1_i))

				+ (S.CoeffCationA(5, j) * X(pot_jp1_i) + S.CoeffCationA(6, j) * X(pot_jm1_i)
					+ S.CoeffCationB(5, i) * X(pot_j_ip1) + S.CoeffCationB(6, i) * X(pot_j_im1)
					+ (S.CoeffCationA(3, j) + S.CoeffCationB(3, i))*X(pot_j_i))*X(cat_j_i)

				+ S.CoeffCationB(4, i) * (X(cat_j_ip1) - X(cat_j_im1))*(X(pot_j_ip1) - X(pot_j_im1))
				+ solution.Ccan(j, i);
			//Potential
			F(pot_j_i) =
				S.CoeffPotentialA(1, j) * X(pot_jm1_i) + S.CoeffPotentialA(2, j) * X(pot_jp1_i)
				+ S.CoeffPotentialB(1, i) * X(pot_j_im1) + S.CoeffPotentialB(2, i) * X(pot_j_ip1)
				+ (S.CoeffPotentialA(0, j) + S.CoeffPotentialB(0, i)) * X(pot_j_i)
				+ (SI.Reactant.Z*X(rea_j_i) + SI.Product.Z*X(pro_j_i)
					+ SI.SupportAnion.Z*X(ani_j_i) + SI.SupportCation.Z*X(cat_j_i)
					+ SI.CxZImmobileCharge)*SI.ReciprocalEpsilon_rEpsilon_0*Thermo.F;
		}
	}
}
EquationCoefficient::EquationCoefficient(const IonSystem& fIons, const mesh& phase, const PotentialSignal& fSignal, const nernst_equation& fThermo) :
	Ions(fIons), Signal(fSignal), F_R_T(fThermo.F_R_T),
	CoeffProductA(7, phase.GetMeshSize()[1]), CoeffProductB(7, phase.GetMeshSize()[0]),
	CoeffReactantA(7, phase.GetMeshSize[1]), CoeffReactantB(7, phase.GetMeshSize()[0]),
	CoeffAnionA(7, phase.GetMeshSize()[1]), CoeffAnionB(7, phase.GetMeshSize()[0]),
	CoeffCationA(7, phase.GetMeshSize()[1]), CoeffCationB(7, phase.GetMeshSize()[0]),
	CoeffPotentialA(3, phase.GetMeshSize()[1]), CoeffPotentialB(3, phase.GetMeshSize[0])
{}

void EquationCoefficient::CalculateCoeff(Eigen::MatrixXd& GeoCoeffA, Eigen::MatrixXd GeoCoeffB)
{
	//initialise coefficients from geometric coefficients
	CoeffProductA = GeoCoeffA;
	CoeffProductB = GeoCoeffB;
	CoeffReactantA = GeoCoeffA;
	CoeffReactantB = GeoCoeffB;
	CoeffAnionA = GeoCoeffA;
	CoeffAnionB= GeoCoeffB;
	CoeffCationA = GeoCoeffA;
	CoeffCationB = GeoCoeffB;

	CoeffPotentialA = GeoCoeffA.topRows(3);
	CoeffPotentialB = GeoCoeffB.topRows(3);
	// adjust the coefficients of diffusion components
	CoeffProductA.topRows(3) *= Ions.Product.D*Signal.dt;
	CoeffProductB.topRows(3) *= Ions.Product.D*Signal.dt;
	CoeffReactantA.topRows(3) *= Ions.Reactant.D*Signal.dt;
	CoeffReactantB.topRows(3) *= Ions.Reactant.D*Signal.dt;
	CoeffAnionA.topRows(3) *= Ions.SupportAnion.D*Signal.dt;
	CoeffAnionB.topRows(3) *= Ions.SupportAnion.D*Signal.dt;
	CoeffCationA.topRows(3) *= Ions.SupportCation.D*Signal.dt;
	CoeffCationB.topRows(3) *= Ions.SupportCation.D*Signal.dt;

	CoeffProductA.row(0).array() += -1;
	CoeffReactantA.row(0).array() += -1;
	CoeffAnionA.row(0).array() += -1;
	CoeffCationA.row(0).array() += -1;
	// adjust the coefficients of migration components
	CoeffProductA.bottomRows(4) *= Ions.Product.D*Signal.dt*Ions.Product.Z*F_R_T;
	CoeffProductB.bottomRows(4) *= Ions.Product.D*Signal.dt*Ions.Product.Z*F_R_T;
	CoeffReactantA.bottomRows(4) *= Ions.Reactant.D*Signal.dt*Ions.Reactant.Z*F_R_T;
	CoeffReactantB.bottomRows(4) *= Ions.Reactant.D*Signal.dt*Ions.Reactant.Z*F_R_T;
	CoeffAnionA.bottomRows(4) *= Ions.SupportAnion.D*Signal.dt*Ions.SupportAnion.Z*F_R_T;
	CoeffAnionB.bottomRows(4) *= Ions.SupportAnion.D*Signal.dt*Ions.SupportAnion.Z*F_R_T;
	CoeffCationA.bottomRows(4) *= Ions.SupportCation.D*Signal.dt*Ions.SupportCation.Z*F_R_T;
	CoeffCationB.bottomRows(4) *= Ions.SupportCation.D*Signal.dt*Ions.SupportCation.Z*F_R_T;
}

