#include "EquationCoefficient.h"

EquationCoefficient::EquationCoefficient(const IonSystem& fIons, const mesh& phase, const PotentialSignal& fSignal, const nernst_equation& fThermo) :
	Ions(fIons), Signal(fSignal), F_R_T(fThermo.F_R_T),
	CoeffProductA(7, phase.GetMeshSize()[1]), CoeffProductB(7, phase.GetMeshSize()[0]),
	CoeffReactantA(7, phase.GetMeshSize()[1]), CoeffReactantB(7, phase.GetMeshSize()[0]),
	CoeffAnionA(7, phase.GetMeshSize()[1]), CoeffAnionB(7, phase.GetMeshSize()[0]),
	CoeffCationA(7, phase.GetMeshSize()[1]), CoeffCationB(7, phase.GetMeshSize()[0]),
	CoeffPotentialA(3, phase.GetMeshSize()[1]), CoeffPotentialB(3, phase.GetMeshSize()[0])
{}

void EquationCoefficient::CalculateCoeff(Eigen::MatrixXd& GeoCoeffA, Eigen::MatrixXd GeoCoeffB)
{
	//initialise coefficients from geometric coefficients
	CoeffProductA = GeoCoeffA;
	CoeffProductB = GeoCoeffB;
	CoeffReactantA = GeoCoeffA;
	CoeffReactantB = GeoCoeffB;
	CoeffAnionA = GeoCoeffA;
	CoeffAnionB = GeoCoeffB;
	CoeffCationA = GeoCoeffA;
	CoeffCationB = GeoCoeffB;

	CoeffPotentialA = GeoCoeffA.topRows(3);
	CoeffPotentialB = GeoCoeffB.topRows(3);
	if (Ions.IonNum == 3) {
		// adjust the coefficients of diffusion components
		CoeffProductA.topRows(3) *= Ions[SpeciesEnum::mProduct].D*Signal.dt;
		CoeffProductB.topRows(3) *= Ions[SpeciesEnum::mProduct].D*Signal.dt;
		CoeffReactantA.topRows(3) *= Ions[SpeciesEnum::mReactant].D*Signal.dt;
		CoeffReactantB.topRows(3) *= Ions[SpeciesEnum::mReactant].D*Signal.dt;
		CoeffCationA.topRows(3) *= Ions[SpeciesEnum::mCation].D*Signal.dt;
		CoeffCationB.topRows(3) *= Ions[SpeciesEnum::mCation].D*Signal.dt;

		CoeffProductA.row(0).array() += -1;
		CoeffReactantA.row(0).array() += -1;
		CoeffCationA.row(0).array() += -1;
		// adjust the coefficients of migration components
		CoeffProductA.bottomRows(4) *= Ions[SpeciesEnum::mProduct].D*Signal.dt*Ions[SpeciesEnum::mProduct].Z*F_R_T;
		CoeffProductB.bottomRows(4) *= Ions[SpeciesEnum::mProduct].D*Signal.dt*Ions[SpeciesEnum::mProduct].Z*F_R_T;
		CoeffReactantA.bottomRows(4) *= Ions[SpeciesEnum::mReactant].D*Signal.dt*Ions[SpeciesEnum::mReactant].Z*F_R_T;
		CoeffReactantB.bottomRows(4) *= Ions[SpeciesEnum::mReactant].D*Signal.dt*Ions[SpeciesEnum::mReactant].Z*F_R_T;
		CoeffCationA.bottomRows(4) *= Ions[SpeciesEnum::mCation].D*Signal.dt*Ions[SpeciesEnum::mCation].Z*F_R_T;
		CoeffCationB.bottomRows(4) *= Ions[SpeciesEnum::mCation].D*Signal.dt*Ions[SpeciesEnum::mCation].Z*F_R_T;
	}
	else if (Ions.IonNum == 4) {
		// adjust the coefficients of diffusion components
		CoeffProductA.topRows(3) *= Ions[SpeciesEnum::sProduct].D*Signal.dt;
		CoeffProductB.topRows(3) *= Ions[SpeciesEnum::sProduct].D*Signal.dt;
		CoeffReactantA.topRows(3) *= Ions[SpeciesEnum::sReactant].D*Signal.dt;
		CoeffReactantB.topRows(3) *= Ions[SpeciesEnum::sReactant].D*Signal.dt;
		CoeffAnionA.topRows(3) *= Ions[SpeciesEnum::sAnion].D*Signal.dt;
		CoeffAnionB.topRows(3) *= Ions[SpeciesEnum::sAnion].D*Signal.dt;
		CoeffCationA.topRows(3) *= Ions[SpeciesEnum::sCation].D*Signal.dt;
		CoeffCationB.topRows(3) *= Ions[SpeciesEnum::sCation].D*Signal.dt;

		CoeffProductA.row(0).array() += -1;
		CoeffReactantA.row(0).array() += -1;
		CoeffAnionA.row(0).array() += -1;
		CoeffCationA.row(0).array() += -1;
		// adjust the coefficients of migration components
		CoeffProductA.bottomRows(4) *= Ions[SpeciesEnum::sProduct].D*Signal.dt*Ions[SpeciesEnum::sProduct].Z*F_R_T;
		CoeffProductB.bottomRows(4) *= Ions[SpeciesEnum::sProduct].D*Signal.dt*Ions[SpeciesEnum::sProduct].Z*F_R_T;
		CoeffReactantA.bottomRows(4) *= Ions[SpeciesEnum::sReactant].D*Signal.dt*Ions[SpeciesEnum::sReactant].Z*F_R_T;
		CoeffReactantB.bottomRows(4) *= Ions[SpeciesEnum::sReactant].D*Signal.dt*Ions[SpeciesEnum::sReactant].Z*F_R_T;
		CoeffAnionA.bottomRows(4) *= Ions[SpeciesEnum::sAnion].D*Signal.dt*Ions[SpeciesEnum::sAnion].Z*F_R_T;
		CoeffAnionB.bottomRows(4) *= Ions[SpeciesEnum::sAnion].D*Signal.dt*Ions[SpeciesEnum::sAnion].Z*F_R_T;
		CoeffCationA.bottomRows(4) *= Ions[SpeciesEnum::sCation].D*Signal.dt*Ions[SpeciesEnum::sCation].Z*F_R_T;
		CoeffCationB.bottomRows(4) *= Ions[SpeciesEnum::sCation].D*Signal.dt*Ions[SpeciesEnum::sCation].Z*F_R_T;
	}
}
