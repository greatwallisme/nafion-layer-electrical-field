#include "thermodynamics.h"
#include <cmath>

nernst_equation::nernst_equation(double fE_formal, int fn, double fk0, double falfa):
E_formal(fE_formal), R(8.3144621), T(293), F(9.64853399 * 10000), n(fn), k0(fk0), alfa(falfa), F_R_T(F/R/T), nF_R_T(fn*F_R_T),
minusAlfaNF_R_T(-alfa*nF_R_T), AlfaMinusOneNF_R_T((alfa-1)*nF_R_T)
{
	ReciprocalThermConst = 1 / nF_R_T;
}

double nernst_equation::ratio_ox2red(double E){
	double k = exp(nF_R_T*(E - E_formal));
	// return the Cox/Cred ratio
	return k;
	
}

double nernst_equation::DrivingPotential(double K)
{
	return ReciprocalThermConst*log(K) + E_formal;
}

ElectrodeReaction::ElectrodeReaction(double fEpsilon_d, double fEpsilon_oc, double fEpsilon_ic, double fmu_i, double fmu, nernst_equation& fInnerThermo) :
	Epsilon_d(fEpsilon_d), Epsilon_oc(fEpsilon_oc), Epsilon_ic(fEpsilon_ic), mu_i(fmu_i), mu(fmu), InnerThermo(fInnerThermo),
	DrivingPotentialCoeff(-(fEpsilon_d / fEpsilon_oc*(fmu - fmu_i) + fEpsilon_d / fEpsilon_ic*fmu)), nF_R_T(fInnerThermo.nF_R_T),
	minusAlfaNF_R_T(fInnerThermo.AlfaMinusOneNF_R_T), AlfaMinusOneNF_R_T(fInnerThermo.AlfaMinusOneNF_R_T), E_formal(fInnerThermo.E_formal)
{
}

InterfaceReaction::InterfaceReaction(nernst_equation& fInnerThermo) :
	InnerThermo(fInnerThermo), nF_R_T(fInnerThermo.nF_R_T),
	minusAlfaNF_R_T(fInnerThermo.AlfaMinusOneNF_R_T), AlfaMinusOneNF_R_T(fInnerThermo.AlfaMinusOneNF_R_T)
{

}