#ifndef THERMODYNAMICS_H_INCLUDED
#define THERMODYNAMICS_H_INCLUDED

class nernst_equation{
	friend class ElectrodeReaction;
	friend class InterfaceReaction;
public:
	nernst_equation(double fE_formal, int fn, double k0, double alfa);
	~nernst_equation(){};
	
	const double F; // Faraday constant, C mol^-1
	const double alfa; // reduction charge transfer coefficient
	const int n; // the number of electron transfered
	const double E_formal; // standard potential, V
	const double F_R_T; // F/(RT)
	const double nF_R_T; // nF/(RT)
	double ratio_ox2red(double E); 	// calculate the Red concentration under thermodynamic equilibarium
									// E: the potential subjected to the reaction
									// C_total: the total concentration of the redox pair
	double DrivingPotential(double K); // return the driving potential of the reaction
									   // K: Cox/Cred

private:
	
	const double R; // gas constant, J K^-1 mol^-1
	const double T; // temperature, K
	double phi;
	double ReciprocalThermConst;
	const double minusAlfaNF_R_T; // -alfa*nF/(RT)
	const double AlfaMinusOneNF_R_T; // (1-alfa)*nF/(RT)
	const double k0; // standard rate constant
};

#endif

class ElectrodeReaction
{
public:
	ElectrodeReaction(double fEpsilon_d, double fEpsilon_oc, double fEpsilon_ic, double fmu_i, double fmu, nernst_equation& fInnerThermo);
	double DrivingPotential(double gradPhi_OHP) const { return DrivingPotentialCoeff*gradPhi_OHP - InnerThermo.E_formal; }
	double kf(double drivingPotnetial) const { return InnerThermo.k0*exp(InnerThermo.minusAlfaNF_R_T*drivingPotnetial); }; // forward reaction rate
	double kb(double drivingPotential) const { return InnerThermo.k0*exp(InnerThermo.AlfaMinusOneNF_R_T*drivingPotential); }; // backward reaction rate

private:
	const double Epsilon_d; // the effective dielectric constants of the diffuse double-layer
	const double Epsilon_oc; // the effective dielectric constants of the outer part of electric double-layer
	const double Epsilon_ic; // the effective dielectric onstants of the inner part of electric double layer
	const double mu_i; // the thickness of the inner part of the electric double-layer
	const double mu; // the thickness of the electric double-layer

	const double DrivingPotentialCoeff; // Electrode potential minus Phi_OHP

	const nernst_equation& InnerThermo;
};

class InterfaceReaction
{

public:
	InterfaceReaction(nernst_equation& fInnerThermo);
	double kf(double dE); // dE is the potential difference between the membrane and the solution phase
	double kb(double dE);
private:

	nernst_equation& InnerThermo
};