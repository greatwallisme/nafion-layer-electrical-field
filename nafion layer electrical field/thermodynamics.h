#ifndef THERMODYNAMICS_H_INCLUDED
#define THERMODYNAMICS_H_INCLUDED

class nernst_equation{
public:
	nernst_equation(double fE_formal, int fn, double k0, double alfa);
	~nernst_equation(){};
	
	const double F; // Faraday constant, C mol^-1
	const double alfa; // reduction charge transfer coefficient
	const double k0; // standard rate constant
	const int n; // the number of electron transfered
	const double E_formal; // standard potential, V
	const double F_R_T; // F/(RT)
	const double nF_R_T; // nF/(RT)

	// calculate the Red concentration under thermodynamic equilibarium
	// E: the potential subjected to the reaction
	// C_total: the total concentration of the redox pair
	double ratio_ox2red(double E); 
	double DrivingPotential(double K); // return the driving potential of the reaction
									   // K: Cox/Cred

private:
	
	const double R; // gas constant, J K^-1 mol^-1
	const double T; // temperature, K
	double phi;
	double ReciprocalThermConst;
};

#endif