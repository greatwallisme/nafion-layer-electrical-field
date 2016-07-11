#ifndef THERMODYNAMICS_H_INCLUDED
#define THERMODYNAMICS_H_INCLUDED

class nernst_equation{
public:
	nernst_equation(double fE_formal, int fn, double k0, double alfa);
	~nernst_equation(){};
	
	double GetF_R_T() const { return F / R / T; } // return F/R/T
	double GetF() const { return F; }
	double GetE_formal() const { return E_formal; }
	double Getk0() const { return k0; }
	double GetAlfa() const{ return alfa; }

	// calculate the Red concentration under thermodynamic equilibarium
	// E: the potential subjected to the reaction
	// C_total: the total concentration of the redox pair
	double ratio_ox2red(double E); 
	double DrivingPotential(double K); // return the driving potential of the reaction
									   // K: Cox/Cred

private:
	const int n; // the number of electron transfered
	const double E_formal; // standard potential, V
	const double R; // gas constant, J K^-1 mol^-1
	const double T; // temperature, K
	const double F; // Faraday constant, C mol^-1

	const double k0; // standard rate constant
	const double alfa; // reduction charge transfer coefficient

	double phi;
	double thermConst; // nF/RT, V^-1
	double ReciprocalThermConst;
};

#endif