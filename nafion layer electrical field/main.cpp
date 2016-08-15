#include "mesh.h"
#include "potential_signal.h"
#include "IonSystem.h"
#include "solver.h"
#include "thermodynamics.h"
#include <iostream>

int main()
{
	/*
	The units used in this program:
	length: cm
	time: s
	mass amount: mol
	*/

	mesh membrane(50, 10e-4, 200, 0.05e-4);
	mesh solution(100, 10e-4, 250, 4e-4);

	Ion mReactant(1e-9, 0, 0.1/1000, membrane);
	Ion mProduct(1e-9, 1, 0, membrane);
	Ion mCation(1e-9, 1, 1/1000, membrane);
	Ion mPotential(0, 0, 0, membrane);
	Ion ImmobileCharge(0, -1, 1/1000, membrane);

	Ion sReactant(1e-5, 0, 1, solution);
	Ion sProduct(1e-5, 1, 0, solution);
	Ion sCation(1e-5, 1, 1, solution);
	Ion sAnion(1e-5, 1, 1, solution);
	Ion sPotential(0, 0, 0, solution);

	IonSystem mIons(12, 1, mReactant, mProduct, mCation, mCation, mPotential, ImmobileCharge); // membrane phase has no Anion, using Cation to replace the argument
	IonSystem sIons(80, 1, sReactant, sProduct, sCation, sAnion, sPotential);
	
	nernst_equation ElecNE(0, 1, 5000, 0.5);
	nernst_equation ProductTransNE(0, 1, 10000, 0.5);
	nernst_equation CationTransNE(0, 1, 10000, 0.5);
	nernst_equation ReactantTransNE(0, 0, 10000, 0.5);

	ElectrodeReaction ElecR(12, 10, 5, 3e-7, 10e-7, ElecNE);
	InterfaceReaction ProductIR(ProductTransNE);
	InterfaceReaction CationIR(CationTransNE);
	InterfaceReaction ReactantIR(ReactantTransNE);

	SquareWave ESignal(-0.5, 0.3, 0.002, 0.0001, 25, 0.02);

	solver Solver(membrane, solution, mIons, sIons, ESignal, ElecNE, ElecR, CationIR, ProductIR, ReactantIR);

	Solver.initialise();
	//OneDIndex Index1d(membrane, solution);
	//TwoDIndex Index2d(membrane, solution);


	for (long i = 0; i < ESignal.GetPeriodNumber(); ++i) {
		ESignal.AppliedPotential(i);
		double I = i*0.01;
		ESignal.RecordCurrent(I);
	}
	ESignal.ExportCurrent();

	std::cin.get();
}