#include "mesh.h"
#include "potential_signal.h"
#include "IonSystem.h"
#include "solver.h"
#include "thermodynamics.h"
#include <iostream>
#include <fstream>
#include <map>

int main()
{
	/*
	The units used in this program:
	length: cm
	time: s
	mass amount: mol
	concentration: mol/L
	*/

	mesh membrane(10, 50e-4, 100, 0.1e-4);
	mesh solution(20, 50e-4, 25, 40e-4);

	std::map<SpeciesEnum::Species, Ion*> mIons, sIons;

	// Meaning of arguments for Ion constructor:
	// Diffusion coefficient, number of charges, density, phase

	nernst_equation ElecNE(0, 1, 1, 0.5);
	nernst_equation ProductTransNE(0, 1, 1, 0.5);
	nernst_equation CationTransNE(0, 1, -0.1, 0.5);
	nernst_equation ReactantTransNE(0, 0, 1, 0.5);

	Ion mReactant(1e-9, 0, 0.1, membrane); mIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::mReactant, &mReactant));
	Ion mProduct(1e-9, 1, 0.1*ElecNE.ratio_ox2red(-0.5), membrane); mIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::mProduct, &mProduct));
	Ion mCation(1e-9, 1, 0.1 - 0.1*ElecNE.ratio_ox2red(-0.5), membrane); mIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::mCation, &mCation));
	Ion mPotential(0, 0, 0, membrane);
	Ion ImmobileCharge(0, -1, 0.1, membrane);

	Ion sReactant(1e-5, 0, 0, solution); sIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::sReactant, &sReactant));
	Ion sProduct(1e-5, 1, 0, solution); sIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::sProduct, &sProduct));
	Ion sCation(1e-5, 1, 0.01, solution); sIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::sCation, &sCation));
	Ion sAnion(1e-5, -1, 0.01, solution); sIons.insert(std::pair<SpeciesEnum::Species, Ion*>(SpeciesEnum::sAnion, &sAnion));
	Ion sPotential(0, 0, 0, solution);

	IonSystem mIonSys(12.0, 1.0, mIons, mPotential, ImmobileCharge); // membrane phase has no Anion, using Cation to replace the argument
	IonSystem sIonSys(80, 1, sIons, sPotential);

	ElectrodeReaction ElecR(12, 10, 5, 3e-7, 10e-7, ElecNE);
	InterfaceReaction ProductIR(ProductTransNE);
	InterfaceReaction CationIR(CationTransNE);
	InterfaceReaction ReactantIR(ReactantTransNE);

	SquareWave ESignal(-0.5, 0.3, 0.002, 0.001, 25, 0.02);
	
	solver Solver(membrane, solution, mIonSys, sIonSys, ESignal, ElecNE, ElecR, CationIR, ProductIR, ReactantIR);

	Solver.initialise();
	double I; // record current
	for (long i = 0; i < ESignal.GetPeriodNumber(); ++i) {
		ESignal.CalculateAppliedPotential(i);
		Solver.solve();
		I = Solver.FaradaicCurrent();
		ESignal.RecordCurrent(I);

		if (ESignal.IsPeak() == true) {
			mReactant.PrintDense("mReactant Concentration.txt");
			mProduct.PrintDense("mProduct Concentration.txt");
			mCation.PrintDense("mCation Concentration.txt");
			mPotential.PrintDense("mPotential Distribution.txt");
			sReactant.PrintDense("sReactant Concentration.txt");
			sProduct.PrintDense("sProduct Concentration.txt");
			sCation.PrintDense("sCation Concentration.txt");
			sAnion.PrintDense("sAnion Concentration.txt");
			sPotential.PrintDense("sPotential Distribution.txt");
		}
	}
	ESignal.ExportCurrent();
	
	std::cout << "\nComplete. Enter Any Key To Terminate";
	std::cin.get();
}