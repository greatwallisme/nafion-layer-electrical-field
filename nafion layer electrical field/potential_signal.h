#pragma once
#include "mesh.h"

class PotentialSignal
{
public:
	PotentialSignal(double fE0, double fEend, double fdE, double fdt):
		E0(fE0), Eend(fEend), dE(fdE), dt(fdt){}
	virtual ~PotentialSignal() {};
	virtual double AppliedPotential() const = 0;
	virtual void RecordData() const = 0;
	virtual void SavePeakConcentration() const = 0;
	virtual void SaveCurrent() const = 0;
	const double dt; // delta time between each time node, s

protected:
	const double E0; // starting potential, V
	const double Eend; // end potential, V
	const double dE; // step potential, V
	
};

class SquareWave : public PotentialSignal
{
public:
	SquareWave(double fE0, double fEend, double fdE, double fdt, int fswf, double fswamp);
	virtual ~SquareWave();
	virtual double AppliedPotential(int i);
	virtual void RecordData();
	virtual void SavePeakConcentration(mesh& membrane, mesh& solution) const;
	virtual void SaveCurrent() const;

private:
	const int swf; // the frequency of square wave potential, Hz
	const double swamp; // the amplitude of square wave potential, V
	const unsigned long q; // number of time nodes
	double Eqm; // the base potential
	double* Er; // recorded electrode potential container
	double* tr; // recorded time container
	double* Is; // recorded square wave current container
	double* Io; // recorded oxidization current container
	double* lastIs; // recorded square wave current container of last period
	double* lastIo; // recorded oxidization current container of last period
 
	int PeriodCounter; // count the number of period
	int WaveJudge; // determine which half wave the past time is in
	int WaveJudge1; // determine which half wave the comming time node is in
	double Iqr[2] = { 0, 0 }; // record the current at the  end of each half period
};
