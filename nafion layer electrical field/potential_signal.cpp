#include "potential_signal.h"
#include <cmath>
#include <iostream>
#include <fstream>
SquareWave::SquareWave(double fE0, double fEend, double fdE, double fdt, int fswf, double fswamp) :
	PotentialSignal(fE0, fEend, fdE, fdt), swf(fswf), swamp(fswamp), q(static_cast<long>(ceil((fEend -fE0 + fdE) / (fswf*fdE*fdt))) + 1), PeriodCounter(0)
{
	Eqm = E0;
	long RcdSize = long(ceil(q*dt*swf) + 5);
	Er = new double[RcdSize]; // recorded electrode potential container
	tr = new double[RcdSize]; // recorded time container
	Is = new double[RcdSize]; // recorded square wave current container
	Io = new double[RcdSize]; // recorded oxidization current container
	lastIs = new double[RcdSize]; // recorded square wave current container of last period
	lastIo = new double[RcdSize]; // recorded oxidization current container of last period
}

SquareWave::~SquareWave()
{
	delete Er;
	delete tr;
	delete Is;
	delete Io;
	delete lastIs;
	delete lastIo;
}

double SquareWave::AppliedPotential(long i)
{
	double tq = (i - 1)*dt; // current time
	double tq1 = i*dt; // next time

			   // define the potential vs time
			   // square wave voltammetry
	if (i == 0) {
		Eqm = E0;
		Eq = E0;
	}
	else {
		if (WaveJudge == -1 && WaveJudge1 == 1) {
			Eqm = Eqm + dE; // provide base potential for the current time node
		}

		// reset wave_judge for the next time node
		// calculate Eq
		if (tq*swf - floor(tq*swf) < 0.5) {
			WaveJudge = 1;
			Eq = Eqm + swamp;
		}
		else {
			WaveJudge = -1;
			Eq = Eqm - swamp;
		}
	}

	// reset wave_judge1 for the next time node
	if (tq1*swf - floor(tq1*swf) < 0.5) {
		WaveJudge1 = 1;
	}
	else {
		WaveJudge1 = -1;
	}

	return Eq;
}

void SquareWave::RecordCurrent(double I)
{
	if (WaveJudge == 1 && WaveJudge1 == -1) {
		Iqr[0] = I;
	}
	else if (WaveJudge == -1 && WaveJudge1 == 1) {
		Iqr[1] = I;
		Is[PeriodCounter] = Iqr[0] - Iqr[1];
		Io[PeriodCounter] = Iqr[0];
		Er[PeriodCounter] = Eqm;
		std::cout << Er[PeriodCounter] << "V " << Is[PeriodCounter] << "A\n ";
		++PeriodCounter;
	}
}

bool SquareWave::IsPeak() const
{
	if (Is[PeriodCounter - 1] > Is[PeriodCounter] && Is[PeriodCounter - 1] > Is[PeriodCounter - 2])
		return true;
	else
		return false;
}

void SquareWave::ExportCurrent() const
{
	std::ofstream fcout("current.txt");

	if (fcout.is_open()) {
		fcout << "parameters: "
			<< "\nstarting potential, V: " << E0
			<< "\nend potential, V: " << Eend
			<< "\nstep potential, V: " << dE
			<< "\nfrequency, Hz: " << swf
			<< "\namplitude, V: " << swamp;

		fcout << "\npotential/V " << "SWV_Current/A " << "Ox_Current/A\n";

		for (int i = 0; i < PeriodCounter; ++i) {
			fcout << Er[i] << " " << Is[i] << " " << Io[i] << "\n";
		}
	}
	else {
		std::cerr << "Cannot open the file to save current";
		exit(1);
	}
}