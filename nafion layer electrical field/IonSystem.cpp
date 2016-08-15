#include "IonSystem.h"
#include <fstream>

IonSystem::IonSystem(double fEpsilon_r, double fEpsilon_0, Ion& fReactant, Ion& fProduct, Ion& fSupportCation, Ion& fSupportAnion, Ion& fPotential, Ion& fImmobileCharge) :
	Reactant(fReactant), Product(fProduct), SupportCation(fSupportCation), SupportAnion(fSupportAnion), ImmobileCharge(fImmobileCharge), Potential(fPotential),
	Epsilon_r(fEpsilon_r), Epsilon_0(fEpsilon_0), ReciprocalEpsilon_rEpsilon_0(1/fEpsilon_0/fEpsilon_r), CxZImmobileCharge(ImmobileCharge.DensityN(0, 0) *ImmobileCharge.Z)
{

}

Ion::Ion(double fD, int fZ, double fCinitial, mesh& Mesh) :
	D(fD), Z(fZ), Cinitial(fCinitial), DensityN(Mesh.n, Mesh.m)
{
	for (long i = 0; i < Mesh.m; ++i) {
		for (long j = 0; j < Mesh.n; ++j) {
			DensityN(j, i) = Cinitial;
		}
	}
}

void Ion::PrintDense(std::string fileName)
{
	std::ofstream fout;
	fout.open(fileName);

	if (fout.is_open())
		fout << DensityN;
}

