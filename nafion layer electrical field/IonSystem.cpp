#include "IonSystem.h"
#include <fstream>

IonSystem::IonSystem(double fEpsilon_r, double fEpsilon_0, Ion& fReactant, Ion& fProduct, Ion& fSupportCation, Ion& fSupportAnion, Ion& fImmobileCharge, double fCImmobileCharge) :
	Reactant(fReactant), Product(fProduct), SupportCation(fSupportCation), SupportAnion(fSupportAnion), ImmobileCharge(fImmobileCharge), CImmobileCharge(fCImmobileCharge),
	Epsilon_r(fEpsilon_r), Epsilon_0(fEpsilon_0), ReciprocalEpsilon_rEpsilon_0(1/fEpsilon_0/fEpsilon_r), CxZImmobileCharge(CImmobileCharge*ImmobileCharge.Z)
{
}

Ion::Ion(double fD, int fZ, double fCinitial, mesh& Mesh) :
	D(fD), Z(fZ), Cinitial(fCinitial), DenseN(Mesh.n, Mesh.m)
{
	for (long i = 0; i < Mesh.m; ++i) {
		for (long j = 0; j < Mesh.n; ++j) {
			DenseN(j, i) = Cinitial;
		}
	}
}

void Ion::PrintDense(string fileName)
{
	ofstream fout;
	fout.open(fileName);

	if (fout.is_open())
		fout << DenseN;
}

