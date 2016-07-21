#include "IonSystem.h"

IonSystem::IonSystem(double fEpsilon_r, double fEpsilon_0, Ion& fReactant, Ion& fProduct, Ion& fSupportCation, Ion& fSupportAnion, Ion& fImmobileCharge, double fCImmobileCharge) :
	Reactant(fReactant), Product(fProduct), SupportCation(fSupportCation), SupportAnion(fSupportAnion), ImmobileCharge(fImmobileCharge), CImmobileCharge(fCImmobileCharge),
	Epsilon_r(fEpsilon_r), Epsilon_0(fEpsilon_0), ReciprocalEpsilon_rEpsilon_0(1/fEpsilon_0/fEpsilon_r), CxZImmobileCharge(CImmobileCharge*ImmobileCharge.Z)
{
}

Ion::Ion(double fD, int fZ, double fCinitial) :
	D(fD), Z(fZ), Cinitial(fCinitial)
{
}