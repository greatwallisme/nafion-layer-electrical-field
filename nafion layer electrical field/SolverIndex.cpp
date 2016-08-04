#include "SolverIndex.h"

OneDIndex::OneDIndex(const mesh& membrane, const mesh& solution)
{
	InnerIndex.resize(SpeciesEnum::Count);

	for (int k = SpeciesEnum::mReactant; k < SpeciesEnum::sReactant; ++k) {
		InnerIndex[k].resize(membrane.m);
		for (int i = 0; i < membrane.m; ++i) {
			InnerIndex[k][i].resize(membrane.n);

			for (int j = 0; j < membrane.n; ++j) {
				InnerIndex[k][i][j] = i*membrane.n + j + k*membrane.Getmxn();
			}
		}
	}

	for (int k = SpeciesEnum::sReactant; k < SpeciesEnum::Count; ++k) {
		InnerIndex[k].resize(solution.m);
		for (int i = 0; i < solution.m; ++i) {
			InnerIndex[k][i].resize(solution.n);

			for (int j = 0; j < solution.n; ++j) {
				InnerIndex[k][i][j] = i*membrane.n + j + membrane.Getmxn()*SpeciesEnum::sReactant + solution.Getmxn()*(k - SpeciesEnum::sReactant);
			}
		}
	}
}

TwoDIndex::TwoDIndex(const mesh& membrane, const mesh& solution)
{
	unsigned long lenth = membrane.Getmxn()*SpeciesEnum::sReactant + solution.Getmxn()*(SpeciesEnum::Count - SpeciesEnum::sReactant);
	InnerIndex.resize(lenth);

	for (int k = SpeciesEnum::mReactant; k < SpeciesEnum::sReactant; ++k) {
		for (int i = 0; i < membrane.m; ++i) {
			for (int j = 0; j < membrane.n; ++j) {

				unsigned long index = i*membrane.n + j + k*membrane.Getmxn();
				InnerIndex[index] = std::make_tuple(k, j, i);
			}
		}
	}

	for (int k = SpeciesEnum::sReactant; k < SpeciesEnum::Count; ++k) {
		for (int i = 0; i < solution.m; ++i) {
			for (int j = 0; j < solution.n; ++j) {

				unsigned long index = i*membrane.n + j + membrane.Getmxn()*SpeciesEnum::sReactant + solution.Getmxn()*(k - SpeciesEnum::sReactant);
				InnerIndex[index] = std::make_tuple(k, j, i);
			}
		}
	}
}