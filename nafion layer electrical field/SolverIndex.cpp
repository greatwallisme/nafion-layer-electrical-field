#include "SolverIndex.h"

OneDIndex::OneDIndex(const mesh& membrane, const mesh& solution)
{
	InnerIndex.resize(OneDIndex::Count);

	for (int k = OneDIndex::mReactant; k < OneDIndex::sReactant; ++k) {
		InnerIndex[k].resize(membrane.m);
		for (int i = 0; i < membrane.m; ++i) {
			InnerIndex[k][i].resize(membrane.n);

			for (int j = 0; j < membrane.n; ++j) {
				InnerIndex[k][i][j] = i*membrane.n + j + k*membrane.Getmxn();
			}
		}
	}

	for (int k = OneDIndex::sReactant; k < OneDIndex::Count; ++k) {
		InnerIndex[k].resize(solution.m);
		for (int i = 0; i < solution.m; ++i) {
			InnerIndex[k][i].resize(solution.n);

			for (int j = 0; j < solution.n; ++j) {
				InnerIndex[k][i][j] = i*membrane.n + j + membrane.Getmxn()*sReactant + solution.Getmxn()*(k - sReactant);
			}
		}
	}
}

TwoDIndex::TwoDIndex(const mesh& membrane, const mesh& solution)
{
	unsigned long lenth = membrane.Getmxn()*TwoDIndex::sReactant + solution.Getmxn()*(TwoDIndex::Count - TwoDIndex::sReactant);
	InnerIndex.resize(lenth);

	for (int k = OneDIndex::mReactant; k < OneDIndex::sReactant; ++k) {
		for (int i = 0; i < membrane.m; ++i) {
			for (int j = 0; j < membrane.n; ++j) {

				unsigned long index = i*membrane.n + j + k*membrane.Getmxn();
				InnerIndex[index] = std::make_tuple(k, j, i);
			}
		}
	}

	for (int k = OneDIndex::sReactant; k < OneDIndex::Count; ++k) {
		for (int i = 0; i < solution.m; ++i) {
			for (int j = 0; j < solution.n; ++j) {

				unsigned long index = i*membrane.n + j + membrane.Getmxn()*sReactant + solution.Getmxn()*(k - sReactant);
				InnerIndex[index] = std::make_tuple(k, j, i);
			}
		}
	}
}