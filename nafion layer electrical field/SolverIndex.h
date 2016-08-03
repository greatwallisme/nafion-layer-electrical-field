#pragma once
#include <vector>
#include "mesh.h"

typedef std::tuple<TwoDIndex::Species, unsigned long, unsigned long> triple;

class OneDIndex
{
public:
	OneDIndex(const mesh& membrane, const mesh& solution);
	enum Species { mReactant, mProduct, mCation, mPotential, sReactant, sProduct, sAnion, sCation, sPotential, Count };
	unsigned long operator() (OneDIndex::Species species, unsigned long j, unsigned long i) const { return InnerIndex[species][i][j]; } ;// this function returns the unsigned long 1d index
	
private:
	vector<vector<vector<unsigned long>>> InnerIndex;
};

class TwoDIndex
{
public:
	TwoDIndex(const mesh& membrane, const mesh& solution);
	enum Species { mReactant, mProduct, mCation, mPotential, sReactant, sProduct, sAnion, sCation, sPotential, Count };
	triple operator() (unsigned long OneDIndex) const { return InnerIndex[OneDIndex]; } // this function return enum species, unsigned long j, unsigned long i

private:
	vector<triple> InnerIndex;

};