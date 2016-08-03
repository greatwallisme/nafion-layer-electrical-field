#pragma once
#include <vector>
#include "mesh.h"

class OneDIndex
{
public:
	OneDIndex(const mesh& membrane, const mesh& solution);
	enum Species { mReactant, mProduct, mCation, mPotential, sReactant, sProduct, sAnion, sCation, sPotential, Count };
	unsigned long operator() (OneDIndex::Species species, unsigned long j, unsigned long i) { return InnerIndex[species][i][j]; } // this function returns the unsigned long 1d index
	
private:
	vector<vector<vector<unsigned long>>> InnerIndex;
};

class TwoDIndex
{
public:
	TwoDIndex(const mesh& membrane, const mesh& solution);
	enum Species { mReactant, mProduct, mCation, mPotential, sReactant, sProduct, sAnion, sCation, sPotential, Count };
	vector<unsigned long> operator() (unsigned long OneDIndex) { return InnerIndex[OneDIndex]; } // this function return enum species, unsigned long j, unsigned long i

private:
	vector<vector<unsigned long>> InnerIndex;

};