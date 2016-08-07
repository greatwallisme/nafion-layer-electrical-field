#pragma once
#include <vector>
#include "mesh.h"
#include "EnumStruct.h"


class OneDIndex
{
public:
	OneDIndex(const mesh& membrane, const mesh& solution);
	long operator() (SpeciesEnum::Species species, long j, long i) const { return InnerIndex[species][i][j]; } ;// this function returns the long 1d index
	
private:
	vector<vector<vector<long>>> InnerIndex;
};

class TwoDIndex
{
	typedef std::tuple<SpeciesEnum::Species, long, long> triple;
public:
	TwoDIndex(const mesh& membrane, const mesh& solution);
	triple operator() (long OneDIndex) const { return InnerIndex[OneDIndex]; } // this function return enum species, long j, long i

private:
	vector<triple> InnerIndex;

};