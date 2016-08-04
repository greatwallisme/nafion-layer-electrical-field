#pragma once
#include <vector>
#include "mesh.h"
#include "EnumStruct.h"


class OneDIndex
{
public:
	OneDIndex(const mesh& membrane, const mesh& solution);
	unsigned long operator() (SpeciesEnum::Species species, unsigned long j, unsigned long i) const { return InnerIndex[species][i][j]; } ;// this function returns the unsigned long 1d index
	
private:
	vector<vector<vector<unsigned long>>> InnerIndex;
};

class TwoDIndex
{
	typedef std::tuple<SpeciesEnum::Species, unsigned long, unsigned long> triple;
public:
	TwoDIndex(const mesh& membrane, const mesh& solution);
	triple operator() (unsigned long OneDIndex) const { return InnerIndex[OneDIndex]; } // this function return enum species, unsigned long j, unsigned long i

private:
	vector<triple> InnerIndex;

};