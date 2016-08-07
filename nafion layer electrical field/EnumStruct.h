#pragma once
#include <vector>


struct SpeciesEnum
{
	const enum Species { mReactant, mProduct, mCation, mPotential, sReactant, sProduct, sAnion, sCation, sPotential, Count };
};

struct BoundaryEnum
{
	enum Boundary { bulk, bottom, top, left, right, right_bottom, left_bottom_corner, right_bottom_corner, left_upper_corner, right_upper_corner }; // add enum struct for boundary
};
