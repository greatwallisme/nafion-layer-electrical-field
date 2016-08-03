#pragma once
#include "IonSystem.h"
#include "mesh.h"
#include "potential_signal.h"
#include "thermodynamics.h"

class EquationCoefficient
{
	friend class solver;
public:
	EquationCoefficient(const IonSystem& fIons, const mesh& phase, const PotentialSignal& fSignal, const nernst_equation& fThermo);
	void CalculateCoeff(Eigen::MatrixXd& GeoCoeffA, Eigen::MatrixXd GeoCoeffB);

private:
	Eigen::MatrixXd CoeffProductA;
	Eigen::MatrixXd CoeffProductB;
	Eigen::MatrixXd CoeffReactantA;
	Eigen::MatrixXd CoeffReactantB;
	Eigen::MatrixXd CoeffAnionA;
	Eigen::MatrixXd CoeffAnionB;
	Eigen::MatrixXd CoeffCationA;
	Eigen::MatrixXd CoeffCationB;

	Eigen::MatrixXd CoeffPotentialA;
	Eigen::MatrixXd CoeffPotentialB;

	const IonSystem& Ions;
	const PotentialSignal& Signal;
	const double F_R_T;
};