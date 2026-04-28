#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include "Define.h"
#include "Variables.h"
#include "VariablesParticle.h"

#include "stCalFlag.h"

class Output {
private:
	std::ofstream fout;
	FILE *plot_;
	FILE *plot2D_;
	FILE *plotVec_;
	char fname_[100];

public:
	// File
	void Output_Cal_Condition(const struct timeval *begin, const struct timeval *end,
		MPI_Library *mpi,
		CalFlag *calfrg, Domain *cdo);

	// channel flow
	void Output_Channel(MPI_Library *mpi, const int step, Domain *cdo, const BasisVariables *cq, const Stress *str, const FluidProperty *cfp);
};

#endif

