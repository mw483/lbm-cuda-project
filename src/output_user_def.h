#ifndef OUTPUT_USER_DEF_H_
#define OUTPUT_USER_DEF_H_

#include <cstdio>
#include <cmath>
#include <string>

#include <fstream>
#include "paramDomain.h"
#include "paramMPI.h"
#include "stVariables.h"
#include "stBasisVariables.h"
#include "stFluidProperty.h"

// (YOKOUCHI 2020)
#include "stStress.h"

void
output_user_def (
	int						t,
	char					*program_name,
	int						argc,
	char					*argv[], 
	const paramMPI			&pmpi, 
	const paramDomain		&pdomain,
	const Variables     	*const variables,
	const BasisVariables	*const cbq,
	const FluidProperty		*const cfp
	);

// (YOKOUCHI 2020)

void
output_vis_sgs (
	int			t,
	char			*program_name,
	int			argc,
	char			*argv[],
	const paramMPI		&pmpi,
	const paramDomain	&pdomain,
	const Stress		*const str,
	const BasisVariables	*const cbq,
	const FluidProperty	*const cfp
	);

#endif
