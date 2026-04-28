#ifndef PARAMBASISVARIABLES_H_
#define PARAMBASISVARIABLES_H_


#include "definePrecision.h"
#include "defineMemory.h"
#include "stBasisVariables.h"
#include "paramMPI.h"
#include "paramCalFlag.h"
#include "paramDomain.h"

class
paramBasisVariables {
private:	
	// mpi //
	int		rank_;

	// domain //
	int		nx_,  ny_,  nz_;
	int		nn_;

	// lbm //
	FLOAT	cfl_ref_;

	FLOAT	vel_cfl_ref_;	// MOD2019a

	// flag //
	int		restart_;

public:	
	paramBasisVariables () {}

	paramBasisVariables (
		char				*program_name,
		int					argc,
	   	char				*argv[], 
		const paramMPI		&pmpi, 
		const paramCalFlag	&pcalflg,
		const paramDomain	&pdomain
		)
	{
		set (
			pmpi,
			pcalflg,
			pdomain
			);
	}

	~paramBasisVariables () {}

public:	
	void
	set (
		const paramMPI		&pmpi, 
		const paramCalFlag	&pcalflg,
		const paramDomain	&pdomain
		);


	void
	allocate (
		BasisVariables	*basisv,
		defineMemory::FlagHostDevice	flag_memory
		);


	void	init_host (BasisVariables	*basisv);


	void
	memcpy_BasisVariables (
		      BasisVariables *basisvn,
		const BasisVariables *basisv
		);


	void	output_restart_BasisVariables (BasisVariables *basisv);

private:
	// allocate stBasisVariables.h //
	void	allocate_host   (BasisVariables	*basisv); 
	void	allocate_device (BasisVariables	*basisv);


	void	init_data (BasisVariables *basisv);
	void	read_data (BasisVariables *basisv);

};


#endif
