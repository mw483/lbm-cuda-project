#ifndef PARAMVARIABLES_H_
#define PARAMVARIABLES_H_


#include "definePrecision.h"
#include "defineMemory.h"
#include "stVariables.h"
#include "paramMPI.h"
#include "paramCalFlag.h"
#include "paramDomain.h"


class
paramVariables {
private:	
	// mpi //
	int		rank_;
	int		rank_x_, rank_y_, rank_z_;

	// domain //
	Domain	domain_;


	// flag
	int		restart_;
	int		num_direction_vel_;

// MOD 2018
	int		nx_;
	int		ny_;
	int		nz_;
	FLOAT		dx_;
// MOD 2018

public:	
	paramVariables () {}

	paramVariables (
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

	~paramVariables () {}

//	const Variables	&variables() const { return variables_; }

public:	
	void
	set (
		const paramMPI		&pmpi, 
		const paramCalFlag	&pcalflg,
		const paramDomain	&pdomain
		);


	void
	allocate (
		Variables	*variables,
		defineMemory::FlagHostDevice	flag_memory
		);


	void	init_host (Variables	*variables_h);


	void
	memcpy_Variables (
		      Variables *variablesn,
	   	const Variables *variables
		);


	void	output_restart_Variables (Variables *variables);

private:
	// allocate stVariables.h //
	void	allocate_host   (Variables	*variables); 
	void	allocate_device (Variables	*variables);


	// initialize //
	void	init_data (Variables *variables);
	void	read_data (Variables *cbq);

};


#endif
