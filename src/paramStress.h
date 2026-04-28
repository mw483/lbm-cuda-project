#ifndef PARAMSTRESS_H_
#define PARAMSTRESS_H_


#include "defineMemory.h"
#include "definePrecision.h"

#include "stStress.h"
#include "paramDomain.h"


class
paramStress {
public:
	int		nn_;

public:	
	paramStress () {}

	paramStress (
		char				*program_name,
		int					argc,
	   	char				*argv[], 
		const paramDomain	&pdomain
		)
	{
		set (pdomain);
	}

	~paramStress () {}

public:
	void	set (const paramDomain	&pdomain);

	void
	allocate (
		Stress	*stress,
		defineMemory::FlagHostDevice	flag_memory
		);


	void	init_host (Stress	*stress_h);

	void	memcpy_Stress (Stress *stressn, const Stress *stress);

private:
	// allocate stStress.h //
	void	allocate_host   (Stress	*stress); 
	void	allocate_device (Stress	*stress);

	// initializea //
	void	init_data (Stress *stress);

};


#endif
