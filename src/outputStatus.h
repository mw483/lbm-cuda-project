#ifndef OUTPUTSTATUS_H_
#define OUTPUTSTATUS_H_


#include "definePrecision.h"
#include "stCalFlag.h"
#include "stDomain.h"
#include "stMPIinfo.h"


class
outputStatus {
public:	
	// mpi
	int		rank_;
	int		ncpu_;

	// grid
	int		nx_,  ny_,  nz_;
	int		nn_;

	int		halo_;

	int		nxg_,  nyg_,  nzg_;

	FLOAT	dt_;

	// lbm //
	FLOAT	c_ref_;

public:	
	outputStatus () {}

	outputStatus (
		char	*program_name,
		int		argc,
	   	char	*argv[], 
		const MPIinfo	&mpi, 
		const CalFlag		&calflg,
		const Domain		&domain)
	{
//        std::cout << __PRETTY_FUNCTION__ << std::endl;

		set (
			mpi,
			calflg,
			domain
			);
	}

	~outputStatus () {}

public:	
	void
	set (
		const MPIinfo	&mpi, 
		const CalFlag	&calflg,
		const Domain	&domain
		);


	void
	cout_Status (
		const FLOAT *r,
		const FLOAT *u,
		const FLOAT *v,
		const FLOAT *w,
		const FLOAT *vis,
		FLOAT 		vis0,
		const FLOAT *vis_sgs,
		const FLOAT *Div,
		const char	*status
		);
};


#endif
