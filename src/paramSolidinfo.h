#ifndef PARAMSOLIDINFO_H_
#define PARAMSOLIDINFO_H_


#include "definePrecision.h"
#include "defineMemory.h"
#include "stFluidProperty.h"
#include "paramMPI.h"
#include "paramDomain.h"

#include "stSolidInfo.h"


class
paramSolidinfo {
private:
	// mpiinfo //
	MPIinfo	mpiinfo_;

	// domain //
	Domain	domain_;

	// solid //
	int		num_solid_;

	solidData	*soliddata_h_;
	solidData	*soliddata_d_;

public:	
	paramSolidinfo () {}

	paramSolidinfo (
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain
		)
	{
		set (
			pmpi,
			pdomain
			);
	}

	~paramSolidinfo () {}

public:
	void
	set (
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain
		);


	void	
	set_levelset_obstacle (FluidProperty	*fluid);


private:
	// solidData //
	void
	allocate_solidData ();

	void
	set_solidData ();


	void	setObstacleData (FluidProperty	*fluid);


	// 物体status flag設定
	void
	set_status_flag (
		      char	*status,
		const FLOAT	*lv
		);

};


#endif
