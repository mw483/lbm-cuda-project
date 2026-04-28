#include "paramSolidinfo.h"

#include "functionLib.h"
#include "allocateLib.h"
#include "macroCUDA.h"

#include "defineBoundaryFlag.h"
#include "defineCoefficient.h"


// public //
void paramSolidinfo::
set (
	const paramMPI		&pmpi, 
	const paramDomain	&pdomain
	)
{
	// mpiinfo //
	mpiinfo_ = pmpi.mpiinfo();

	// domain //
	domain_ = pdomain.domain();
}


void	paramSolidinfo::
set_levelset_obstacle (FluidProperty	*fluid)
{
	setObstacleData (fluid);
}


// memcpy //
// private //
void paramSolidinfo::
setObstacleData (FluidProperty	*fluid)
{
	if (mpiinfo_.rank == 0) { std::cout << "set obstacle\n"; }

	// STL
//	read_STL_levelset_data (
//		sforce_h_.id_solidData,
//		fluid->l_obs,
//		stl_solid_h_
//		);
	classSTL	cstl (domain_);

	cstl.read_STL_levelset_data (
		sforce_h_.id_solidData,
		fluid->l_obs,
		);


	// status
	set_status_flag (
		fluid->status,
	   	fluid->l_obs
		);
}


// 物体status flag設定
void	paramSolidinfo::
set_status_flag (
	      char	*status,
	const FLOAT	*lv
	)
{
	const int	nn = domain_.nn;

	// wall +
	for (int i=0; i<nn; i++) {
		status[i] = (lv[i] >= 0.0) ? STATUS_WALL : STATUS_FLUID;
	}
}
