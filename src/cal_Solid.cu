#include "cal_Solid.h"

#include "functionLib.h"
#include "allocateLib.h"
#include "macroCUDA.h"

#include "defineBoundaryFlag.h"
#include "defineCoefficient.h"


// public //
// initialize //
void	cal_Solid::
init_cal_Solid (
	const MPIinfo		&mpiinfo, 
	const Domain		&domain,
	const FluidProperty	*fluid
	)
{
	// mpiinfo //
	mpiinfo_ = mpiinfo;


	// domain //
	domain_ = domain;


	// classSolidData //
	csoliddata_.init_classSolidData (
		mpiinfo_,
		domain_
		);


	// classSTL //
	creadstl_.init_classReadSTL (
		mpiinfo_.rank,
		domain_,
		csoliddata_.soliddata()
		);


	// classSolidForce //
	csolidforce_.init_classSolidForce (
		mpiinfo_,
		domain_,
		fluid
		);
}


// output levelset function //
void	cal_Solid::
set_levelset_obstacle (
	FluidProperty	*fluid_h
	)
{
	creadstl_.read_STL_levelset_data (
		fluid_h->id_obs,
		fluid_h->l_obs,
		csoliddata_.soliddata()
		);


	// status
	set_status_flag (
		fluid_h->status,
	   	fluid_h->l_obs
		);
}


void	cal_Solid::
move_solid_data_gpu (
	FluidProperty	*fluid_d
	)
{
	FLOAT	force_x = 0.0;
	FLOAT	force_y = 0.0;
	FLOAT	force_z = 0.0;


	csoliddata_.move_SolidData (
		force_x,
		force_y,
		force_z
		);

	creadstl_.read_STL_levelset_data_gpu (
		fluid_d->id_obs,
		fluid_d->l_obs,
		csoliddata_.soliddata()
		);

	creadstl_.read_STL_velocity_data_gpu (
		fluid_d->l_obs,
		fluid_d->u_obs,
		fluid_d->v_obs,
		fluid_d->w_obs,
		csoliddata_.soliddata()
		);
}


// solid Force //
void	cal_Solid::
get_solidForce (
	const int	*id_solidData,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv
	)
{
	// id stl //
	csolidforce_.set_id_solidData (
		id_solidData
		);


	// force //
	csolidforce_.cal_solidForce (
		r,
		u, v, w,
		lv
		);


//	FLOAT	force_solid[3] = {
//				csolidforce_.solidForce_x (0),
//				csolidforce_.solidForce_y (0),
//				csolidforce_.solidForce_z (0)
//				};
}


void	cal_Solid::
cout_solidForce ()
{
	std::cout << "solidForce (x, y, z) = "
		      << csolidforce_.solidForce_x (0) << " ,  "
		      << csolidforce_.solidForce_y (0) << " ,  "
		      << csolidforce_.solidForce_z (0) << std::endl;
}


void	cal_Solid::
output_solidForce (
	bool	first_cal,
	FLOAT	time
	)
{
	csolidforce_.output_solidForce (
		first_cal,
		time
		);
}


void	cal_Solid::
memcpy_solidForce_DeviceToHost ()
{
	csolidforce_.memcpy_solidForce_DeviceToHost ();
}


// 物体status flag設定 //
void	cal_Solid::
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


// cal_Solid.cu //
