#ifndef PARAMFLUIDPROPERTY_H_
#define PARAMFLUIDPROPERTY_H_


#include <string>
#include <iostream>
#include "definePrecision.h"
#include "defineMemory.h"
#include "stFluidProperty.h"
#include "paramMPI.h"
#include "paramDomain.h"

//#include "stSolidInfo.h"


class
paramFluidProperty {
private:	
	// mpiinfo //
	MPIinfo	mpiinfo_;

	// domain //
	Domain	domain_;

public:	
	paramFluidProperty () {}

	paramFluidProperty (
		char				*program_name,
		int					argc,
	   	char				*argv[], 
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain
		)
	{
		set (
			pmpi,
			pdomain
			);
	}

	~paramFluidProperty () {}

public:
	void
	set (
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain
		);


	void
	allocate (
		FluidProperty	*fluid,
		defineMemory::FlagHostDevice	flag_memory
		);


	void	init_host        (FluidProperty	*fluid);
	void	read_global_map  (FluidProperty	*fluid);


	void
	memcpy_FluidProperty (
		      FluidProperty *      fluidn,
	   	const FluidProperty *const fluid
		);

private:
	void	allocate_host   (FluidProperty	*fluid_h); 
	void	allocate_device (FluidProperty	*fluid_d);

	void	initViscosity (FluidProperty	*fluid);

	void	setBounaryViscosity (FluidProperty	*fluid);
	void	setBounaryViscosity_city (FluidProperty	*fluid);

	void
	read_map (
		      int	*id_obs,
		      FLOAT	*lv,
		const FLOAT	z_domain_min,		// MOD 2018
		const FLOAT	lbm_scale
		);
	
	void setPoiseuille (int *id_obs, FLOAT *lv);
	void setSingleCube (int *id_obs, FLOAT *lv);

	void set_objectDirection(FLOAT *l_obs, int *l_obs_x, int *l_obs_y, int *l_obs_z);

	void
	set_status_flag (
		      char	*status,
		const FLOAT	*lv
		);

    void init_thermal_flux (FluidProperty* fluid_h);
    void read_thermal_flux(std::string fname, FLOAT* hflux, int nx_, int ny_, int nz_);
//    void read_thermal_flux_from_global_data(std::string fname, FLOAT* hflux);
    void constant_thermal_flux(FLOAT* hflux, FLOAT hf, int nx_, int ny_, int nz_);
};


#endif
