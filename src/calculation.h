#ifndef CALCULATION_H_
#define CALCULATION_H_


#include <cuda.h>
#include <cuda_runtime.h>
#include "definePrecision.h"
#include "stVariables.h"
#include "stBasisVariables.h"
#include "stFluidProperty.h"
#include "stStress.h"
#include "stMPIinfo.h"
#include "paramMPI.h"
#include "paramDomain.h"


class
Calculation {
public:
	// mpi //
	paramMPI	pmpi_;

	// mpi
	int		rank_;
	int		rank_x_,  rank_y_,  rank_z_;

	int		ncpu_;
	int		ncpu_x_,  ncpu_y_,  ncpu_z_;

	// domain
	int		nx_,  ny_,  nz_;
	int		nn_;

	int		nxg_,  nyg_,  nzg_;

	int		halo_;

    FLOAT dx_;
    FLOAT dt_;

	// lbm //
	FLOAT	c_ref_;
	FLOAT	cfl_ref_;

	FLOAT	vel_cfl_ref_;	// MOD2019a

	// cuda
	dim3	grid_,
			block_2d_;

public:
	Calculation () {}

	Calculation (
		const paramMPI	&pmpi,
		const paramDomain	&pdomain) :
		pmpi_(pmpi)
	{
		set (pdomain);
	}

	~Calculation () {}

public:
	void
	set (const paramDomain	&pdomain);


	// initialize
	void 
	initialize_calculation (
		Variables		*cq,
		Variables		*cqn,
		BasisVariables	*cbq,
		FluidProperty	*cfp
		);


	// Calculation
	void
	gpu_calculation (
		Variables		*cq,
   		Variables		*cqn,
		BasisVariables	*cbq,
		Stress			*str,
   		FluidProperty	*cfp
		);

	// (YOKOUCHI 2020)
	// Mean velocity
	void
	mean_velocity (
		const	BasisVariables	*cbq,
			Stress		*str,
			int 		t
		);

	// (YOKOUCHI 2020)
	// TKE sgs
	void
	sgs_tke_LBM (
		const	BasisVariables	*cbq,
			Stress		*str,
			int		t
		);
	

private:
	void
	calculation_LBM (
   		const Variables			*cq,
   		      Variables			*cqn, 
		      BasisVariables	*cbq,
		      Stress			*str,
   		      FluidProperty		*cfp
		);


	// sgs model //
	void
	sgs_model (
		const BasisVariables	*const cbq,
		      Stress			*str,
   		const FluidProperty		*const cfp
		);


	// LBM //
	void
	stream_collision (
	   	const Variables			*cq,
	   	      Variables			*cqn, 
		      BasisVariables	*cbq,
		      Stress			*str,
	   	      FluidProperty		*cfp
		);


	void
	stream_collision_moving_boundary (
	   	const Variables			*cq,
	   	      Variables			*cqn, 
		      BasisVariables	*cbq,
		      Stress			*str,
	   	      FluidProperty		*cfp
		);


    void
    stream_collision_thermal_convection (
       	const Variables			*cq,
       	      Variables			*cqn,
    	      BasisVariables	*cbq,
    	      Stress			*str,
       	      FluidProperty		*cfp
    	);


	// force //
	void
	set_force (
		      Stress			*str,
	   	const FluidProperty		*cfp
		);


	void
	force_acceleration (
	   	const Variables			*cq,
	   	      Variables			*cqn, 
		      Stress			*str
		);


	// obstacle //
	void
	obstacle_variables (
	   	      Variables			*cq,
		      BasisVariables	*cbq,
	   	const FluidProperty		*cfp
		);


	// lbm_function_to_velocity //
	void
	lbm_function_to_velocity (
	   	const Variables			*cq,
		      BasisVariables	*cbq
		);


	// obstacle filter //
	void
	velocity_obstacle_filter (
		      BasisVariables	*cbq,
		const FluidProperty		*cfp
		);


	// velocity_to_lbm_function //
	void
	velocity_to_lbm_function (
	   	      Variables			*cq,
		const BasisVariables	*cbq
		);


	// mpi & boundary //
	void
	mpi_boundary_Variables (
		Variables		*cq,
	   	BasisVariables	*cbq,
	   	FluidProperty	*cfp
		);


	// mpi communication //
	void
	mpi_Variables (
		Variables		*cq,
	   	BasisVariables	*cbq,
	   	FluidProperty	*cfp
		);


	void
	boundary_Variables (
		Variables		*cq,
	   	BasisVariables	*cbq,
	   	FluidProperty	*cfp
		);


	void
	boundary_x_Variables (
		Variables		*cq,
	   	BasisVariables	*cbq,
	   	FluidProperty	*cfp
		);


	void
	boundary_y_Variables (
		Variables		*cq,
	   	BasisVariables	*cbq,
	   	FluidProperty	*cfp
		);


	void
	boundary_z_Variables (
		Variables		*cq,
	   	BasisVariables	*cbq,
	   	FluidProperty	*cfp
		);

// For Boussinesq approximation  MOD2018
	void
	Vertical_Profile_T (
		const FLOAT *T,
		FLOAT *T_ref
		);

// For Boussinesq approximation  MOD2018
	void
	Vertical_Profile_T_const (
		FLOAT *T_ref
		);

// For Boussinesq approximation  MOD2018
	void
	Vertical_Profile_T_const2 (
		const FLOAT *T,
		FLOAT *T_ref
		);

};


#endif
