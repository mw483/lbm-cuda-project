#ifndef MATHFUNCLBM_H_
#define MATHFUNCLBM_H_


#include <iostream>
#include "definePrecision.h"
#include "defineCoefficient.h"


namespace	
mathFuncLBM {


// bounce-back model for solid boundary //
inline __device__ 
void 
solid_boundary_bounce_back (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	);


// Bouzidi's model for solid boundary //
inline __device__ 
void 
solid_boundary_BOUZIDI (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	);


// Bouzidi's model for moving boundary //
inline __device__ 
void 
moving_boundary_BOUZIDI (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const FLOAT		force_obs[],
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	);


// momentum exchange on boundary //
inline __device__ 
void 
momentum_exchange_on_boundary_D3Q19 (
	      FLOAT		force_obs[],
	const FLOAT*	l_obs,
	const FLOAT*	u_obs,
	const FLOAT*	v_obs,
	const FLOAT*	w_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		nn
	);


}

#include "mathFuncLBM_inc.h"

#endif
