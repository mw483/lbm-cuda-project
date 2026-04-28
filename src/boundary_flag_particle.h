#ifndef BOUNDARY_FLAG_PARTICLE_H_
#define BOUNDARY_FLAG_PARTICLE_H_


#include <cuda.h>
#include "definePrecision.h"
#include "VariablesParticle.h"


__global__ void
boundary_flag_particle_cuda (
	ParticlePosition	*ppos,
	FLOAT	xs_min, 
	FLOAT	ys_min, 
	FLOAT	zs_min,
	FLOAT	xs_max, 
	FLOAT	ys_max, 
	FLOAT	zs_max,
	int		num
	);


#endif
