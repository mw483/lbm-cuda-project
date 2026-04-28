#include "boundary_flag_particle.h"

#include "defineParticleFlag.h"
#include "Math_Lib_Particle_GPU.h"


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
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }
	if (ppos[id].state_p == PARTICLE_NA)	{ return; }


	// calculation //
	int		state_p = ppos[id].state_p;

	FLOAT	x = ppos[id].x_p;
	FLOAT	y = ppos[id].y_p;
	FLOAT	z = ppos[id].z_p;

	if (	x >= xs_max ||	x < xs_min 
		||	y >= ys_max ||	y < ys_min 
		||	z >= zs_max ||	z < zs_min ) { 
		state_p = PARTICLE_NA; 

		x = xs_min;
		y = ys_min;
		z = zs_min;
	}


	// update //
	ppos[id].state_p = state_p;

	ppos[id].x_p = x;
	ppos[id].y_p = y;
	ppos[id].z_p = z;

}


// boundary_flag_particle.cu //
