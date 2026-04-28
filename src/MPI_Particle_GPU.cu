#include "MPI_Particle_GPU.h"

#include "defineParticleFlag.h"


// MPI frag
__global__ void CUDA_MPI_Frag_Particle(
	ParticlePosition *ppos,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max,
	int num)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }
	if (ppos[id].state_p == PARTICLE_NA)	{ return; }

	const FLOAT	x = ppos[id].x_p;
	const FLOAT	y = ppos[id].y_p;
	const FLOAT	z = ppos[id].z_p;

	int		state_p = ppos[id].state_p;


	// 1D
	if      ( x >= xs_max )	{ state_p = PARTICLE_XP; }
	else if ( x <  xs_min )	{ state_p = PARTICLE_XM; }

	if      ( y >= ys_max )	{ state_p = PARTICLE_YP; }
	else if ( y <  ys_min )	{ state_p = PARTICLE_YM; }

	if      ( z >= zs_max )	{ state_p = PARTICLE_ZP; }
	else if ( z <  zs_min )	{ state_p = PARTICLE_ZM; }

	__syncthreads();

	// 2D
	if      ( x >= xs_max && y >= ys_max )	{ state_p = PARTICLE_XPYP; }
	else if ( x <  xs_min && y >= ys_max )	{ state_p = PARTICLE_XMYP; }
	else if ( x >= xs_max && y <  ys_min )	{ state_p = PARTICLE_XPYM; }
	else if ( x <  xs_min && y <  ys_min )	{ state_p = PARTICLE_XMYM; }

	if      ( y >= ys_max && z >= zs_max )	{ state_p = PARTICLE_YPZP; }
	else if ( y <  ys_min && z >= zs_max )	{ state_p = PARTICLE_YMZP; }
	else if ( y >= ys_max && z <  zs_min )	{ state_p = PARTICLE_YPZM; }
	else if ( y <  ys_min && z <  zs_min )	{ state_p = PARTICLE_YMZM; }

	if      ( x >= xs_max && z >= zs_max )	{ state_p = PARTICLE_XPZP; }
	else if ( x <  xs_min && z >= zs_max )	{ state_p = PARTICLE_XMZP; }
	else if ( x >= xs_max && z <  zs_min )	{ state_p = PARTICLE_XPZM; }
	else if ( x <  xs_min && z <  zs_min )	{ state_p = PARTICLE_XMZM; }

	__syncthreads();

	// 3D
	if      ( x >= xs_max && y >= ys_max && z >= zs_max )	{ state_p = PARTICLE_XPYPZP; }
	else if ( x <  xs_min && y >= ys_max && z >= zs_max )	{ state_p = PARTICLE_XMYPZP; }
	else if ( x >= xs_max && y <  ys_min && z >= zs_max )	{ state_p = PARTICLE_XPYMZP; }
	else if ( x >= xs_max && y >= ys_max && z <  zs_min )	{ state_p = PARTICLE_XPYPZM; }
	else if ( x <  xs_min && y <  ys_min && z >= zs_max )	{ state_p = PARTICLE_XMYMZP; }
	else if ( x <  xs_min && y >= ys_max && z <  zs_min )	{ state_p = PARTICLE_XMYPZM; }
	else if ( x <  xs_min && y <  ys_min && z >= zs_max )	{ state_p = PARTICLE_XMYMZP; }
	else if ( x <  xs_min && y <  ys_min && z <  zs_min )	{ state_p = PARTICLE_XMYMZM; }

	__syncthreads();

	// update
	ppos[id].state_p = state_p;
}


__global__ void GPU_Particle_State_Cal(
	ParticlePosition *ppos, int num, int index_num)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }

	ppos[id].state_p = PARTICLE_CAL;

	ppos[id].source_index_p = index_num;
}

// (YOKOUCHI 2020)
__global__ void GPU_Particle_State_Cal_LSM (
	ParticlePosition *ppos, int num, int index_num)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }

	ppos[id].state_p = PARTICLE_CAL;

}

__global__ void CUDA_Particle_MPI_Clear(
	ParticlePosition *ppos, int num)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }
	if (ppos[id].state_p == PARTICLE_NA)	{ return; }

	int		state_p = ppos[id].state_p;
	if (state_p != PARTICLE_CAL)	{ state_p = PARTICLE_NA; }

	ppos[id].state_p = state_p;
}

__global__ void GPU_MPI_Frag(
	ParticlePosition *ppos, int *frg, int num)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }

	int		state_p = ppos[id].state_p;
	int		state_frg = 0;
	if (state_p != PARTICLE_CAL && state_p != PARTICLE_NA)	{ state_frg = 1; }

	frg[id] = state_frg;
}

// GPU_CAL_CUDA.cu *****

