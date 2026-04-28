#ifndef ADVECTION_PARTICLE_GPU_H_
#define ADVECTION_PARTICLE_GPU_H_


#include <cuda.h>
#include "definePrecision.h"
#include "VariablesParticle.h"

//(YOKOUCHI 2020)
#include <curand_kernel.h>

__global__ void 
CUDA_Particle_Advection (
	ParticlePosition	*ppos,
	const FLOAT			*u, 
	const FLOAT			*v, 
	const FLOAT			*w,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max,
	FLOAT dx,
	FLOAT dt,
	FLOAT c_ref,
	int nx, int ny, int nz,
	int num,
	int halo
	);

// (YOKOUCHI 2020)
__global__ void 
CUDA_Particle_Advection_LSM (
	ParticlePosition	*ppos,
	const FLOAT		*l_obs,
	const FLOAT		*u, 
	const FLOAT		*v, 
	const FLOAT		*w,
//	const FLOAT		*vis_sgs,
//	const FLOAT		*vis_sgs_old,
	const FLOAT		*tke_sgs,
	const FLOAT		*tke_sgs_old,
	const FLOAT		*um,
	const FLOAT		*vm,
	const FLOAT		*wm,	
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max,
	FLOAT dx,
	FLOAT dt,
	FLOAT c_ref,
	int nx, int ny, int nz,
	int num,
	int halo,
	curandState		*rnd_d
	);

__global__ void setCurand (
	unsigned long long	seed,
	curandState		*state,
	const int		nn
	);

__global__ void genrand_normal (
	FLOAT		*rnd_d,
	curandState	*state,
	const int	nn
	);
#endif
