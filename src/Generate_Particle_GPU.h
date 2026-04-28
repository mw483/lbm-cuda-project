#ifndef GPU_PARTICLES_GENERATE_H_
#define GPU_PARTICLES_GENERATE_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Define.h"
#include "VariablesParticle.h"

#include "Math_Lib_Particle_GPU.h"


__global__ void
gpu_particle_source_box (
	ParticlePosition *ppos,
	int		index_frg,
	int		array_start,
	int		num_x,
	int		num_y,
	int		num_z,
	FLOAT pos_x,  FLOAT pos_y,  FLOAT pos_z,
	FLOAT vec_dx, FLOAT vec_dy, FLOAT vec_dz
	);

// (YOKOUCHI 2020)
__global__ void
gpu_particle_source_LSM (
	ParticlePosition	*ppos,
	int		index_frg,
	int		array_start,
	int	  num_x,  int   num_y,  int   num_z,
	FLOAT pos_x,  FLOAT pos_y,  FLOAT pos_z,
	FLOAT vec_dx, FLOAT vec_dy, FLOAT vec_dz,
	FLOAT *source_xd, FLOAT *source_yd, FLOAT *source_zd,
	FLOAT *vel_us_d,  FLOAT *vel_vs_d,  FLOAT *vel_ws_d,
	int *Group, int *pos_idd,
	int t,
	int pstart,
	int gen_step_,
	int c_ref
	);

__global__ void
gpu_particle_source_sphere (
	ParticlePosition	*ppos,
	int		index_frg,
	int		array_start,
	int   num_x, int   num_y,
	FLOAT pos_x, FLOAT pos_y, FLOAT pos_z,
	FLOAT radius,
	FLOAT theta0_min, FLOAT theta0_max,
	FLOAT theta1_min, FLOAT theta1_max
	);


__global__ void
gpu_particle_region_check (
	ParticlePosition *ppos,
	int		array_start,
	int	  num_x,  int   num_y,  int   num_z,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max
	);


#endif
