#include "Generate_Particle_GPU.h"


__global__ void
gpu_particle_source_box (
	ParticlePosition	*ppos,
	int		index_frg,
	int		array_start,
	int	  num_x,  int   num_y,  int   num_z,
	FLOAT pos_x,  FLOAT pos_y,  FLOAT pos_z,
	FLOAT vec_dx, FLOAT vec_dy, FLOAT vec_dz
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;
	if (id >= num_x*num_y*num_z)	{ return; }


	// particle index //
	const int	id_g = (array_start) + id;


	// source //
	const int	id_x = ( id%(num_x*num_y) ) % num_x;
	const int	id_y = ( id%(num_x*num_y) ) / num_x;
	const int	id_z = ( id/(num_x*num_y) );

	const FLOAT	source_x = pos_x + vec_dx*id_x;
	const FLOAT	source_y = pos_y + vec_dy*id_y;
	const FLOAT	source_z = pos_z + vec_dz*id_z;


	const int	state_p = PARTICLE_CAL;


	// generate particle //
	ppos[id_g].x_p = source_x;
	ppos[id_g].y_p = source_y;
	ppos[id_g].z_p = source_z;

	ppos[id_g].u_sgs = 0.0;
	ppos[id_g].v_sgs = 0.0;
	ppos[id_g].w_sgs = 0.0;

	ppos[id_g].state_p        = state_p;
	ppos[id_g].source_index_p = index_frg;
}

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
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;
	if (id >= num_x*num_y*num_z)	{ return; }


	// particle index //
	const int	id_g = (array_start) + id;


	// source //
	const FLOAT	source_x = source_xd[id];
	const FLOAT	source_y = source_yd[id];
	const FLOAT	source_z = source_zd[id];

	// sgs velocity //
	const FLOAT	vel_us   = vel_us_d[id] / c_ref;
	const FLOAT	vel_vs   = vel_vs_d[id] / c_ref;
	const FLOAT     vel_ws   = vel_ws_d[id] / c_ref;

	// timing //
	int   state_p;
	if ((t-gen_step_*pstart)%(Group[id]*gen_step_) == 0) {
		state_p = PARTICLE_CAL;
	} else {
		state_p = PARTICLE_NA;
	}

	// particle index (position + timing) //
	const int pos_id = pos_idd[id] + (int)(t-gen_step_*pstart)/gen_step_;

	// generate particle //
	ppos[id_g].x_p = source_x;
	ppos[id_g].y_p = source_y;
	ppos[id_g].z_p = source_z;

//	ppos[id_g].u_sgs = 0.0;
//	ppos[id_g].v_sgs = 0.0;
//	ppos[id_g].w_sgs = 0.0;
	ppos[id_g].u_sgs = vel_us;
	ppos[id_g].v_sgs = vel_vs;
	ppos[id_g].w_sgs = vel_ws;

	ppos[id_g].state_p        = state_p;
	ppos[id_g].source_index_p = pos_id;
}

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
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num_x*num_y)	{ return; }

	// particle index //
	const int	id_g = (array_start) + id;

	// source //
	const int	id_theta0 = ( id%(num_x*num_y) ) % num_x;
	const int	id_theta1 = ( id%(num_x*num_y) ) / num_x;

	const FLOAT	offset = (id_theta0%2 == 0) ? 0.5 : 0.0;
	const FLOAT	theta0 = theta0_min + fabs(theta0_max - theta0_min) * ((FLOAT)id_theta0 + 0.5   )/(FLOAT)num_x;
	const FLOAT	theta1 = theta1_min + fabs(theta1_max - theta1_min) * ((FLOAT)id_theta1 + offset)/(FLOAT)num_y;


	const FLOAT	source_x = pos_x + radius * sin(theta0) * cos(theta1);
	const FLOAT	source_y = pos_y + radius * sin(theta0) * sin(theta1);
	const FLOAT	source_z = pos_z + radius * cos(theta0);

	const int	state_p = PARTICLE_CAL;

	// generate particle
	ppos[id_g].x_p = source_x;
	ppos[id_g].y_p = source_y;
	ppos[id_g].z_p = source_z;

	ppos[id_g].state_p        = state_p;
	ppos[id_g].source_index_p = index_frg;
}


__global__ void
gpu_particle_region_check (
	ParticlePosition *ppos,
	int		array_start,
	int	  num_x,  int   num_y,  int   num_z,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;
	if (id >= num_x*num_y*num_z)	{ return; }


	// particle index //
	const int	id_g = (array_start) + id;

	const FLOAT	source_x = ppos[id_g].x_p;
	const FLOAT	source_y = ppos[id_g].y_p;
	const FLOAT	source_z = ppos[id_g].z_p;


	if (   source_x < xs_min || source_x >= xs_max
		|| source_y < ys_min || source_y >= ys_max
		|| source_z < zs_min || source_z >= zs_max ) {
		const int	state_p = PARTICLE_NA;

		// update //
		ppos[id_g].state_p        = state_p;
	}
}


// GPU_Particle_Generate.cu //
