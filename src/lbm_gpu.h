#ifndef LBM_GPU_H_
#define LBM_GPU_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include "definePrecision.h"
#include "defineLBM.h" // LBM scheme (D3Q19, D3Q27) (Bounce-back, 2nd-order)



// streaming step //
__global__ void 
cuda_stream_collision_D3Q19 (
	const FLOAT	*f,
	      FLOAT	*fn,
	const FLOAT	*l_obs,
	const FLOAT	*vis, 
	      FLOAT	*vis_sgs,
	const FLOAT *Fcs,
	int nx, int ny, int nz,
	int halo
	);


__global__ void 
cuda_stream_collision_D3Q27 (
	const FLOAT	*f,
	      FLOAT	*fn,
	const FLOAT	*l_obs,
	const FLOAT	*vis, 
	      FLOAT	*vis_sgs,
	const FLOAT *Fcs,
	int nx, int ny, int nz,
	int halo
	);


__global__ void 
cuda_stream_collision_D3Q19_moving_boundary (
	const FLOAT*	f,
	      FLOAT*	fn,
	const FLOAT*	l_obs,
	const FLOAT*	u_obs,
	const FLOAT*	v_obs,
	const FLOAT*	w_obs,
	const FLOAT*	vis, 
	      FLOAT*	vis_sgs,
	const FLOAT*	Fcs,
	const int		nx, 
	const int		ny, 
	const int		nz,
	const int		halo
	);


__global__ void
cuda_stream_collision_T_D3Q27 (
	const FLOAT	*f,
	      FLOAT	*fn,
	const FLOAT	*T,
	      FLOAT	*Tn,
	const FLOAT	*l_obs,
	      FLOAT	*u,			//MOD2018
	      FLOAT	*v,			//MOD2018
	      FLOAT	*w,			//MOD2018
	      FLOAT	*rho,			//MOD2018
	const FLOAT	*vis,
	      FLOAT	*vis_sgs,
//	      FLOAT     *vis_sgs_old,		//(YOKOUCHI 2020)
//	      FLOAT	*tke_sgs,		//(YOKOUCHI 2020)
//	      FLOAT	*tke_sgs_old,		//(YOKOUCHI 2020)
	const FLOAT	*T_ref,
	const FLOAT *Fcs,
    const FLOAT* bcTw, //  1,  0,  0 //
    const FLOAT* bcTe, // -1,  0,  0 //
    const FLOAT* bcTs, //  0,  1,  0 //
    const FLOAT* bcTn, //  0, -1,  0 //
    const FLOAT* bcTr, //  0,  0,  1 //
    const FLOAT dx,
    const FLOAT dt,
	const FLOAT	c_ref,
	int nx, int ny, int nz,
	int halo
	);

// force //
__global__ void 
cuda_gravity_force (
	FLOAT	*force_x,
	FLOAT	*force_y,
	FLOAT	*force_z,
	const FLOAT	*l_obs,
	FLOAT	c_ref,
	int nx, int ny, int nz,
	int halo
	);


__global__ void 
cuda_force_acceleration (
	const FLOAT	*f,
	      FLOAT	*fn,
	const FLOAT	*force_x,
	const FLOAT	*force_y,
	const FLOAT	*force_z,
	int nx, int ny, int nz,
	int halo,
	LBM_VELOCITY_MODEL	d3qx_velocity
	);


// lbm_function_to_velocity //
__global__ void 
cuda_lbm_function_to_velocity (
	const FLOAT *f,
	FLOAT *rho,
   	FLOAT *u, FLOAT *v, FLOAT *w,
	int nx, int ny, int nz,
	int halo,
	LBM_VELOCITY_MODEL	d3qx_velocity
	);


// velocity_to_lbm_function //
__global__ void
cuda_velocity_to_lbm_function (
	      FLOAT	*f,
	const FLOAT	*rho, 
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	int nx,     int ny,     int nz,
	int halo,
	LBM_VELOCITY_MODEL	d3qx_velocity
	);



// lbm_function_to_velocity //
__global__ void
cuda_velocity_obstacle_filter (
	      FLOAT	*u,
	      FLOAT	*v,
	      FLOAT	*w,
	const FLOAT	*lv,
	int nx,     int ny,     int nz,
	int halo
	);


// LBM boundary //
__device__
void
LBM_solid_boundary (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	);

__device__
void 
LBM_solid_slip_boundary (
	      FLOAT  fs[],
	const FLOAT* f,
	const FLOAT* l_obs,
	const int*   eD3Q27_x,
	const int*   eD3Q27_y,
	const int*   eD3Q27_z,
	const int    idg_lbm[],
	const int*   direction_lbm_slip[],
	const int    num_direction_vel,
	const int    nx,
	const int    ny,
	const int    nz,
	const int    nn
	);

__device__
void
LBM_moving_boundary_D3Q19 (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const FLOAT*	u_obs,
	const FLOAT*	v_obs,
	const FLOAT*	w_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		nn
	);

__device__ FLOAT wall_func       (FLOAT ut, FLOAT U, FLOAT y);
__device__ FLOAT d_wall_func_dut (FLOAT ut, FLOAT U, FLOAT y);
__device__ FLOAT d_wall_func_dU  (FLOAT ut, FLOAT U, FLOAT y);

// (YOKOUCHI 2020)
// Mean velocity GPU ... host function is in "Calculation" class
__global__ void mean_velocity_GPU (
	const FLOAT *u,	const FLOAT *v,	const FLOAT *w,
	      FLOAT *um,      FLOAT *vm,      FLOAT *wm,
	      int   nx, int ny, int nz,
	      int   halo,
	      int   t
	);

__global__ void tke_LBM_GPU (
	const FLOAT *u, const FLOAT *v, const FLOAT *w,
	      FLOAT *tke_sgs,
	      FLOAT *tke_sgs_old,
	      int   nx, int ny, int nz,
	      int   halo,
	      int   t
	);


#endif
