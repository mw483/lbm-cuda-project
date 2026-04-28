#ifndef BOUNDARY_H_
#define BOUNDARY_H_


#include <cuda.h>
#include <cuda_runtime.h>

// define //
#include "definePrecision.h"
#include "defineLBM.h"


// boundary
// x //
__global__ void
CUDA_Boundary_x_Neuman (
	int		rank_x,
	int		ncpu_x,
	FLOAT	*f,
	int nx, int ny, int nz);


// y //
__global__ void
CUDA_Boundary_y_Neuman (
	int rank_y,
	int ncpu_y,
	FLOAT *f,
	int nx, int ny, int nz);


// z //
__global__ void
CUDA_Boundary_z_Neuman (
	int rank_z,
	int ncpu_z,
	FLOAT *f,
	int nx, int ny, int nz);


// bounbary velocity
__global__ void 
status_velocity (
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	const char *status,
	int nx,     int ny,     int nz,
	int halo
	);


__global__ void 
status_lv_velocity (
	      FLOAT	*r,
	      FLOAT	*u,
	      FLOAT	*v,
	      FLOAT	*w,
	const FLOAT	*lv_obs,
	int nx,     int ny,     int nz,
	int halo
	);


__global__ void 
status_lv_lbm_velocity (
	      FLOAT*	f,
		  FLOAT*    T, // mod 2019
	      FLOAT*	r,
	      FLOAT*	u,
	      FLOAT*	v,
	      FLOAT*	w,
	const FLOAT*	lv_obs,
	const int nx,     
	const int ny, 
	const int nz,
	const int halo,
	LBM_VELOCITY_MODEL	d3qx_velocity
	);


//__global__ void 		// MOD 2018
//status_lv_lbm_velocity (
//	      FLOAT*	f,
//	      FLOAT*	r,
//	      FLOAT*	u,
//	      FLOAT*	v,
//	      FLOAT*	w,
//	      FLOAT*	T,
//	const FLOAT*	lv_obs,
//	const int nx,     
//	const int ny, 
//	const int nz,
//	const int halo,
//	LBM_VELOCITY_MODEL	d3qx_velocity
//	);


#endif
