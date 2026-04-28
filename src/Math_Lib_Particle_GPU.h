#ifndef MATH_LIB_PARTICLE_GPU_H_
#define MATH_LIB_PARTICLE_GPU_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Define.h"

// (YOKOUCHI 2020)
#include "stDomain.h"
#include "stStress.h"
#include <curand_kernel.h>


// math function *****
// device *****
inline __device__ int Check_Particle_Status(
		const char *frg,
		FLOAT x, FLOAT y, FLOAT z, 
		FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
		FLOAT dx, FLOAT dy, FLOAT dz, 
		int nx, int ny, int nz,
		int halo);


inline __device__ __host__ FLOAT 
Interpolate_Particle_Velocity (
	const FLOAT	*f,
	FLOAT	x,
	FLOAT	y,
	FLOAT	z, 
	FLOAT	xs_min,
	FLOAT	ys_min,
	FLOAT	zs_min,
	FLOAT	dx, 
	FLOAT	dy, 
	FLOAT	dz, 
	int nx, int ny, int nz,
	int halo);


inline __device__ __host__ int 
Index_left_collocate_period (
	FLOAT	x,
   	FLOAT	xc_min,
   	FLOAT	dx,
   	int		nx,
   	int		offset_x
	);


inline __device__ __host__ int 
Index_left_collocate (
	FLOAT	x,
	FLOAT	xc_min,
	FLOAT	dx,
	int		nx, 
	int		offset_x
	);

	
// 1次元補間
template<typename T>
inline __device__ __host__ T
interpolate_f_1d (
	const T	fc[],
	T	wx);


// 3次元補間
template<typename T>
inline __device__ __host__ T
interpolate_f_3d (
	const T	fc[],
	T	wx,
	T	wy,
	T	wz);


// 粒子の格子点上での左側のindex
template<typename T>
inline __device__ __host__ int 
index_particle_left_collocate (
	T	x_particle,
	T	xmin_center,
	T	dx,
	int	nx,
	int	offset_x);

// (YOKOUCHI 2020)
// Random SGS velocity
inline __device__ __host__ FLOAT
Random_SGS_Velocity_Energy (
	FLOAT		vel_sgs, 
	const FLOAT *vis_sgs, const FLOAT *vis_sgs_old,
	const FLOAT k_GS,
	FLOAT x, FLOAT y, FLOAT z,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT rand,
	FLOAT dx, FLOAT dy, FLOAT dz,
	int nx, int ny, int nz,
	int		halo,
	FLOAT		dt,
	int		dim	
	);

// the local SGS TKE
inline __device__ __host__ FLOAT
local_SGS_TKE (
	const int	id,
	FLOAT		dx,
	const FLOAT	*vis_sgs,
	FLOAT		c_ref
	);

// the local dissipation rate
inline __device__ __host__ FLOAT
local_dissipation_rate (
	FLOAT		dx,
	FLOAT		tke_sgs	
	);

// the mean contribution of the SGS TKE to the total TKE
inline __device__ __host__ FLOAT
contribution_SGS_TKE (
	const	FLOAT	vis,
	const	FLOAT	GS_tke,
	const	FLOAT	dx
	);

// Random SGS velocity (vis_sgs)
inline __device__ __host__ FLOAT
Random_SGS_Velocity_Vis_sgs (
	FLOAT		vel_sgs,
	const FLOAT *vis_sgs, const FLOAT *vis_sgs_old,
//	const FLOAT *total_TKE,
	const FLOAT k_GS,
	FLOAT x, FLOAT y, FLOAT z,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT rand,
	FLOAT dx, FLOAT dy, FLOAT dz,
	int nx, int ny, int nz,
	int		halo,
	FLOAT		dt,
	int		dim	
	);

inline __device__ void particle_reflection (
	const FLOAT	*l_obs,
	const FLOAT x,		const FLOAT y, 		const FLOAT z,
	      FLOAT &x_new, 	      FLOAT &y_new,	      FLOAT &z_new,
	      FLOAT &u_s,	      FLOAT &v_s,	      FLOAT &w_s,
	      FLOAT xs_nin,	      FLOAT ys_min,	      FLOAT zs_min,
	      FLOAT dx,		      FLOAT dy,		      FLOAT dz,
	      int   nx,		      int   ny,		      int   nz,
	      int   halo	
	);

// include 
#include "Math_Lib_Particle_GPU_inc.h"

#endif

