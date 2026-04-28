#ifndef SOLID_FORCE_GPU_H_
#define SOLID_FORCE_GPU_H_


#include <cuda.h>
#include "definePrecision.h"


__global__ void
cuda_get_solidForce_tensor_bb (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv,
		  FLOAT	delta,
		  FLOAT	vis,
		  FLOAT	rho_ref,
		  FLOAT	vel_ref,
	      FLOAT	dx,
	int nx, int ny, int nz,
	int halo
	);


__global__ void
cuda_get_solidForce_tensor_bounce_back (
	      FLOAT*	force_inner_x,
	      FLOAT*	force_inner_y,
	      FLOAT*	force_inner_z,
	const FLOAT*	r,
	const FLOAT*	u,
	const FLOAT*	v,
	const FLOAT*	w,
	const FLOAT*	lv,
	const FLOAT		delta,
	const FLOAT		vis,
	const FLOAT		rho_ref,
	const FLOAT		vel_ref,
	const FLOAT		dx,
	const int		nx, 
	const int		ny, 
	const int		nz,
	const int		halo
	);


__global__ void
cuda_get_solidForce_tensor_bb_nvec (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv,
		  FLOAT	delta,
		  FLOAT	vis,
		  FLOAT	rho_ref,
		  FLOAT	vel_ref,
	      FLOAT	dx,
	int nx, int ny, int nz,
	int halo
	);


__global__ void
cuda_get_solidForce_tensor (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv,
		  FLOAT	delta,
		  FLOAT	vis,
		  FLOAT	rho_ref,
		  FLOAT	vel_ref,
	      FLOAT	dx,
	int nx, int ny, int nz,
	int halo
	);


__global__ void
cuda_filter_solidForce (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const int	*id_solidData,
	int nx, int ny, int nz,
	int halo
	);


#endif
