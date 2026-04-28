#ifndef BOUNDARY_LBM_H_
#define BOUNDARY_LBM_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Define.h"



__global__ void
boundary_LBM_y_Neumann (
	int		rank_y,
	int		ncpu_y,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz);


__global__ void
boundary_LBM_z_Neumann (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz);


__global__ void
boundary_LBM_z_Slip (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz);


__global__ void
boundary_LBM_z_Upper (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*T,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	const FLOAT	rho_ref,
	const FLOAT	u_ref,
	const FLOAT	pt_ref,
	int nx, int ny, int nz);


__global__ void
boundary_LBM_z_D3Q19_Upper (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz
	);


// inflow & outflow //
__global__ void
boundary_LBM_x_D3Q19_inflow_outflow (
	int		rank_x,
	int		ncpu_x,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	FLOAT	rho_ref,
	FLOAT	u_ref,
	int nx, int ny, int nz
	);


__global__ void
boundary_LBM_x_inflow_outflow (
	int		rank_x,
	int		ncpu_x,
	FLOAT	*f,
	FLOAT	*T,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	FLOAT	rho_ref,
	FLOAT	u_ref,
	int nx, int ny, int nz);


__global__ void
boundary_LBM_z_inflow_outflow (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	FLOAT	rho_ref,
	FLOAT	u_ref,
	int nx, int ny, int nz);

__global__ void
boundary_LBM_x_inflow_outflow_driver (
	int		rank_x,
	int		ncpu_x,
	FLOAT	*f,
	FLOAT	*T,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	FLOAT	rho_ref,
	FLOAT	u_ref,
	int nx, int ny, int nz);

#endif
