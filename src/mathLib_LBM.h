#ifndef MATHLIBLBM_H_
#define MATHLIBLBM_H_


#include <iostream>
#include "definePrecision.h"
#include "defineCoefficient.h"


namespace	
mathLib_LBM {


template <typename T>
inline __device__ void
device_lbm_feq (
	FLOAT	rho,	// input
	FLOAT	us,
	FLOAT	vs,
	FLOAT	ws,
	FLOAT	feq[],	// output
	T		lbm_model
	);


// feq
inline __device__ void
Device_LBM_D3Q19_feq (
	FLOAT	rho,	// input
	FLOAT	us,
	FLOAT	vs,
	FLOAT	ws,
	FLOAT	feq[]	// output
	);


inline __device__ void
Device_LBM_D3Q27_feq (
	FLOAT	rho,	// input
	FLOAT	us,
	FLOAT	vs,
	FLOAT	ws,
	FLOAT	feq[]	// output
	);


// rho
template <typename T>
inline __device__ void
device_lbm_rho (
	      FLOAT	&rho,
	const FLOAT	*fs,
	T		lbm_model
	);


inline __device__ void
Device_LBM_D3Q19_rho (
	      FLOAT	&rho,
	const FLOAT	*fs
	);


inline __device__ void
Device_LBM_D3Q27_rho (
	      FLOAT	&rho,
	const FLOAT	*fs
	);


// velocity
template <typename T>
inline __device__ void 
device_lbm_velocity (
	      FLOAT	rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs,
	T		lbm_model
	);


inline __device__ void 
Device_LBM_D3Q19_velocity (
	      FLOAT	rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	);


inline __device__ void 
Device_LBM_D3Q27_velocity (
	      FLOAT	rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	);


// rho & velocity
template <typename T>
inline __device__ void 
device_lbm_rho_velocity (
	      FLOAT	&rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs,
	T	lbm_model
	);


inline __device__ void 
Device_LBM_D3Q19_rho_velocity (
	      FLOAT	&rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	);


inline __device__ void 
Device_LBM_D3Q27_rho_velocity (
	      FLOAT	&rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	);

// feq 
inline __device__ void
Device_LBM_D3Q27_equivalent(
		  FLOAT &rho,
		  FLOAT &us,
		  FLOAT &vs,
		  FLOAT &ws,
		  FLOAT fs[]
	);

// fs - feq
template <typename T>
inline __device__ void
device_lbm_d_equivalent (
	      FLOAT	fs_deq[],
	      FLOAT	&rho, 
	const FLOAT	fs[],
	T		lbm_model
	);


inline __device__ void
Device_LBM_D3Q19_d_equivalent (
	      FLOAT	fs_deq[],
	      FLOAT	&rho, 
	const FLOAT	fs[]
	);


inline __device__ void
Device_LBM_D3Q27_d_equivalent (
	      FLOAT	fs_deq[],
	      FLOAT	&rho, 
	const FLOAT	fs[]
	);

inline __device__ void		//MOD2018
Device_LBM_D3Q27_d_equivalent_force (
		FLOAT	fs_deq[],
		FLOAT	&rho,
	const	FLOAT	fs[],
		FLOAT	&force_x,
		FLOAT	&force_y,
		FLOAT	&force_z
	);


// sgs viscosity
template <typename T>
inline __device__ FLOAT 
device_lbm_SGS_viscosity_deq (
	      FLOAT	Fcs,
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[],
	T		lbm_model
	);


inline __device__ FLOAT 
Device_LBM_D3Q19_SGS_viscosity_deq (
	      FLOAT	Fcs,
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	);


inline __device__ FLOAT 
Device_LBM_D3Q27_SGS_viscosity_deq (
	      FLOAT	Fcs,
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	);


// stress tensor
template <typename T>
inline __device__ FLOAT
device_lbm_stress_deq (
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[],
	T		lbm_model
	);


inline __device__ FLOAT
Device_LBM_D3Q19_stress_deq (
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	);


inline __device__ FLOAT
Device_LBM_D3Q27_stress_deq (
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	);


// force
template <typename T>
inline __device__ void 
device_lbm_force (
	FLOAT	rho,
	FLOAT	force_x,
	FLOAT	force_y,
	FLOAT	force_z,
	FLOAT	force_lbm[],
	T		lbm_model
	);


inline __device__ void 
Device_LBM_D3Q19_Force (
	FLOAT	rho,
	FLOAT	force_x,
	FLOAT	force_y,
	FLOAT	force_z,
	FLOAT	force_lbm[]
	);


inline __device__ void 
Device_LBM_D3Q27_Force (
	FLOAT	rho,
	FLOAT	force_x,
	FLOAT	force_y,
	FLOAT	force_z,
	FLOAT	force_lbm[]
	);


// boundary //
template <typename T>
inline __device__ void 
device_boundary_stream (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn,
	T		boundary_model
	);


inline __device__ void 
device_boundary_bounce_back (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	);


inline __device__ void 
device_boundary_BOUZIDI (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	);


inline __device__ void 
device_boundary_YU (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	);


inline __device__ void 
device_boundary_2nd_poly (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	);


// index global
inline __device__ int Index_Global(int id_x, int id_y, int id_z, 
		int x_offset, int y_offset, int z_offset,
		int  nx, int  ny, int  nz);

// (YOKOUCHI 2020)
// total TKE for Lagrangian Stochastic Model
inline __device__ FLOAT  
Device_LBM_D3Q27_totalTKE (
	FLOAT	fs_deq[],
	FLOAT	rho
	);

}

#include "mathLib_LBM_inc.h"

#endif
