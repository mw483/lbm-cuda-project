#ifndef IBM_GPU_H_
#define IBM_GPU_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include "definePrecision.h"


__global__ void	
cuda_set_ibm_f_force (
	const FLOAT *f,	// 物理量
	const FLOAT *l,	// 物体からの距離
	FLOAT *f_f,		// Immeresed boundary 外力項
	int nx, int ny, int nz,
	int halo,
	FLOAT dx,
	FLOAT c_dt
	);


__global__ void	
cuda_set_ibm_velocity_force (
	const FLOAT *u,	// 物理量
	const FLOAT *l,	// 物体からの距離
	const FLOAT	*u_bc,
	      FLOAT *force_u,		// Immeresed boundary 外力項
	int nx, int ny, int nz,
	int halo,
	FLOAT dx,
	FLOAT c_dt
	);


#endif
