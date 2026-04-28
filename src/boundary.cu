#include "boundary.h"

#include "defineCUDA.h"
#include "defineBoundaryFlag.h"
#include "defineReferenceVel.h"
#include "defineLBM.h"

// Lib //
#include "mathLib_LBM.h"


// x //
__global__ void
CUDA_Boundary_x_Neuman (
	int		rank_x,
	int		ncpu_x,
	FLOAT	*f,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_z  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;

	if (id_y >= ny || id_z >= nz)	{ return; }

	id_g = nx*id_y + nxy*id_z;


	// f
	if (rank_x == 0) {
		f[id_g         ] =  f[id_g + 1     ];
	}
	if (rank_x == (ncpu_x-1)) {
		f[id_g + (nx-1)] =  f[id_g + (nx-2)];
	}
}


// y //
__global__ void
CUDA_Boundary_y_Neuman (
	int rank_y,
	int ncpu_y,
	FLOAT *f,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_z  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;

	if (id_x >= nx || id_z >= nz)	{ return; }

	id_g = id_x + nxy*id_z;


	// f
	if (rank_y == 0) {
		f[id_g            ] =  f[id_g + nx       ];
	}
	if (rank_y == (ncpu_y-1)) {
		f[id_g + nx*(ny-1)] =  f[id_g + nx*(ny-2)];
	}
}


// z //
__global__ void
CUDA_Boundary_z_Neuman (
	int rank_z,
	int ncpu_z,
	FLOAT *f,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;

	if (id_x >= nx || id_y >= ny)	{ return; }

	id_g = id_x + nx*id_y;

	// f
	if (rank_z == 0) {
		f[id_g             ] =  f[id_g + nxy       ];
	}
	if (rank_z == (ncpu_z-1)) {
		f[id_g + nxy*(nz-1)] =  f[id_g + nxy*(nz-2)];
	}
}


// bounbary velocity
__global__ void
status_velocity (
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	const char *status,
	int nx,     int ny,     int nz,
	int halo
	)
{
	// cuda index
	const int	nxy = nx*ny;
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		if (status[id_c0_c0_c0] == STATUS_WALL) {
			const FLOAT	us = 0.0;
			const FLOAT	vs = 0.0;
			const FLOAT	ws = 0.0;

			// update //
			u[id_c0_c0_c0]   = us;
			v[id_c0_c0_c0]   = vs;
			w[id_c0_c0_c0]   = ws;
		}
		else {
		}
	}
}


// bounbary velocity
__global__ void
status_lv_velocity (
	      FLOAT	*r,
	      FLOAT	*u,
	      FLOAT	*v,
	      FLOAT	*w,
	const FLOAT	*lv_obs,
	int nx,     int ny,     int nz,
	int halo
	)
{
	// cuda index //
	const int	nxy = nx*ny;
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		if (lv_obs[id_c0_c0_c0] >= 0.0) {
			const FLOAT	rs = 1.0;

			const FLOAT	us = 0.0;
			const FLOAT	vs = 0.0;
			const FLOAT	ws = 0.0;

			// update //
			r[id_c0_c0_c0]   = rs;
			u[id_c0_c0_c0]   = us;
			v[id_c0_c0_c0]   = vs;
			w[id_c0_c0_c0]   = ws;
		}
	}
}


__global__ void
status_lv_lbm_velocity (
	      FLOAT*	f,
		  FLOAT*	T, // mod2019
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
	)
{
	// cuda index //
	const int	nxy = nx*ny;
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// variables //
	const int	nn = nx*ny*nz;
	FLOAT	feq[NUM_DIRECTION_VEL];


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		if (lv_obs[id_c0_c0_c0] >= 0.0) {
			const FLOAT	rs = 1.0;

			const FLOAT	us = 0.0;
			const FLOAT	vs = 0.0;
			const FLOAT	ws = 0.0;

			mathLib_LBM::device_lbm_feq (rs, us, vs, ws, feq, d3qx_velocity);


			// update //
			r[id_c0_c0_c0]   = rs;
			u[id_c0_c0_c0]   = us;
			v[id_c0_c0_c0]   = vs;
			w[id_c0_c0_c0]   = ws;
			
			T[id_c0_c0_c0]   = 300.0;

			for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
				f[nn*ii + id_c0_c0_c0] = feq[ii];
			}
		}
	}
}
