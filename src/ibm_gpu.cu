#include "ibm_gpu.h"

#include "defineCUDA.h"
#include "defineCoefficient.h"
#include "indexLib.h"
#include "mathLib.h"
#include "functionLib.h"
#include "matharrayLib.h"


// ibm neumann //
__global__ void	
cuda_set_ibm_f_force (
	const FLOAT *f,	// 物理量
	const FLOAT *l,	// 物体からの距離
	FLOAT *f_f,		// Immeresed boundary 外力項
	int nx, int ny, int nz,
	int halo,
	FLOAT dx,
	FLOAT c_dt
	)
{
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// variables //
	const FLOAT	dd = dx;

	FLOAT	nvec[3];
	int		id_cell[8];
	FLOAT	weight3[3];

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// local index //
		const int	index[3] = { id_x, id_y, id_z };
		const int	grid [3] = { nx, ny, nz };


		// global index //
		const int	id_c0_c0_c0 = indexLib::get_index (index, grid);

		if (l[id_c0_c0_c0] <= 0.0 || fabs(l[id_c0_c0_c0]) > dd*coefficient::IBM_INTERFACE) {
			f_f[id_c0_c0_c0] = 0.0;
			continue;
		}


		// calculation //
		matharrayLib::normal_vector_from_surface (nvec, &l[id_c0_c0_c0], nx, ny, nz);


		// ID, Weight //
		const FLOAT	l_w     =  l[id_c0_c0_c0];
		const FLOAT	l_f     = -l[id_c0_c0_c0];
		const FLOAT	vec3[3] = {	-nvec[0]*(fabs(l_w) + fabs(l_f)),
								-nvec[1]*(fabs(l_w) + fabs(l_f)),
								-nvec[2]*(fabs(l_w) + fabs(l_f)) };

		indexLib::get_cellId_weight_3D (id_cell, weight3, index, grid, vec3, dx);

		// 流体領域の物理量 //
		const FLOAT	cff = functionLib::interpolate_f_in_cell  (f, id_cell, weight3);

		// Neumann境界 //
		const FLOAT	ep = coefficient::NON_ZERO_EP;
		const FLOAT	cfg =   fabs(l_w)/(fabs(l_f)+ep) * cff;

		// update
		f_f[id_c0_c0_c0] = ( cfg - f[id_c0_c0_c0] ) / c_dt;
	}
}


// ibm dirichlet //
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
	)
{
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// variables //
	const FLOAT	dd = dx;

	FLOAT	nvec[3];
	int		id_cell[8];
	FLOAT	weight3[3];

	int		id_cell_bc[8];
	FLOAT	weight3_bc[3];

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// local index //
		const int	index[3] = { id_x, id_y, id_z };
		const int	grid [3] = { nx, ny, nz };


		// global index //
		const int	id_c0_c0_c0 = indexLib::get_index (index, grid);

		if (l[id_c0_c0_c0] <= 0.0) {
			force_u[id_c0_c0_c0] = 0.0;
			continue;
		}
		else if (fabs(l[id_c0_c0_c0]) > dd*coefficient::IBM_INTERFACE) {
			force_u[id_c0_c0_c0] = ( u_bc[id_c0_c0_c0] - u[id_c0_c0_c0] ) / c_dt;
			continue;
		}


		// calculation //
		matharrayLib::normal_vector_from_surface (nvec, &l[id_c0_c0_c0], nx, ny, nz);


		// ID, Weight //
		const FLOAT	l_w     =  l[id_c0_c0_c0];
		const FLOAT	l_f     = -l[id_c0_c0_c0];
		const FLOAT	vec3[3]    = {	-nvec[0]*(fabs(l_w) + fabs(l_f)),
									-nvec[1]*(fabs(l_w) + fabs(l_f)),
									-nvec[2]*(fabs(l_w) + fabs(l_f)) };
		const FLOAT	vec3_bc[3] = {	-nvec[0]*(fabs(l_w)),
									-nvec[1]*(fabs(l_w)),
									-nvec[2]*(fabs(l_w)) };


		indexLib::get_cellId_weight_3D (id_cell,    weight3,    index, grid, vec3,    dx);
		indexLib::get_cellId_weight_3D (id_cell_bc, weight3_bc, index, grid, vec3_bc, dx);


		// 流体領域の物理量 //
		const FLOAT	cuf   = functionLib::interpolate_f_in_cell  (u,    id_cell,    weight3   );
		const FLOAT	cu_bc = functionLib::interpolate_f_in_cell  (u_bc, id_cell_bc, weight3_bc);

		// Dirichlet //
		const FLOAT	ep = coefficient::NON_ZERO_EP;
		const FLOAT	cug =   (fabs(l_w) + fabs(l_f)) / (fabs(l_f)+ep) * cu_bc 
						   - fabs(l_w)              / (fabs(l_f)+ep) * cuf;

		// update
		const FLOAT	wall = functionLib::heaviside_function ( l_w, coefficient::IBM_INTERFACE*dd );
		force_u[id_c0_c0_c0] = ( (1.0 - wall)*cug + wall*u_bc[id_c0_c0_c0] - u[id_c0_c0_c0] ) / c_dt;
	}
}


// ibm_gpu.cu //
