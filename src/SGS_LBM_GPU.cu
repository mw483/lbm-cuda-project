#include "SGS_LBM_GPU.h"

#include "defineCUDA.h"
#include "defineBoundaryFlag.h"


// SGS model
__global__ void CUDA_Fcs_CSM_Model(
	const FLOAT *u, const FLOAT *v, const FLOAT *w,  
	const char *status,
	FLOAT *Fcs_sgs,
	FLOAT *Div,
	FLOAT *SS, FLOAT *WW,
	int nx,     int ny,     int nz,
	int halo_x, int halo_y, int halo_z)
{
	const int	id_x  = halo_x + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo_y + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo_z + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	// index local
	const int	i_m1 = (id_x-1+nx)%nx,
				i_p1 = (id_x+1   )%nx;
	const int	j_m1 = (id_y-1),
				j_p1 = (id_y+1);
	const int	nxy = nx*ny;

	// variables *****
	FLOAT	cu[2], cv[2], cw[2];
	const FLOAT	ep = 1.0e-16;


#pragma unroll
	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// index local
		const int	k_m1 = id_z - 1;
		const int	k_p1 = id_z + 1;

		// index
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;
		const int	id_m1_c0_c0 = i_m1 + nx*id_y + nxy*id_z;
		const int	id_p1_c0_c0 = i_p1 + nx*id_y + nxy*id_z;
		const int	id_c0_m1_c0 = id_x + nx*j_m1 + nxy*id_z;
		const int	id_c0_p1_c0 = id_x + nx*j_p1 + nxy*id_z;
		const int	id_c0_c0_m1 = id_x + nx*id_y + nxy*k_m1;
		const int	id_c0_c0_p1 = id_x + nx*id_y + nxy*k_p1;

		// calculation *****
		// fx
		cu[0] = (u[id_c0_c0_c0] + u[id_m1_c0_c0]) * 0.5f;
		cu[1] = (u[id_c0_c0_c0] + u[id_p1_c0_c0]) * 0.5f;

		cv[0] = (v[id_c0_c0_c0] + v[id_m1_c0_c0]) * 0.5f;
		cv[1] = (v[id_c0_c0_c0] + v[id_p1_c0_c0]) * 0.5f;

		cw[0] = (w[id_c0_c0_c0] + w[id_m1_c0_c0]) * 0.5f;
		cw[1] = (w[id_c0_c0_c0] + w[id_p1_c0_c0]) * 0.5f;

//		if (status[id_m1_c0_c0] == STATUS_WALL)	{ cu[0] = 0.0;  cv[0]  = 0.0;  cw[0]  = 0.0; }
//		if (status[id_p1_c0_c0] == STATUS_WALL)	{ cu[1] = 0.0;  cv[1]  = 0.0;  cw[1]  = 0.0; }
		if (status[id_m1_c0_c0] == STATUS_WALL)	{ cu[0] = u[id_c0_c0_c0]*0.5f;  cv[0] = v[id_c0_c0_c0]*0.5f;  cw[0] = w[id_c0_c0_c0]*0.5f; }
		if (status[id_p1_c0_c0] == STATUS_WALL)	{ cu[1] = u[id_c0_c0_c0]*0.5f;  cv[1] = v[id_c0_c0_c0]*0.5f;  cw[1] = w[id_c0_c0_c0]*0.5f; }

		const FLOAT	ux = (cu[1] - cu[0]); // 2 flops
		const FLOAT	vx = (cv[1] - cv[0]);
		const FLOAT	wx = (cw[1] - cw[0]);

		// fy
		cu[0] = (u[id_c0_c0_c0] + u[id_c0_m1_c0]) * 0.5f;
		cu[1] = (u[id_c0_c0_c0] + u[id_c0_p1_c0]) * 0.5f;

		cv[0] = (v[id_c0_c0_c0] + v[id_c0_m1_c0]) * 0.5f;
		cv[1] = (v[id_c0_c0_c0] + v[id_c0_p1_c0]) * 0.5f;

		cw[0] = (w[id_c0_c0_c0] + w[id_c0_m1_c0]) * 0.5f;
		cw[1] = (w[id_c0_c0_c0] + w[id_c0_p1_c0]) * 0.5f;

//		if (status[id_c0_m1_c0] == STATUS_WALL)	{ cu[0] = 0.0;  cv[0] = 0.0;  cw[0] = 0.0; }
//		if (status[id_c0_p1_c0] == STATUS_WALL)	{ cu[1] = 0.0;  cv[1] = 0.0;  cw[1] = 0.0; }
		if (status[id_c0_m1_c0] == STATUS_WALL)	{ cu[0] = u[id_c0_c0_c0]*0.5f;  cv[0] = v[id_c0_c0_c0]*0.5f;  cw[0] = w[id_c0_c0_c0]*0.5f; }
		if (status[id_c0_p1_c0] == STATUS_WALL)	{ cu[1] = u[id_c0_c0_c0]*0.5f;  cv[1] = v[id_c0_c0_c0]*0.5f;  cw[1] = w[id_c0_c0_c0]*0.5f; }

		const FLOAT	uy = (cu[1] - cu[0]);
		const FLOAT	vy = (cv[1] - cv[0]);
		const FLOAT	wy = (cw[1] - cw[0]);

		// fz
		cu[0] = (u[id_c0_c0_c0] + u[id_c0_c0_m1]) * 0.5f;
		cu[1] = (u[id_c0_c0_c0] + u[id_c0_c0_p1]) * 0.5f;

		cv[0] = (v[id_c0_c0_c0] + v[id_c0_c0_m1]) * 0.5f;
		cv[1] = (v[id_c0_c0_c0] + v[id_c0_c0_p1]) * 0.5f;

		cw[0] = (w[id_c0_c0_c0] + w[id_c0_c0_m1]) * 0.5f;
		cw[1] = (w[id_c0_c0_c0] + w[id_c0_c0_p1]) * 0.5f;

//		if (status[id_c0_c0_m1] == STATUS_WALL)	{ cu[0] = 0.0;  cv[0] = 0.0;  cw[0] = 0.0; }
//		if (status[id_c0_c0_p1] == STATUS_WALL)	{ cu[1] = 0.0;  cv[1] = 0.0;  cw[1] = 0.0; }
		if (status[id_c0_c0_m1] == STATUS_WALL)	{ cu[0] = u[id_c0_c0_c0]*0.5f;  cv[0] = v[id_c0_c0_c0]*0.5f;  cw[0] = w[id_c0_c0_c0]*0.5f; }
		if (status[id_c0_c0_p1] == STATUS_WALL)	{ cu[1] = u[id_c0_c0_c0]*0.5f;  cv[1] = v[id_c0_c0_c0]*0.5f;  cw[1] = w[id_c0_c0_c0]*0.5f; }

		const FLOAT	uz = (cu[1] - cu[0]);
		const FLOAT	vz = (cv[1] - cv[0]);
		const FLOAT	wz = (cw[1] - cw[0]);


		// Sij, Wij
		const FLOAT	s11 = ux;
		const FLOAT	s22 = vy;
		const FLOAT	s33 = wz;

		const FLOAT	s12 = (vx + uy) * 0.5f; // 2 flops
		const FLOAT	s23 = (wy + vz) * 0.5f;
		const FLOAT	s13 = (uz + wx) * 0.5f;

		const FLOAT	w12 = (vx - uy) * 0.5f;
		const FLOAT	w23 = (wy - vz) * 0.5f;
		const FLOAT	w13 = (uz - wx) * 0.5f;


		// SS, WW
		const FLOAT	ss = ( s11*s11 + s22*s22 + s33*s33 ) + 2.0f*( s12*s12 + s23*s23 + s13*s13 ); // 18 flops
		const FLOAT	ww = 								   2.0f*( w12*w12 + w23*w23 + w13*w13 );

		// update *****
		Fcs_sgs[id_c0_c0_c0] = fabs( ww-ss ) / ( ww+ss + ep );  // 4
		Div    [id_c0_c0_c0] = ux + vy + wz;

		SS [id_c0_c0_c0] = ss;
		WW [id_c0_c0_c0] = ww;
	}
}


// SGS_LBM_GPU.cu *****
