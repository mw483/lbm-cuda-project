#include "lbm_gpu.h"

#include "defineCUDA.h"
#include "defineReferenceVel.h"
#include "defineCoefficient.h"
#include "mathLib_LBM.h"

// Func //
#include "mathFuncLBM.h"

//cumulant
#include "FuncCumulantLBM.h"

// User defined parameters //
#include "Define_user.h"

#include <assert.h>

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
		)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// D3Q19:
	// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)


	// global index
	// lbm stream from
	// local id (e[ii] : depature point )
	const int	direction_lbm[19] = {
		0, 2, 1, 4, 3, 6, 5,
		10,  9,  8,  7,
		14, 13, 12, 11,
		18, 17, 16, 15  };


	// index local //
	const int	i_m1 = id_x-1,
		  i_p1 = id_x+1;
	const int	j_m1 = id_y-1,
		  j_p1 = id_y+1;
	const int	nxy  = nx*ny;


	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos;	// variables after streaming step //
	FLOAT	fs[19], fs_deq[19];

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		const int	k_m1 = id_z - 1;
		const int	k_p1 = id_z + 1;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		// center (lbm) //
		const int	idg_lbm[19] = {
			id_x + nx*id_y + nxy*id_z,
			i_m1 + nx*id_y + nxy*id_z,
			i_p1 + nx*id_y + nxy*id_z,
			id_x + nx*j_m1 + nxy*id_z,
			id_x + nx*j_p1 + nxy*id_z,
			id_x + nx*id_y + nxy*k_m1,
			id_x + nx*id_y + nxy*k_p1,
			i_m1 + nx*j_m1 + nxy*id_z,
			i_p1 + nx*j_m1 + nxy*id_z,
			i_m1 + nx*j_p1 + nxy*id_z,
			i_p1 + nx*j_p1 + nxy*id_z,
			i_m1 + nx*id_y + nxy*k_m1,
			i_p1 + nx*id_y + nxy*k_m1,
			i_m1 + nx*id_y + nxy*k_p1,
			i_p1 + nx*id_y + nxy*k_p1,
			id_x + nx*j_m1 + nxy*k_m1,
			id_x + nx*j_p1 + nxy*k_m1,
			id_x + nx*j_m1 + nxy*k_p1,
			id_x + nx*j_p1 + nxy*k_p1
		};


		// streaming step //
#pragma unroll
		for (int ii=0; ii<19; ii++) { fs[ii] = f[ii*nxyz + idg_lbm[ii]]; }


		// boundary condition //
		LBM_solid_boundary (
				fs,
				f,
				l_obs,
				idg_lbm,
				direction_lbm,
				19,
				nxyz
				);


		// collision (fs -> rhos, us, vs, ws)
		// f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
		// fs_deq = fs - feq
		mathLib_LBM::Device_LBM_D3Q19_d_equivalent(fs_deq, rhos, fs);

		// sgs model
		// fs, feq -> csvis
		const FLOAT	csvis = mathLib_LBM::Device_LBM_D3Q19_SGS_viscosity_deq (Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
		vis_sgs[id_c0_c0_c0] = csvis;


		// relaxation time
		const FLOAT		tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
		const FLOAT		omega     = (FLOAT)1.0/tau_total; // 1.0/tau


		// fnew = fs - 1/tau * (fs - feq)
#pragma unroll
		for (int ii=0; ii<19; ii++) {
			// viscsity
			fs[ii] -= omega*fs_deq[ii];

			fn[nxyz*ii + id_c0_c0_c0] = fs[ii];
		}
	}
}


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
		)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	// D3Q27:
	// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
	// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
	//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)


	// global index
	// lbm stream from
	// local id (e[ii] : depature point )
	const int	direction_lbm[27] = {
		0, 2, 1, 4, 3, 6, 5,
		10,  9,  8,  7,
		14, 13, 12, 11,
		18, 17, 16, 15,
		26, 23, 24, 25,
		20, 21, 22, 19 	};


	// index local
	const int	i_m1 = id_x-1,
		  i_p1 = id_x+1;
	const int	j_m1 = id_y-1,
		  j_p1 = id_y+1;
	const int	nxy = nx*ny;

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos;	// variables after streaming step
	FLOAT	fs[27], fs_deq[27];


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		const int	k_m1 = id_z - 1;
		const int	k_p1 = id_z + 1;

		// index
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		// center (lbm)
		const int	idg_lbm[27] = {
			id_x + nx*id_y + nxy*id_z,
			i_m1 + nx*id_y + nxy*id_z,
			i_p1 + nx*id_y + nxy*id_z,
			id_x + nx*j_m1 + nxy*id_z,
			id_x + nx*j_p1 + nxy*id_z,
			id_x + nx*id_y + nxy*k_m1,
			id_x + nx*id_y + nxy*k_p1,
			i_m1 + nx*j_m1 + nxy*id_z,
			i_p1 + nx*j_m1 + nxy*id_z,
			i_m1 + nx*j_p1 + nxy*id_z,
			i_p1 + nx*j_p1 + nxy*id_z,
			i_m1 + nx*id_y + nxy*k_m1,
			i_p1 + nx*id_y + nxy*k_m1,
			i_m1 + nx*id_y + nxy*k_p1,
			i_p1 + nx*id_y + nxy*k_p1,
			id_x + nx*j_m1 + nxy*k_m1,
			id_x + nx*j_p1 + nxy*k_m1,
			id_x + nx*j_m1 + nxy*k_p1,
			id_x + nx*j_p1 + nxy*k_p1,
			i_m1 + nx*j_m1 + nxy*k_m1, // 19 ~ 26
			i_p1 + nx*j_m1 + nxy*k_m1,
			i_m1 + nx*j_p1 + nxy*k_m1,
			i_m1 + nx*j_m1 + nxy*k_p1,
			i_m1 + nx*j_p1 + nxy*k_p1,
			i_p1 + nx*j_m1 + nxy*k_p1,
			i_p1 + nx*j_p1 + nxy*k_m1,
			i_p1 + nx*j_p1 + nxy*k_p1  };


		// calculation //
		// streaming & bounce back step
		// streaming step
#pragma unroll
		for (int ii=0; ii<27; ii++) { fs[ii] = f[ii*nxyz + idg_lbm[ii]]; }

		// boundary condition
		LBM_solid_boundary (
				fs,
				f,
				l_obs,
				idg_lbm,
				direction_lbm,
				27,
				nxyz
				);


		// collision (fs -> rhos, us, vs, ws)
		// f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
		// fs_deq = fs - feq
		mathLib_LBM::Device_LBM_D3Q27_d_equivalent(fs_deq, rhos, fs);

		// sgs model
		// fs, feq -> csvis
		const FLOAT	csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
		vis_sgs[id_c0_c0_c0] = csvis;


		// relaxation time
		const FLOAT		tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
		const FLOAT		omega     = (FLOAT)1.0/tau_total; // 1.0/tau


		// fnew = fs - 1/tau * (fs - feq)
#pragma unroll
		for (int ii=0; ii<27; ii++) {
			// viscsity
			fs[ii] -= omega*fs_deq[ii];

			fn[nxyz*ii + id_c0_c0_c0] = fs[ii];
		}
	}
}


// lbm kernel for moving boundary //
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
		)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// D3Q19:
	// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)


	// global index
	// lbm stream from
	// local id (e[ii] : depature point )
	const int	direction_lbm[19] = {
		0, 2, 1, 4, 3, 6, 5,
		10,  9,  8,  7,
		14, 13, 12, 11,
		18, 17, 16, 15
	};


	// index local //
	const int	i_m1 = id_x-1,
		  i_p1 = id_x+1;
	const int	j_m1 = id_y-1,
		  j_p1 = id_y+1;
	const int	nxy  = nx*ny;


	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos;	// variables after streaming step //
	FLOAT	fs[19], fs_deq[19];

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		const int	k_m1 = id_z - 1;
		const int	k_p1 = id_z + 1;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		// center (lbm) //
		const int	idg_lbm[19] = {
			id_x + nx*id_y + nxy*id_z,
			i_m1 + nx*id_y + nxy*id_z,
			i_p1 + nx*id_y + nxy*id_z,
			id_x + nx*j_m1 + nxy*id_z,
			id_x + nx*j_p1 + nxy*id_z,
			id_x + nx*id_y + nxy*k_m1,
			id_x + nx*id_y + nxy*k_p1,
			i_m1 + nx*j_m1 + nxy*id_z,
			i_p1 + nx*j_m1 + nxy*id_z,
			i_m1 + nx*j_p1 + nxy*id_z,
			i_p1 + nx*j_p1 + nxy*id_z,
			i_m1 + nx*id_y + nxy*k_m1,
			i_p1 + nx*id_y + nxy*k_m1,
			i_m1 + nx*id_y + nxy*k_p1,
			i_p1 + nx*id_y + nxy*k_p1,
			id_x + nx*j_m1 + nxy*k_m1,
			id_x + nx*j_p1 + nxy*k_m1,
			id_x + nx*j_m1 + nxy*k_p1,
			id_x + nx*j_p1 + nxy*k_p1
		};


		// streaming step //
#pragma unroll
		for (int ii=0; ii<19; ii++) { fs[ii] = f[ii*nxyz + idg_lbm[ii]]; }


		// boundary condition //
		LBM_moving_boundary_D3Q19 (
				fs,
				f,
				l_obs,
				u_obs,
				v_obs,
				w_obs,
				idg_lbm,
				direction_lbm,
				nxyz
				);


		// collision (fs -> rhos, us, vs, ws)
		// f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
		// fs_deq = fs - feq
		mathLib_LBM::Device_LBM_D3Q19_d_equivalent(fs_deq, rhos, fs);

		// sgs model
		// fs, feq -> csvis
		const FLOAT	csvis = mathLib_LBM::Device_LBM_D3Q19_SGS_viscosity_deq (Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
		vis_sgs[id_c0_c0_c0] = csvis;


		// relaxation time
		const FLOAT		tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
		const FLOAT		omega     = (FLOAT)1.0/tau_total; // 1.0/tau


		// fnew = fs - 1/tau * (fs - feq)
#pragma unroll
		for (int ii=0; ii<19; ii++) {
			// viscsity
			fs[ii] -= omega*fs_deq[ii];

			fn[nxyz*ii + id_c0_c0_c0] = fs[ii];
		}
	}
}


__global__ void
cuda_stream_collision_T_D3Q27 (
		const FLOAT	*f,
		FLOAT	*fn,
		const FLOAT	*T,
		FLOAT	*Tn,
		const FLOAT	*l_obs,
		FLOAT	*u,
		FLOAT	*v,
		FLOAT	*w,
		FLOAT *rho,
		const FLOAT	*vis,
		FLOAT	*vis_sgs,
//		FLOAT	*vis_sgs_old, 	//(YOKOUCHI 2020)
//		FLOAT	*tke_sgs,    	//(YOKOUCHI 2020)
//		FLOAT	*tke_sgs_old,	//(YOKOUCHI 2020)
		const FLOAT* T_ref, // MOD2018 Boussinesq
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
		)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
	id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
	id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	// D3Q27:
	// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
	// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
	//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)
	const int eD3Q27_x[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1, 1, 1,-1,-1,-1 };
	const int eD3Q27_y[27] = { 0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1 };
	const int eD3Q27_z[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1 };

	// global index
	// lbm stream from
	// local id (e[ii] : depature point )
	//							      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}
	const int direction_lbm[27]     = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,14,13,12,11,18,17,16,15,26,23,24,25,20,21,22,19};
	const int direction_lbm_x[27]   = {0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,18,17,16,15,20,19,25,24,26,22,21,23};
	const int direction_lbm_y[27]   = {0, 2, 1, 4, 3, 6, 5, 9,10, 7, 8,14,13,12,11,16,15,18,17,21,25,19,23,22,26,20,24};
	const int direction_lbm_z[27]   = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,13,14,11,12,17,18,15,16,22,24,23,19,21,20,26,25};
	const int direction_lbm_xy[27]  = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,12,11,14,13,16,15,18,17,25,21,20,26,24,23,19,22};
	const int direction_lbm_xz[27]  = {0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,14,13,12,11,17,18,15,16,24,22,26,20,25,19,23,21};
	const int direction_lbm_yz[27]  = {0, 2, 1, 4, 3, 6, 5, 9,10, 7, 8,13,14,11,12,18,17,16,15,23,26,22,21,19,25,24,20};
	const int direction_lbm_xyz[27] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26};
	const int *direction_lbm_slip[8] = {direction_lbm, direction_lbm_x, direction_lbm_y, direction_lbm_xy, 
		direction_lbm_z, direction_lbm_xz, direction_lbm_yz, direction_lbm_xyz};

	// index local
	const int	i_m1 = id_x-1,
		  i_p1 = id_x+1;
	const int	j_m1 = id_y-1,
		  j_p1 = id_y+1;
	const int	nxy = nx*ny;

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos;	// variables after streaming step
	FLOAT	fs[27], fs_deq[27];

	int k__=0;
	for (int k=0; k<BLOCKDIM_Z; k++) {
		k__ = k__ + 1;
		const int	id_z = id_zs + k;

		const int	k_m1 = id_z - 1;
		const int	k_p1 = id_z + 1;

		// index
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		// center (lbm)
		const int	idg_lbm[27] = {
			id_x + nx*id_y + nxy*id_z,
			i_m1 + nx*id_y + nxy*id_z,
			i_p1 + nx*id_y + nxy*id_z,
			id_x + nx*j_m1 + nxy*id_z,
			id_x + nx*j_p1 + nxy*id_z,
			id_x + nx*id_y + nxy*k_m1,
			id_x + nx*id_y + nxy*k_p1,
			i_m1 + nx*j_m1 + nxy*id_z,
			i_p1 + nx*j_m1 + nxy*id_z,
			i_m1 + nx*j_p1 + nxy*id_z,
			i_p1 + nx*j_p1 + nxy*id_z,
			i_m1 + nx*id_y + nxy*k_m1,
			i_p1 + nx*id_y + nxy*k_m1,
			i_m1 + nx*id_y + nxy*k_p1,
			i_p1 + nx*id_y + nxy*k_p1,
			id_x + nx*j_m1 + nxy*k_m1,
			id_x + nx*j_p1 + nxy*k_m1,
			id_x + nx*j_m1 + nxy*k_p1,
			id_x + nx*j_p1 + nxy*k_p1,
			i_m1 + nx*j_m1 + nxy*k_m1, // 19 ~ 26
			i_p1 + nx*j_m1 + nxy*k_m1,
			i_m1 + nx*j_p1 + nxy*k_m1,
			i_m1 + nx*j_m1 + nxy*k_p1,
			i_m1 + nx*j_p1 + nxy*k_p1,
			i_p1 + nx*j_m1 + nxy*k_p1,
			i_p1 + nx*j_p1 + nxy*k_m1,
			i_p1 + nx*j_p1 + nxy*k_p1  };


#pragma unroll
		// streaming and collision cycle (MOD2018), bounce back step
		//	Z. Guo, C. Zheng, B. Shi, Discrete lattice effects on the forcing term
		//	in the latticeBoltzmann method, Phys. Rev. E 65 (046308) (2002) 1.6.
		// 	force->moment update->equilibrium->output->forcing term->collision
		//					->streaming->new timestep->force ...

		// streaming
		for (int ii=0; ii<27; ii++) { fs[ii] = f[ii*nxyz + idg_lbm[ii]];}
		//mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);

		// boundary condition
		if(user_flags::flg_wallFunction==1){
			LBM_solid_slip_boundary(
					fs,
					f,
					l_obs,
					eD3Q27_x,
					eD3Q27_y,
					eD3Q27_z,
					idg_lbm,
					direction_lbm_slip,
					27,
					nx,ny,nz,nxyz
					);
		}else{
			LBM_solid_boundary (
					fs,
					f,
					l_obs,
					idg_lbm,
					direction_lbm,
					27,
					nxyz
					);
		}

		// calculate density (rho)
		mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
		// f,F -> u,v,w,rho (for calculation of collistion operator) //
		FLOAT us,vs,ws;
		mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);

		// wall treatment (ISHIBASHI 2019)
		FLOAT tau_wall_x = 0.0;
		FLOAT tau_wall_y = 0.0;
		FLOAT tau_wall_z = 0.0;
		FLOAT factor_WFB = 2.0; // inverse ratio of Wall Function Bounce (Han et al. 2019)
		FLOAT factor_BF = 2.0; // inverse ratio of Body Force

		if(user_flags::flg_wallFunction==1){
			// upload streamed u,v,w
			FLOAT u_bb_wall = us * c_ref;
			FLOAT v_bb_wall = vs * c_ref;
			FLOAT w_bb_wall = ws * c_ref;
			FLOAT U_bb_wall = sqrt(u_bb_wall*u_bb_wall + v_bb_wall*v_bb_wall + w_bb_wall*w_bb_wall);

			// wall detection
			int dir_n = 0;
			int dir_solid = 0;
			for(int ii=1; ii<=6; ii++){
				if(l_obs[idg_lbm[ii]] > 0.0){
					dir_solid = ii;
					dir_n += 1;
				}
			}

			FLOAT u_tau = 0.0;
			FLOAT u_neighbor = 0.0;
			FLOAT v_neighbor = 0.0;
			FLOAT w_neighbor = 0.0;
			FLOAT U_neighbor = 0.0;
			FLOAT tau_wall   = 0.0;

			if((dir_n == 1 || dir_n == 2) && l_obs[id_c0_c0_c0] < 0.0){
				// fetch velocity from opposite of solid grid
				u_neighbor = u_bb_wall;
				v_neighbor = v_bb_wall;
				w_neighbor = w_bb_wall;
				// ignore normal velocity
				if      (dir_solid==1 || dir_solid==2){
					u_neighbor = 0.0;
					u_bb_wall = 0.0;
				}else if(dir_solid==3 || dir_solid==4){
					v_neighbor = 0.0;
					v_bb_wall = 0.0;
				}else if(dir_solid==5 || dir_solid==6){
					w_neighbor = 0.0;
					w_bb_wall = 0.0;
				}
				// calculate u_tau from loglaw(U_neighbor, y)
				U_neighbor = sqrt(u_neighbor*u_neighbor + v_neighbor*v_neighbor + w_neighbor*w_neighbor);
				assert(U_neighbor >= 0.0);
				if(user_init::z0 > 0.0){ // loglaw
					u_tau = 0.4 * U_neighbor / log(dx*0.5/user_init::z0);
				}else{ // Spalding law
					u_tau = U_neighbor / 30.0;
					for(int c=0; c<30; c++){
						u_tau = u_tau - wall_func(u_tau, U_neighbor, dx*0.5) / d_wall_func_dut(u_tau, U_neighbor, dx*0.5);
					}
				}
				//calculate tau_wall
				tau_wall = coefficient::DENSITY_AIR * u_tau * u_tau;
				tau_wall_x = tau_wall * u_neighbor / U_neighbor;
				tau_wall_y = tau_wall * v_neighbor / U_neighbor;
				tau_wall_z = tau_wall * w_neighbor / U_neighbor;

				//// divide into component
				if(dir_solid==6 || isnan(tau_wall_x) || isnan(tau_wall_y) || isnan(tau_wall_z)){
					tau_wall_x = 0.0;
					tau_wall_y = 0.0;
					tau_wall_z = 0.0;
				}
				for(int ii=7; ii<NUM_DIRECTION_VEL; ii++){
					switch(dir_solid){
						case 1:	factor_WFB *= (__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
								fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/3.0/factor_WFB * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] > 0);
								fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/3.0/factor_WFB * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] > 0);
								break;                                                                                 
						case 2:	factor_WFB *= (__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
								fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/3.0/factor_WFB * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] < 0);
								fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/3.0/factor_WFB * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] < 0);
								break;                                                                                 
						case 3:	factor_WFB *= (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
								fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/3.0/factor_WFB * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] > 0);
								fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/3.0/factor_WFB * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] > 0);
								break;                                                                                 
						case 4:	factor_WFB *= (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
								fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/3.0/factor_WFB * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] < 0);
								fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/3.0/factor_WFB * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] < 0);
								break;                                                                                 
						case 5:	factor_WFB *= (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0));
								fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/3.0/factor_WFB * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] > 0);
								fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/3.0/factor_WFB * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] > 0);
								break;                                                                                 
						case 6:	factor_WFB *= (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0));
								fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/3.0/factor_WFB * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] < 0);
								fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/3.0/factor_WFB * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] < 0);
								break;
						default:break;
					}
				}
			}
		}

		// calculate density (rho)
		mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
		// f,F -> u,v,w,rho (for calculation of collistion operator) //
		mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);

		// force : nuoyancy:Boussinesq approximation
		const int id = id_c0_c0_c0;
		const int stride[3] = { 1, nx, nx*ny };
		const FLOAT  rho_gas = coefficient::DENSITY_AIR;
		const FLOAT  gravity = coefficient::GRAVITY;	
		const FLOAT  beta       = 1.0 / BASE_TEMPERATURE;
		const FLOAT  nondim_lbm = (1.0/c_ref/rho_gas) * dt;
		FLOAT  force_lbm[NUM_DIRECTION_VEL];
		FLOAT  force_x = 0.0;
		FLOAT  force_y = 0.0;
		FLOAT  force_z = 0.0;
		// pressure gradient
		if(user_flags::flg_dpdx==1)	{force_x = force_x - user_init::dpdx * nondim_lbm;}	//MOD2019
		if(user_flags::flg_dpdy==1)	{force_y = force_y - user_init::dpdy * nondim_lbm;}	//MOD2019
		if(user_flags::flg_coriolis==1) {
			const FLOAT pi=3.14159265;
			force_x = force_x + 2.0 * user_init::angular_velocity * vs * c_ref * sin(user_init::latitude/180.0 * pi) * nondim_lbm ;
			force_y = force_y - 2.0 * user_init::angular_velocity * us * c_ref * sin(user_init::latitude/180.0 * pi) * nondim_lbm ;
		}
		if(user_flags::flg_buoyancy==1) {
			force_z = force_z + rho_gas*fabs(gravity)*beta*(T[id] - BASE_TEMPERATURE) * nondim_lbm;
		}

		// wall treatment: add shear force;
		if(user_flags::flg_wallFunction==1){
			force_x = force_x - tau_wall_x / dx * nondim_lbm / factor_BF;
			force_y = force_y - tau_wall_y / dx * nondim_lbm / factor_BF;
			force_z = force_z - tau_wall_z / dx * nondim_lbm / factor_BF;
		}

		mathLib_LBM::
			Device_LBM_D3Q27_Force (
					rhos,
					force_x,
					force_y,
					force_z,
					force_lbm
					);

		// f,F -> u,v,w,rho (for calculation of collistion operator) //
		//	FLOAT us,vs,ws;
		//	mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);

		// Coriolis force
		//	const FLOAT     omega_rotation = 7.29212*1.0e-5;		// angular velocity of the earth [rad\s]
		//	const FLOAT	latitude = 35.681236;				// latuitude of Tokyo [rad]
		//	const FLOAT	f_cor1 = 0.0;					// Coriolis parameter in 1th direction
		//	const FLOAT	f_cor2 = 2.0 * omega_rotation * cos(latitude);	// Coriolis parameter in 2nd direction
		//	const FLOAT	f_cor3 = 2.0 * omega_rotation * sin(latitude);	// Coriolis parameter in 3rd direction
		//	force_x =   force_x + f_cor3 * vs;
		//	force_y = - force_y + f_cor3 * us;

		us = us + force_x*(FLOAT)0.5;
		vs = vs + force_y*(FLOAT)0.5;
		ws = ws + force_z*(FLOAT)0.5;

		// collision term : fs, force -> fs_deq=fs-fs_eq
		//	f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
		//	fs_deq = fs - feq
		mathLib_LBM::Device_LBM_D3Q27_d_equivalent_force(fs_deq, rhos, fs, force_x, force_y, force_z);

/*		// (YOKOUCHI 2020)
		if (user_flags::flg_particle == 1) {
			total_TKE[id_c0_c0_c0] = mathLib_LBM::Device_LBM_D3Q27_totalTKE(fs_deq, rhos);
		}	
*/		
		// OUTPUT
		u[id] = us;
		v[id] = vs;
		w[id] = ws;
		rho[id] = rhos;

		// source term force_x,y,z -> force_lbm(i)

		// sgs model : fs, feq -> csvis
		const FLOAT     csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
		
/*		// (YOKOUCHI 2020)
		if (user_flags::flg_particle == 1) {
			vis_sgs_old[id_c0_c0_c0] = vis_sgs[id_c0_c0_c0];
		}
*/		
		vis_sgs[id_c0_c0_c0] = csvis;

		// relaxation time
		const FLOAT     tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
		const FLOAT     omega     = (FLOAT)1.0/tau_total;               // 1.0/tau

		// cumulant LBM (collision process)
		// cumulant //
		FLOAT fs_cum[27], fIs_cum[27];
		FLOAT fIs[27];
		if(user_flags::flg_collision==1){

			fs_cum[0]  = fs[26];
			fs_cum[1]  = fs[18];
			fs_cum[2]  = fs[23];
			fs_cum[3]  = fs[14];
			fs_cum[4]  = fs[6];
			fs_cum[5]  = fs[13];
			fs_cum[6]  = fs[24];
			fs_cum[7]  = fs[17];
			fs_cum[8]  = fs[22];
			fs_cum[9]  = fs[10];
			fs_cum[10] = fs[4];
			fs_cum[11] = fs[9];
			fs_cum[12] = fs[2];
			fs_cum[13] = fs[0];
			fs_cum[14] = fs[1];
			fs_cum[15] = fs[8];
			fs_cum[16] = fs[3];
			fs_cum[17] = fs[7];
			fs_cum[18] = fs[25];
			fs_cum[19] = fs[16];
			fs_cum[20] = fs[21];
			fs_cum[21] = fs[12];
			fs_cum[22] = fs[5];
			fs_cum[23] = fs[11];
			fs_cum[24] = fs[20];
			fs_cum[25] = fs[15];
			fs_cum[26] = fs[19];

			FuncCumulantLBM::fs_cumulant_lbm(fIs_cum, fs_cum, omega, rhos, us,vs,ws);

			fIs[0]  = fIs_cum[13];
			fIs[1]  = fIs_cum[14];
			fIs[2]  = fIs_cum[12];
			fIs[3]  = fIs_cum[16];
			fIs[4]  = fIs_cum[10];
			fIs[5]  = fIs_cum[22];
			fIs[6]  = fIs_cum[4];
			fIs[7]  = fIs_cum[17];
			fIs[8]  = fIs_cum[15];
			fIs[9]  = fIs_cum[11];
			fIs[10] = fIs_cum[9];
			fIs[11] = fIs_cum[23];
			fIs[12] = fIs_cum[21];
			fIs[13] = fIs_cum[5];
			fIs[14] = fIs_cum[3];
			fIs[15] = fIs_cum[25];
			fIs[16] = fIs_cum[19];
			fIs[17] = fIs_cum[7];
			fIs[18] = fIs_cum[1];
			fIs[19] = fIs_cum[26];
			fIs[20] = fIs_cum[24];
			fIs[21] = fIs_cum[20];
			fIs[22] = fIs_cum[8];
			fIs[23] = fIs_cum[2];
			fIs[24] = fIs_cum[6];
			fIs[25] = fIs_cum[18];
			fIs[26] = fIs_cum[0];

			// cumulant //
		}

#pragma unroll
		// add collision and source terms
		//	fnew = fs - 1/tau * (fs - feq)
		if(user_flags::flg_collision==1){
			for (int ii=0; ii<27; ii++) {
				//	fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
				fn[nxyz*ii + id_c0_c0_c0] = fIs[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
			}
		}else{
			for (int ii=0; ii<27; ii++) {
				fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
			}
		}

		// thermal convection (MOD 2018) //

		if(user_flags::flg_scalar==0) { continue; }	//MOD2019

		if (l_obs[id] >= 0.0) { continue; }

		//    before 2019/06/04
		//	const FLOAT Tx = (u[id] >= 0.0) ? (T[id] - T[id-stride[0]])/dx : (T[id+stride[0]] - T[id])/dx;
		//	const FLOAT Ty = (v[id] >= 0.0) ? (T[id] - T[id-stride[1]])/dx : (T[id+stride[1]] - T[id])/dx;
		//	const FLOAT Tz = (w[id] >= 0.0) ? (T[id] - T[id-stride[2]])/dx : (T[id+stride[2]] - T[id])/dx;

		// boundary condition //
		const FLOAT HFxl = (l_obs[id-stride[0]] < 0.0) ? 0.0 : bcTe[id];
		const FLOAT HFyl = (l_obs[id-stride[1]] < 0.0) ? 0.0 : bcTs[id];
		const FLOAT HFzl = (l_obs[id-stride[2]] < 0.0) ? 0.0 : bcTr[id];

		const FLOAT HFxr = (l_obs[id+stride[0]] < 0.0) ? 0.0 : bcTw[id];
		const FLOAT HFyr = (l_obs[id+stride[1]] < 0.0) ? 0.0 : bcTn[id];
		const FLOAT HFzr = (l_obs[id+stride[2]] < 0.0) ? 0.0 : 0.0;

		// diffusion //
		const FLOAT Txl = (l_obs[id-stride[0]] < 0.0) ? (T[id] - T[id-stride[0]])/dx : 0.0;
		const FLOAT Tyl = (l_obs[id-stride[1]] < 0.0) ? (T[id] - T[id-stride[1]])/dx : 0.0;
		const FLOAT Tzl = (l_obs[id-stride[2]] < 0.0) ? (T[id] - T[id-stride[2]])/dx : 0.0;

		const FLOAT Txr = (l_obs[id+stride[0]] < 0.0) ? (T[id+stride[0]] - T[id])/dx : 0.0;
		const FLOAT Tyr = (l_obs[id+stride[1]] < 0.0) ? (T[id+stride[1]] - T[id])/dx : 0.0;
		const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0; // MIKAEL 2026 Change 0.0 to Tzl to test buoyancy break or not

		//	const FLOAT uTx = (u[id] >= 0.0) ? u[id]*(T[id] - T[id-stride[0]])/dx : u[id]*(T[id+stride[0]] - T[id])/dx;
		//	const FLOAT vTy = (v[id] >= 0.0) ? v[id]*(T[id] - T[id-stride[1]])/dx : v[id]*(T[id+stride[1]] - T[id])/dx;
		//	const FLOAT wTz = (w[id] >= 0.0) ? w[id]*(T[id] - T[id-stride[2]])/dx : w[id]*(T[id+stride[2]] - T[id])/dx;
		const FLOAT uTx = ( 0.5*(u[id]+u[id-stride[0]])*Txl + 0.5*(u[id]+u[id+stride[0]])*Txr ) * (FLOAT)0.5;
		const FLOAT vTy = ( 0.5*(v[id]+v[id-stride[1]])*Tyl + 0.5*(v[id]+v[id+stride[1]])*Tyr ) * (FLOAT)0.5;
		const FLOAT wTz = ( 0.5*(w[id]+w[id-stride[2]])*Tzl + 0.5*(w[id]+w[id+stride[2]])*Tzr ) * (FLOAT)0.5;

		const FLOAT Txx = (Txr - Txl) / (dx);
		const FLOAT Tyy = (Tyr - Tyl) / (dx);
		const FLOAT Tzz = (Tzr - Tzl) / (dx);

		const FLOAT coefT = COEF_THERMAL + csvis / PR_SGS * c_ref * (dx);        //* 5.0;       // MOD 2018

		const FLOAT adv_term = - (uTx + vTy + wTz)*c_ref;
		//const FLOAT adv_term  = - ( u[id]*Tx + v[id]*Ty + w[id]*Tz )*c_ref;
		const FLOAT diff_term = coefT * ( Txx + Tyy + Tzz ) + (HFxr + HFxl)/dx + (HFyr + HFyl)/dx + (HFzr + HFzl)/dx;

		Tn[id] = T[id] + (adv_term + diff_term)*dt;


		/*
		// heat flux
		const FLOAT HFxl = (l_obs[id-stride[0]] < 0.0) ? 0.0 : bcTe[id];
		const FLOAT HFyl = (l_obs[id-stride[1]] < 0.0) ? 0.0 : bcTs[id];
		const FLOAT HFzl = (l_obs[id-stride[2]] < 0.0) ? 0.0 : bcTr[id];
		const FLOAT HFxr = (l_obs[id+stride[0]] < 0.0) ? 0.0 : bcTw[id];
		const FLOAT HFyr = (l_obs[id+stride[1]] < 0.0) ? 0.0 : bcTn[id];
		const FLOAT HFzr = (l_obs[id+stride[2]] < 0.0) ? 0.0 : 0.0;
		// advection
		const FLOAT Fxl = (l_obs[id-stride[0]] < 0.0) ? 0.5 * ( u[id]+u[id-stride[0]] ) * 0.5 * ( T[id]+T[id-stride[0]] ) : 0.0;
		const FLOAT Fyl = (l_obs[id-stride[1]] < 0.0) ? 0.5 * ( v[id]+v[id-stride[1]] ) * 0.5 * ( T[id]+T[id-stride[1]] ) : 0.0;
		const FLOAT Fzl = (l_obs[id-stride[2]] < 0.0) ? 0.5 * ( w[id]+w[id-stride[2]] ) * 0.5 * ( T[id]+T[id-stride[2]] ) : 0.0;
		const FLOAT Fxr = (l_obs[id+stride[0]] < 0.0) ? 0.5 * ( u[id+stride[0]]+u[id] ) * 0.5 * ( T[id+stride[0]]+T[id] ) : 0.0;
		const FLOAT Fyr = (l_obs[id+stride[1]] < 0.0) ? 0.5 * ( v[id+stride[1]]+v[id] ) * 0.5 * ( T[id+stride[1]]+T[id] ) : 0.0;
		const FLOAT Fzr = (l_obs[id+stride[2]] < 0.0) ? 0.5 * ( w[id+stride[2]]+w[id] ) * 0.5 * ( T[id+stride[2]]+T[id] ) : 0.0;
		// diffusion
		const FLOAT Txl = (l_obs[id-stride[0]] < 0.0) ? (T[id] - T[id-stride[0]])/dx : 0.0;
		const FLOAT Tyl = (l_obs[id-stride[1]] < 0.0) ? (T[id] - T[id-stride[1]])/dx : 0.0;
		const FLOAT Tzl = (l_obs[id-stride[2]] < 0.0) ? (T[id] - T[id-stride[2]])/dx : 0.0;
		const FLOAT Txr = (l_obs[id+stride[0]] < 0.0) ? (T[id+stride[0]] - T[id])/dx : 0.0;
		const FLOAT Tyr = (l_obs[id+stride[1]] < 0.0) ? (T[id+stride[1]] - T[id])/dx : 0.0;
		const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;
		const FLOAT Txx = (Txr - Txl) / (dx);
		const FLOAT Tyy = (Tyr - Tyl) / (dx);
		const FLOAT Tzz = (Tzr - Tzl) / (dx);
		const FLOAT coefT = COEF_THERMAL + csvis / PR_SGS * c_ref * (dx);        //* 5.0;       // MOD 2018

		const FLOAT diff_term = coefT * ( Txx + Tyy + Tzz ) + (HFxr + HFxl)/dx + (HFyr + HFyl)/dx + (HFzr + HFzl)/dx + (-Fxr + Fxl)/dx + (-Fyr + Fyl)/dx + (-Fzr + Fzl)/dx;

		Tn[id] = T[id] + diff_term*dt;
		 */
	}
}

//__global__ void
//cuda_stream_collision_T_D3Q27_wall (
//		const FLOAT	*f,
//		FLOAT	*fn,
//		const FLOAT	*T,
//		FLOAT	*Tn,
//		const FLOAT	*l_obs,
//		const int *l_obs_x,
//		const int *l_obs_y,
//		const int *l_obs_z,
//		FLOAT	*u,
//		FLOAT	*v,
//		FLOAT	*w,
//		FLOAT *rho,
//		const FLOAT	*vis,
//		FLOAT	*vis_sgs,
//		const FLOAT* T_ref, // MOD2018 Boussinesq
//		const FLOAT *Fcs,
//		const FLOAT* bcTw, //  1,  0,  0 //
//		const FLOAT* bcTe, // -1,  0,  0 //
//		const FLOAT* bcTs, //  0,  1,  0 //
//		const FLOAT* bcTn, //  0, -1,  0 //
//		const FLOAT* bcTr, //  0,  0,  1 //
//		const FLOAT dx,
//		const FLOAT dt,
//
//		const FLOAT	c_ref,
//		int nx, int ny, int nz,
//		int halo
//		)
//{
//	// cuda index
//	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
//	id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
//	id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);
//
//	// D3Q27:
//	// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
//	// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
//	// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
//	// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
//	// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//	//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)
//	const int eD3Q27_x[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1, 1, 1,-1,-1,-1 };
//	const int eD3Q27_y[27] = { 0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1 };
//	const int eD3Q27_z[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1 };
//	//const FLOAT l1 = 1;
//	//const FLOAT l2 = (FLOAT)sqrtf(2);
//	//const FLOAT l3 = (FLOAT)sqrtf(3);
//	//const FLOAT eD3Q27_l[27]={ 0,l1,l1,l1,l1,l1,l1,l2,l2,l2,l2,l2,l2,l2,l2,l2,l2,l2,l2,l3,l3,l3,l3,l3,l3,l3,l3 };
//	//const FLOAT w1 = (FLOAT)8.0/(FLOAT)27.0;
//	//const FLOAT w2 = (FLOAT)2.0/(FLOAT)27.0;
//	//const FLOAT w3 = (FLOAT)1.0/(FLOAT)54.0;
//	//const FLOAT w4 = (FLOAT)1.0/(FLOAT)216.0;
//	//const FLOAT eD3Q27_w[27]={w1,w2,w2,w2,w2,w2,w2,w3,w3,w3,w3,w3,w3,w3,w3,w3,w3,w3,w3,w4,w4,w4,w4,w4,w4,w4,w4 };
//
//	// global index
//	// lbm stream from
//	// local id (e[ii] : depature point )
//	//							      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}
//	const int direction_lbm[27]     = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,14,13,12,11,18,17,16,15,26,23,24,25,20,21,22,19};
//	const int direction_lbm_x[27]   = {0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,18,17,16,15,20,19,25,24,26,22,21,23};
//	const int direction_lbm_y[27]   = {0, 2, 1, 4, 3, 6, 5, 9,10, 7, 8,14,13,12,11,16,15,18,17,21,25,19,23,22,26,20,24};
//	const int direction_lbm_z[27]   = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,13,14,11,12,17,18,15,16,22,24,23,19,21,20,26,25};
//	const int direction_lbm_xy[27]  = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,12,11,14,13,16,15,18,17,25,21,20,26,24,23,19,22};
//	const int direction_lbm_xz[27]  = {0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,14,13,12,11,17,18,15,16,24,22,26,20,25,19,23,21};
//	const int direction_lbm_yz[27]  = {0, 2, 1, 4, 3, 6, 5, 9,10, 7, 8,13,14,11,12,18,17,16,15,23,26,22,21,19,25,24,20};
//	//const int direction_lbm_xyz[27] = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,14,13,12,11,18,17,16,15,26,23,24,25,20,21,22,19};
//	const int direction_lbm_xyz[27] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26};
//	const int *direction_lbm_slip[8] = {direction_lbm, direction_lbm_x, direction_lbm_y, direction_lbm_xy, 
//		direction_lbm_z, direction_lbm_xz, direction_lbm_yz, direction_lbm_xyz};
//
//
//	// index local
//	const int	i_m1 = id_x-1,
//		  i_p1 = id_x+1;
//	const int	j_m1 = id_y-1,
//		  j_p1 = id_y+1;
//	const int	nxy = nx*ny;
//
//	// variables //
//	const int	nxyz = nx*ny*nz;
//	FLOAT	rhos;	// variables after streaming step
//	FLOAT	fs[27], fs_deq[27];
//
//	int k__=0;
//	for (int k=0; k<BLOCKDIM_Z; k++) {
//		k__ = k__ + 1;
//		const int	id_z = id_zs + k;
//
//		const int	k_m1 = id_z - 1;
//		const int	k_p1 = id_z + 1;
//
//		// index
//		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;
//
//		// center (lbm)
//		const int	idg_lbm[27] = {
//			id_x + nx*id_y + nxy*id_z,
//			i_m1 + nx*id_y + nxy*id_z,
//			i_p1 + nx*id_y + nxy*id_z,
//			id_x + nx*j_m1 + nxy*id_z,
//			id_x + nx*j_p1 + nxy*id_z,
//			id_x + nx*id_y + nxy*k_m1,
//			id_x + nx*id_y + nxy*k_p1,
//			i_m1 + nx*j_m1 + nxy*id_z,
//			i_p1 + nx*j_m1 + nxy*id_z,
//			i_m1 + nx*j_p1 + nxy*id_z,
//			i_p1 + nx*j_p1 + nxy*id_z,
//			i_m1 + nx*id_y + nxy*k_m1,
//			i_p1 + nx*id_y + nxy*k_m1,
//			i_m1 + nx*id_y + nxy*k_p1,
//			i_p1 + nx*id_y + nxy*k_p1,
//			id_x + nx*j_m1 + nxy*k_m1,
//			id_x + nx*j_p1 + nxy*k_m1,
//			id_x + nx*j_m1 + nxy*k_p1,
//			id_x + nx*j_p1 + nxy*k_p1,
//			i_m1 + nx*j_m1 + nxy*k_m1, // 19 ~ 26
//			i_p1 + nx*j_m1 + nxy*k_m1,
//			i_m1 + nx*j_p1 + nxy*k_m1,
//			i_m1 + nx*j_m1 + nxy*k_p1,
//			i_m1 + nx*j_p1 + nxy*k_p1,
//			i_p1 + nx*j_m1 + nxy*k_p1,
//			i_p1 + nx*j_p1 + nxy*k_m1,
//			i_p1 + nx*j_p1 + nxy*k_p1  };
//
//
//#pragma unroll
//		// streaming and collision cycle (MOD2018), bounce back step
//		//	Z. Guo, C. Zheng, B. Shi, Discrete lattice effects on the forcing term
//		//	in the latticeBoltzmann method, Phys. Rev. E 65 (046308) (2002) 1.6.
//		// 	force->moment update->equilibrium->output->forcing term->collision
//		//					->streaming->new timestep->force ...
//
//		// streaming
//		//for (int ii=0; ii<27; ii++) { fs[ii] = f[ii*nxyz + idg_lbm[ii]];}
//		//mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
//
//		// boundary condition
//		//LBM_solid_boundary (
//		//                    fs,
//		//                    f,
//		//                    l_obs,
//		//                    idg_lbm,
//		//                    direction_lbm,
//		//                    27,
//		//                    nxyz
//		//                    );
//		LBM_solid_slip_boundary(
//				fs,
//				f,
//				l_obs,
//				eD3Q27_x,
//				eD3Q27_y,
//				eD3Q27_z,
//				idg_lbm,
//				direction_lbm_slip,
//				27,
//				nx,ny,nz,nxyz
//				);
//
//		// calculate density (rho)
//		mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
//		// f,F -> u,v,w,rho (for calculation of collistion operator) //
//		FLOAT us,vs,ws;
//		mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);
//
//		// wall treatment (ISHIBASHI 2019)
//		// upload streamed u,v,w
//
//		FLOAT u_bb_wall = us * c_ref;
//		FLOAT v_bb_wall = vs * c_ref;
//		FLOAT w_bb_wall = ws * c_ref;
//		FLOAT U_bb_wall = sqrt(u_bb_wall*u_bb_wall + v_bb_wall*v_bb_wall + w_bb_wall*w_bb_wall);
//
//		// wall detection
//		int dir_n = 0;
//		int dir_solid = 0;
//		for(int ii=1; ii<=6; ii++){
//			if(l_obs[idg_lbm[ii]] > 0.0){
//				dir_solid = ii;
//				dir_n += 1;
//			}
//		}
//
//		FLOAT u_tau = 0.0;
//		FLOAT u_neighbor = 0.0;
//		FLOAT v_neighbor = 0.0;
//		FLOAT w_neighbor = 0.0;
//		FLOAT U_neighbor = 0.0;
//		//FLOAT u_loglaw   = 0.0;
//		//FLOAT v_loglaw   = 0.0;
//		//FLOAT w_loglaw   = 0.0;
//		//FLOAT U_loglaw   = 0.0;
//		FLOAT tau_wall_x = 0.0;
//		FLOAT tau_wall_y = 0.0;
//		FLOAT tau_wall_z = 0.0;
//		FLOAT tau_wall   = 0.0;
//
//		if(dir_n == 1 && l_obs[id_c0_c0_c0] < 0.0){
//			// fetch velocity from opposite of solid grid
//			u_neighbor = u_bb_wall;
//			v_neighbor = v_bb_wall;
//			w_neighbor = w_bb_wall;
//			// ignore normal velocity
//			if      (dir_solid==1 || dir_solid==2){
//				u_neighbor = 0.0;
//				u_bb_wall = 0.0;
//			}else if(dir_solid==3 || dir_solid==4){
//				v_neighbor = 0.0;
//				v_bb_wall = 0.0;
//			}else if(dir_solid==5 || dir_solid==6){
//				w_neighbor = 0.0;
//				w_bb_wall = 0.0;
//			}
//			// calculate u_tau from loglaw(U_neighbor, y)
//			U_neighbor = sqrt(u_neighbor*u_neighbor + v_neighbor*v_neighbor + w_neighbor*w_neighbor);
//			assert(U_neighbor >= 0.0);
//			//u_tau = 0.4 * U_neighbor / log(dx*0.5/user_init::z0);
//			u_tau = U_neighbor / 30.0;
//			for(int c=0; c<30; c++){
//				u_tau = u_tau - wall_func(u_tau, U_neighbor, dx*0.5) / d_wall_func_dut(u_tau, U_neighbor, dx*0.5);
//			}
//			//assert(u_tau>=0.0);
//			//calculate tau_wall
//			tau_wall = coefficient::DENSITY_AIR * u_tau * u_tau;
//			//assert(tau_wall >= 0.0);
//			tau_wall_x = tau_wall * u_neighbor / U_neighbor;
//			tau_wall_y = tau_wall * v_neighbor / U_neighbor;
//			tau_wall_z = tau_wall * w_neighbor / U_neighbor;
//
//			// calculate U_loglaw form loglaw(u_tau, y)
//			//U_loglaw = U_neighbor;
//			//for(int c=0; c<100; c++){
//			//	U_loglaw = U_loglaw - wall_func(u_tau, U_loglaw, dx*0.5f) / d_wall_func_dU(u_tau, U_loglaw, dx*0.5f);
//			//}
//			//// divide into component
//			//u_loglaw = U_loglaw * u_neighbor / U_neighbor;
//			//v_loglaw = U_loglaw * v_neighbor / U_neighbor;
//			//w_loglaw = U_loglaw * w_neighbor / U_neighbor;
//			//tau_wall_x = u_bb_wall - u_loglaw;
//			//tau_wall_y = v_bb_wall - v_loglaw;
//			//tau_wall_z = w_bb_wall - w_loglaw;
//			if(dir_solid==6 || isnan(tau_wall_x) || isnan(tau_wall_y) || isnan(tau_wall_z)){
//				tau_wall_x = 0.0;
//				tau_wall_y = 0.0;
//				tau_wall_z = 0.0;
//			}
//			//assert(!isnan(tau_wall_x));
//			//assert(!isnan(tau_wall_y));
//			//assert(!isnan(tau_wall_z));
//			for(int ii=7; ii<NUM_DIRECTION_VEL; ii++){
//				FLOAT factor = 0.0;
//				switch(dir_solid){
//					case 1:	factor = (__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
//							fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/6.0/factor * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] > 0);
//							fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/6.0/factor * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] > 0);
//							break;                                                                                 
//					case 2:	factor = (__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_y[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
//							fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/6.0/factor * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] < 0);
//							fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/6.0/factor * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_x[ii] < 0);
//							break;                                                                                 
//					case 3:	factor = (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
//							fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/6.0/factor * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] > 0);
//							fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/6.0/factor * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] > 0);
//							break;                                                                                 
//					case 4:	factor = (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_z[ii],0,0));
//							fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/6.0/factor * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] < 0);
//							fs[ii] = fs[ii] -eD3Q27_z[ii]*1.0/6.0/factor * tau_wall_z /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_y[ii] < 0);
//							break;                                                                                 
//					case 5:	factor = (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0));
//							fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/6.0/factor * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] > 0);
//							fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/6.0/factor * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] > 0);
//							break;                                                                                 
//					case 6:	factor = (__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0))*(__sad(eD3Q27_x[ii],0,0)+__sad(eD3Q27_y[ii],0,0));
//							fs[ii] = fs[ii] -eD3Q27_x[ii]*1.0/6.0/factor * tau_wall_x /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] < 0);
//							fs[ii] = fs[ii] -eD3Q27_y[ii]*1.0/6.0/factor * tau_wall_y /coefficient::DENSITY_AIR/c_ref/c_ref*(eD3Q27_z[ii] < 0);
//							break;
//					default:break;
//				}
//			}
//		}
//
//		// calculate density (rho)
//		mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
//		// f,F -> u,v,w,rho (for calculation of collistion operator) //
//		mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);
//
//		// force : nuoyancy:Boussinesq approximation
//		const int id = id_c0_c0_c0;
//		const int stride[3] = { 1, nx, nx*ny };
//		const FLOAT  rho_gas = coefficient::DENSITY_AIR;
//		const FLOAT  gravity = coefficient::GRAVITY;	
//		const FLOAT  beta       = 1.0 / BASE_TEMPERATURE;
//		const FLOAT  nondim_lbm = (1.0/c_ref/rho_gas) * dt;
//		FLOAT  force_lbm[NUM_DIRECTION_VEL];
//		FLOAT  force_x = 0.0;
//		FLOAT  force_y = 0.0;
//		FLOAT  force_z = 0.0;
//		// pressure gradient
//		//if(id_x < 60 + 1){
//		if(user_flags::flg_dpdx==1)	{force_x = force_x - user_init::dpdx * nondim_lbm;}	//MOD2019
//		if(user_flags::flg_dpdy==1)	{force_y = force_y - user_init::dpdy * nondim_lbm;}	//MOD2019
//		//}
//		if(user_flags::flg_coriolis==1) {
//			const FLOAT pi=3.14159265;
//			force_x = force_x + 2.0 * user_init::angular_velocity * vs * c_ref * sin(user_init::latitude/180.0 * pi) * nondim_lbm ;
//			force_y = force_y - 2.0 * user_init::angular_velocity * us * c_ref * sin(user_init::latitude/180.0 * pi) * nondim_lbm ;
//		}
//		// buoyancy
//		if(user_flags::flg_buoyancy==1) {force_z = force_z + rho_gas*fabs(gravity)*beta*(T[id] - BASE_TEMPERATURE) * nondim_lbm;}
//
//		// wall treatment: add shear force;
//		force_x = force_x - tau_wall_x / dx * nondim_lbm / 2.0;
//		force_y = force_y - tau_wall_y / dx * nondim_lbm / 2.0;
//		force_z = force_z - tau_wall_z / dx * nondim_lbm / 2.0;
//
//		mathLib_LBM::
//			Device_LBM_D3Q27_Force (
//					rhos,
//					force_x,
//					force_y,
//					force_z,
//					force_lbm
//					);
//
//		// f,F -> u,v,w,rho (for calculation of collistion operator) //
//		//	FLOAT us,vs,ws;
//		//	mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);
//
//		// Coriolis force
//		//	const FLOAT     omega_rotation = 7.29212*1.0e-5;		// angular velocity of the earth [rad\s]
//		//	const FLOAT	latitude = 35.681236;				// latuitude of Tokyo [rad]
//		//	const FLOAT	f_cor1 = 0.0;					// Coriolis parameter in 1th direction
//		//	const FLOAT	f_cor2 = 2.0 * omega_rotation * cos(latitude);	// Coriolis parameter in 2nd direction
//		//	const FLOAT	f_cor3 = 2.0 * omega_rotation * sin(latitude);	// Coriolis parameter in 3rd direction
//		//	force_x =   force_x + f_cor3 * vs;
//		//	force_y = - force_y + f_cor3 * us;
//
//		us = us + force_x*(FLOAT)0.5;
//		vs = vs + force_y*(FLOAT)0.5;
//		ws = ws + force_z*(FLOAT)0.5;
//
//		// collision term : fs, force -> fs_deq=fs-fs_eq
//		//	f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
//		//	fs_deq = fs - feq
//		mathLib_LBM::Device_LBM_D3Q27_d_equivalent_force(fs_deq, rhos, fs, force_x, force_y, force_z);
//
//		// OUTPUT
//		u[id] = us;
//		v[id] = vs;
//		w[id] = ws;
//		rho[id] = rhos;
//
//		// source term force_x,y,z -> force_lbm(i)
//
//		// sgs model : fs, feq -> csvis
//		const FLOAT     csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
//		vis_sgs[id_c0_c0_c0] = csvis;
//
//		// relaxation time
//		const FLOAT     tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
//		const FLOAT     omega     = (FLOAT)1.0/tau_total;               // 1.0/tau
//
//#pragma unroll
//		// add collision and source terms
//		//	fnew = fs - 1/tau * (fs - feq)
//		for (int ii=0; ii<27; ii++) {
//			fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
//		}
//
//
//		// thermal convection (MOD 2018) //
//
//		if(user_flags::flg_scalar==0) { continue; }	//MOD2019
//
//		if (l_obs[id] >= 0.0) { continue; }
//
//		//    before 2019/06/04
//		//	const FLOAT Tx = (u[id] >= 0.0) ? (T[id] - T[id-stride[0]])/dx : (T[id+stride[0]] - T[id])/dx;
//		//	const FLOAT Ty = (v[id] >= 0.0) ? (T[id] - T[id-stride[1]])/dx : (T[id+stride[1]] - T[id])/dx;
//		//	const FLOAT Tz = (w[id] >= 0.0) ? (T[id] - T[id-stride[2]])/dx : (T[id+stride[2]] - T[id])/dx;
//
//		// boundary condition //
//		const FLOAT HFxl = (l_obs[id-stride[0]] < 0.0) ? 0.0 : bcTe[id];
//		const FLOAT HFyl = (l_obs[id-stride[1]] < 0.0) ? 0.0 : bcTs[id];
//		const FLOAT HFzl = (l_obs[id-stride[2]] < 0.0) ? 0.0 : bcTr[id];
//
//		const FLOAT HFxr = (l_obs[id+stride[0]] < 0.0) ? 0.0 : bcTw[id];
//		const FLOAT HFyr = (l_obs[id+stride[1]] < 0.0) ? 0.0 : bcTn[id];
//		const FLOAT HFzr = (l_obs[id+stride[2]] < 0.0) ? 0.0 : 0.0;
//
//		// diffusion //
//		const FLOAT Txl = (l_obs[id-stride[0]] < 0.0) ? (T[id] - T[id-stride[0]])/dx : 0.0;
//		const FLOAT Tyl = (l_obs[id-stride[1]] < 0.0) ? (T[id] - T[id-stride[1]])/dx : 0.0;
//		const FLOAT Tzl = (l_obs[id-stride[2]] < 0.0) ? (T[id] - T[id-stride[2]])/dx : 0.0;
//
//		const FLOAT Txr = (l_obs[id+stride[0]] < 0.0) ? (T[id+stride[0]] - T[id])/dx : 0.0;
//		const FLOAT Tyr = (l_obs[id+stride[1]] < 0.0) ? (T[id+stride[1]] - T[id])/dx : 0.0;
//		const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;
//
//		//	const FLOAT uTx = (u[id] >= 0.0) ? u[id]*(T[id] - T[id-stride[0]])/dx : u[id]*(T[id+stride[0]] - T[id])/dx;
//		//	const FLOAT vTy = (v[id] >= 0.0) ? v[id]*(T[id] - T[id-stride[1]])/dx : v[id]*(T[id+stride[1]] - T[id])/dx;
//		//	const FLOAT wTz = (w[id] >= 0.0) ? w[id]*(T[id] - T[id-stride[2]])/dx : w[id]*(T[id+stride[2]] - T[id])/dx;
//		const FLOAT uTx = ( 0.5*(u[id]+u[id-stride[0]])*Txl + 0.5*(u[id]+u[id+stride[0]])*Txr ) * (FLOAT)0.5;
//		const FLOAT vTy = ( 0.5*(v[id]+v[id-stride[1]])*Tyl + 0.5*(v[id]+v[id+stride[1]])*Tyr ) * (FLOAT)0.5;
//		const FLOAT wTz = ( 0.5*(w[id]+w[id-stride[2]])*Tzl + 0.5*(w[id]+w[id+stride[2]])*Tzr ) * (FLOAT)0.5;
//
//		const FLOAT Txx = (Txr - Txl) / (dx);
//		const FLOAT Tyy = (Tyr - Tyl) / (dx);
//		const FLOAT Tzz = (Tzr - Tzl) / (dx);
//
//		const FLOAT coefT = COEF_THERMAL + csvis / PR_SGS * c_ref * (dx);        //* 5.0;       // MOD 2018
//
//		const FLOAT adv_term = - (uTx + vTy + wTz)*c_ref;
//		//const FLOAT adv_term  = - ( u[id]*Tx + v[id]*Ty + w[id]*Tz )*c_ref;
//		const FLOAT diff_term = coefT * ( Txx + Tyy + Tzz ) + (HFxr + HFxl)/dx + (HFyr + HFyl)/dx + (HFzr + HFzl)/dx;
//
//		Tn[id] = T[id] + (adv_term + diff_term)*dt;
//
//
//		/*
//		// heat flux
//		const FLOAT HFxl = (l_obs[id-stride[0]] < 0.0) ? 0.0 : bcTe[id];
//		const FLOAT HFyl = (l_obs[id-stride[1]] < 0.0) ? 0.0 : bcTs[id];
//		const FLOAT HFzl = (l_obs[id-stride[2]] < 0.0) ? 0.0 : bcTr[id];
//		const FLOAT HFxr = (l_obs[id+stride[0]] < 0.0) ? 0.0 : bcTw[id];
//		const FLOAT HFyr = (l_obs[id+stride[1]] < 0.0) ? 0.0 : bcTn[id];
//		const FLOAT HFzr = (l_obs[id+stride[2]] < 0.0) ? 0.0 : 0.0;
//		// advection
//		const FLOAT Fxl = (l_obs[id-stride[0]] < 0.0) ? 0.5 * ( u[id]+u[id-stride[0]] ) * 0.5 * ( T[id]+T[id-stride[0]] ) : 0.0;
//		const FLOAT Fyl = (l_obs[id-stride[1]] < 0.0) ? 0.5 * ( v[id]+v[id-stride[1]] ) * 0.5 * ( T[id]+T[id-stride[1]] ) : 0.0;
//		const FLOAT Fzl = (l_obs[id-stride[2]] < 0.0) ? 0.5 * ( w[id]+w[id-stride[2]] ) * 0.5 * ( T[id]+T[id-stride[2]] ) : 0.0;
//		const FLOAT Fxr = (l_obs[id+stride[0]] < 0.0) ? 0.5 * ( u[id+stride[0]]+u[id] ) * 0.5 * ( T[id+stride[0]]+T[id] ) : 0.0;
//		const FLOAT Fyr = (l_obs[id+stride[1]] < 0.0) ? 0.5 * ( v[id+stride[1]]+v[id] ) * 0.5 * ( T[id+stride[1]]+T[id] ) : 0.0;
//		const FLOAT Fzr = (l_obs[id+stride[2]] < 0.0) ? 0.5 * ( w[id+stride[2]]+w[id] ) * 0.5 * ( T[id+stride[2]]+T[id] ) : 0.0;
//		// diffusion
//		const FLOAT Txl = (l_obs[id-stride[0]] < 0.0) ? (T[id] - T[id-stride[0]])/dx : 0.0;
//		const FLOAT Tyl = (l_obs[id-stride[1]] < 0.0) ? (T[id] - T[id-stride[1]])/dx : 0.0;
//		const FLOAT Tzl = (l_obs[id-stride[2]] < 0.0) ? (T[id] - T[id-stride[2]])/dx : 0.0;
//		const FLOAT Txr = (l_obs[id+stride[0]] < 0.0) ? (T[id+stride[0]] - T[id])/dx : 0.0;
//		const FLOAT Tyr = (l_obs[id+stride[1]] < 0.0) ? (T[id+stride[1]] - T[id])/dx : 0.0;
//		const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;
//		const FLOAT Txx = (Txr - Txl) / (dx);
//		const FLOAT Tyy = (Tyr - Tyl) / (dx);
//		const FLOAT Tzz = (Tzr - Tzl) / (dx);
//		const FLOAT coefT = COEF_THERMAL + csvis / PR_SGS * c_ref * (dx);        //* 5.0;       // MOD 2018
//
//		const FLOAT diff_term = coefT * ( Txx + Tyy + Tzz ) + (HFxr + HFxl)/dx + (HFyr + HFyl)/dx + (HFzr + HFzl)/dx + (-Fxr + Fxl)/dx + (-Fyr + Fyl)/dx + (-Fzr + Fzl)/dx;
//
//		Tn[id] = T[id] + diff_term*dt;
//		 */
//	}
//}

//__global__ void
//cuda_stream_collision_T_D3Q27_cum (
//		const FLOAT	*f,
//		FLOAT	*fn,
//		const FLOAT	*T,
//		FLOAT	*Tn,
//		const FLOAT	*l_obs,
//		FLOAT	*u,
//		FLOAT	*v,
//		FLOAT	*w,
//		FLOAT *rho,
//		const FLOAT	*vis,
//		FLOAT	*vis_sgs,
//		const FLOAT* T_ref, // MOD2018 Boussinesq
//		const FLOAT *Fcs,
//		const FLOAT* bcTw, //  1,  0,  0 //
//		const FLOAT* bcTe, // -1,  0,  0 //
//		const FLOAT* bcTs, //  0,  1,  0 //
//		const FLOAT* bcTn, //  0, -1,  0 //
//		const FLOAT* bcTr, //  0,  0,  1 //
//		const FLOAT dx,
//		const FLOAT dt,
//
//		const FLOAT	c_ref,
//		int nx, int ny, int nz,
//		int halo
//		)
//{
//	// cuda index
//	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
//	id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
//	id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);
//
//	// D3Q27:
//	// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
//	// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
//	// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
//	// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
//	// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//	//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)
//
//
//	// global index
//	// lbm stream from
//	// local id (e[ii] : depature point )
//	const int	direction_lbm[27] = {
//		0, 2, 1, 4, 3, 6, 5,
//		10,  9,  8,  7,
//		14, 13, 12, 11,
//		18, 17, 16, 15,
//		26, 23, 24, 25,
//		20, 21, 22, 19 	};
//
//
//	// index local
//	const int	i_m1 = id_x-1,
//		  i_p1 = id_x+1;
//	const int	j_m1 = id_y-1,
//		  j_p1 = id_y+1;
//	const int	nxy = nx*ny;
//
//	// variables //
//	const int	nxyz = nx*ny*nz;
//	FLOAT	rhos;	// variables after streaming step
//	FLOAT	fs[27], fs_deq[27];
//
//	int k__=0;
//	for (int k=0; k<BLOCKDIM_Z; k++) {
//		k__ = k__ + 1;
//		const int	id_z = id_zs + k;
//
//		const int	k_m1 = id_z - 1;
//		const int	k_p1 = id_z + 1;
//
//		// index
//		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;
//
//		// center (lbm)
//		const int	idg_lbm[27] = {
//			id_x + nx*id_y + nxy*id_z,
//			i_m1 + nx*id_y + nxy*id_z,
//			i_p1 + nx*id_y + nxy*id_z,
//			id_x + nx*j_m1 + nxy*id_z,
//			id_x + nx*j_p1 + nxy*id_z,
//			id_x + nx*id_y + nxy*k_m1,
//			id_x + nx*id_y + nxy*k_p1,
//			i_m1 + nx*j_m1 + nxy*id_z,
//			i_p1 + nx*j_m1 + nxy*id_z,
//			i_m1 + nx*j_p1 + nxy*id_z,
//			i_p1 + nx*j_p1 + nxy*id_z,
//			i_m1 + nx*id_y + nxy*k_m1,
//			i_p1 + nx*id_y + nxy*k_m1,
//			i_m1 + nx*id_y + nxy*k_p1,
//			i_p1 + nx*id_y + nxy*k_p1,
//			id_x + nx*j_m1 + nxy*k_m1,
//			id_x + nx*j_p1 + nxy*k_m1,
//			id_x + nx*j_m1 + nxy*k_p1,
//			id_x + nx*j_p1 + nxy*k_p1,
//			i_m1 + nx*j_m1 + nxy*k_m1, // 19 ~ 26
//			i_p1 + nx*j_m1 + nxy*k_m1,
//			i_m1 + nx*j_p1 + nxy*k_m1,
//			i_m1 + nx*j_m1 + nxy*k_p1,
//			i_m1 + nx*j_p1 + nxy*k_p1,
//			i_p1 + nx*j_m1 + nxy*k_p1,
//			i_p1 + nx*j_p1 + nxy*k_m1,
//			i_p1 + nx*j_p1 + nxy*k_p1  };
//
//
//#pragma unroll
//		// streaming and collision cycle (MOD2018), bounce back step
//		//	Z. Guo, C. Zheng, B. Shi, Discrete lattice effects on the forcing term
//		//	in the latticeBoltzmann method, Phys. Rev. E 65 (046308) (2002) 1.6.
//		// 	force->moment update->equilibrium->output->forcing term->collision
//		//					->streaming->new timestep->force ...
//
//		// streaming
//		for (int ii=0; ii<27; ii++) { fs[ii] = f[ii*nxyz + idg_lbm[ii]];}
//		//mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
//
//		// boundary condition
//		LBM_solid_boundary (
//				fs,
//				f,
//				l_obs,
//				idg_lbm,
//				direction_lbm,
//				27,
//				nxyz
//				);
//
//		// calculate density (rho)
//		mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
//		// f,F -> u,v,w,rho (for calculation of collistion operator) //
//		FLOAT us,vs,ws;
//		mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);
//
//		// force : nuoyancy:Boussinesq approximation
//		const int id = id_c0_c0_c0;
//		const int stride[3] = { 1, nx, nx*ny };
//		const FLOAT  rho_gas = coefficient::DENSITY_AIR;
//		const FLOAT  gravity = coefficient::GRAVITY;	
//		const FLOAT  beta       = 1.0 / BASE_TEMPERATURE;
//		const FLOAT  nondim_lbm = (1.0/c_ref/rho_gas) * dt;
//		FLOAT  force_lbm[NUM_DIRECTION_VEL];
//		FLOAT  force_x = 0.0;
//		FLOAT  force_y = 0.0;
//		FLOAT  force_z = 0.0;
//		// pressure gradient
//		if(user_flags::flg_dpdx==1)	{force_x = force_x - user_init::dpdx * nondim_lbm;}	//MOD2019
//		if(user_flags::flg_dpdy==1)	{force_y = force_y - user_init::dpdy * nondim_lbm;}	//MOD2019
//		if(user_flags::flg_coriolis==1) {
//			const FLOAT pi=3.14159265;
//			force_x = force_x + 2.0 * user_init::angular_velocity * vs * c_ref * sin(user_init::latitude/180.0 * pi) * nondim_lbm ;
//			force_y = force_y - 2.0 * user_init::angular_velocity * us * c_ref * sin(user_init::latitude/180.0 * pi) * nondim_lbm ;
//		}
//		// buoyancy
//		if(user_flags::flg_buoyancy==1) {force_z = force_z + rho_gas*fabs(gravity)*beta*(T[id] - BASE_TEMPERATURE) * nondim_lbm;}
//		mathLib_LBM::
//			Device_LBM_D3Q27_Force (
//					rhos,
//					force_x,
//					force_y,
//					force_z,
//					force_lbm
//					);
//
//		// f,F -> u,v,w,rho (for calculation of collistion operator) //
//		//	FLOAT us,vs,ws;
//		//	mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);
//
//		// Coriolis force
//		//	const FLOAT     omega_rotation = 7.29212*1.0e-5;		// angular velocity of the earth [rad\s]
//		//	const FLOAT	latitude = 35.681236;				// latuitude of Tokyo [rad]
//		//	const FLOAT	f_cor1 = 0.0;					// Coriolis parameter in 1th direction
//		//	const FLOAT	f_cor2 = 2.0 * omega_rotation * cos(latitude);	// Coriolis parameter in 2nd direction
//		//	const FLOAT	f_cor3 = 2.0 * omega_rotation * sin(latitude);	// Coriolis parameter in 3rd direction
//		//	force_x =   force_x + f_cor3 * vs;
//		//	force_y = - force_y + f_cor3 * us;
//
//		us = us + force_x*(FLOAT)0.5;
//		vs = vs + force_y*(FLOAT)0.5;
//		ws = ws + force_z*(FLOAT)0.5;
//
//		// collision term : fs, force -> fs_deq=fs-fs_eq
//		//	f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
//		//	fs_deq = fs - feq
//		mathLib_LBM::Device_LBM_D3Q27_d_equivalent_force(fs_deq, rhos, fs, force_x, force_y, force_z);
//
//		// OUTPUT
//		u[id] = us;
//		v[id] = vs;
//		w[id] = ws;
//		rho[id] = rhos;
//
//		// source term force_x,y,z -> force_lbm(i)
//
//		// sgs model : fs, feq -> csvis
//		const FLOAT     csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
//		vis_sgs[id_c0_c0_c0] = csvis;
//
//		// relaxation time
//		const FLOAT     tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
//		const FLOAT     omega     = (FLOAT)1.0/tau_total;               // 1.0/tau
//
//		// cumulant LBM (collision process)
//		// cumulant //
//		FLOAT fs_cum[27], fIs_cum[27];
//		FLOAT fIs[27];
//
//		fs_cum[0]  = fs[26];
//		fs_cum[1]  = fs[18];
//		fs_cum[2]  = fs[23];
//		fs_cum[3]  = fs[14];
//		fs_cum[4]  = fs[6];
//		fs_cum[5]  = fs[13];
//		fs_cum[6]  = fs[24];
//		fs_cum[7]  = fs[17];
//		fs_cum[8]  = fs[22];
//		fs_cum[9]  = fs[10];
//		fs_cum[10] = fs[4];
//		fs_cum[11] = fs[9];
//		fs_cum[12] = fs[2];
//		fs_cum[13] = fs[0];
//		fs_cum[14] = fs[1];
//		fs_cum[15] = fs[8];
//		fs_cum[16] = fs[3];
//		fs_cum[17] = fs[7];
//		fs_cum[18] = fs[25];
//		fs_cum[19] = fs[16];
//		fs_cum[20] = fs[21];
//		fs_cum[21] = fs[12];
//		fs_cum[22] = fs[5];
//		fs_cum[23] = fs[11];
//		fs_cum[24] = fs[20];
//		fs_cum[25] = fs[15];
//		fs_cum[26] = fs[19];
//
//		FuncCumulantLBM::fs_cumulant_lbm(fIs_cum, fs_cum, omega, rhos, us,vs,ws);
//
//		fIs[0]  = fIs_cum[13];
//		fIs[1]  = fIs_cum[14];
//		fIs[2]  = fIs_cum[12];
//		fIs[3]  = fIs_cum[16];
//		fIs[4]  = fIs_cum[10];
//		fIs[5]  = fIs_cum[22];
//		fIs[6]  = fIs_cum[4];
//		fIs[7]  = fIs_cum[17];
//		fIs[8]  = fIs_cum[15];
//		fIs[9]  = fIs_cum[11];
//		fIs[10] = fIs_cum[9];
//		fIs[11] = fIs_cum[23];
//		fIs[12] = fIs_cum[21];
//		fIs[13] = fIs_cum[5];
//		fIs[14] = fIs_cum[3];
//		fIs[15] = fIs_cum[25];
//		fIs[16] = fIs_cum[19];
//		fIs[17] = fIs_cum[7];
//		fIs[18] = fIs_cum[1];
//		fIs[19] = fIs_cum[26];
//		fIs[20] = fIs_cum[24];
//		fIs[21] = fIs_cum[20];
//		fIs[22] = fIs_cum[8];
//		fIs[23] = fIs_cum[2];
//		fIs[24] = fIs_cum[6];
//		fIs[25] = fIs_cum[18];
//		fIs[26] = fIs_cum[0];
//
//
//		// cumulant //
//#pragma unroll
//		// add collision and source terms
//		//	fnew = fs - 1/tau * (fs - feq)
//		for (int ii=0; ii<27; ii++) {
//			//	fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
//			fn[nxyz*ii + id_c0_c0_c0] = fIs[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
//		}
//
//		// thermal convection (MOD 2018) //
//
//		if(user_flags::flg_scalar==0) { continue; }	//MOD2019
//
//		if (l_obs[id] >= 0.0) { continue; }
//
//		//    before 2019/06/04
//		//	const FLOAT Tx = (u[id] >= 0.0) ? (T[id] - T[id-stride[0]])/dx : (T[id+stride[0]] - T[id])/dx;
//		//	const FLOAT Ty = (v[id] >= 0.0) ? (T[id] - T[id-stride[1]])/dx : (T[id+stride[1]] - T[id])/dx;
//		//	const FLOAT Tz = (w[id] >= 0.0) ? (T[id] - T[id-stride[2]])/dx : (T[id+stride[2]] - T[id])/dx;
//
//		// boundary condition //
//		const FLOAT HFxl = (l_obs[id-stride[0]] < 0.0) ? 0.0 : bcTe[id];
//		const FLOAT HFyl = (l_obs[id-stride[1]] < 0.0) ? 0.0 : bcTs[id];
//		const FLOAT HFzl = (l_obs[id-stride[2]] < 0.0) ? 0.0 : bcTr[id];
//
//		const FLOAT HFxr = (l_obs[id+stride[0]] < 0.0) ? 0.0 : bcTw[id];
//		const FLOAT HFyr = (l_obs[id+stride[1]] < 0.0) ? 0.0 : bcTn[id];
//		const FLOAT HFzr = (l_obs[id+stride[2]] < 0.0) ? 0.0 : 0.0;
//
//		// diffusion //
//		const FLOAT Txl = (l_obs[id-stride[0]] < 0.0) ? (T[id] - T[id-stride[0]])/dx : 0.0;
//		const FLOAT Tyl = (l_obs[id-stride[1]] < 0.0) ? (T[id] - T[id-stride[1]])/dx : 0.0;
//		const FLOAT Tzl = (l_obs[id-stride[2]] < 0.0) ? (T[id] - T[id-stride[2]])/dx : 0.0;
//
//		const FLOAT Txr = (l_obs[id+stride[0]] < 0.0) ? (T[id+stride[0]] - T[id])/dx : 0.0;
//		const FLOAT Tyr = (l_obs[id+stride[1]] < 0.0) ? (T[id+stride[1]] - T[id])/dx : 0.0;
//		const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;
//
//		//	const FLOAT uTx = (u[id] >= 0.0) ? u[id]*(T[id] - T[id-stride[0]])/dx : u[id]*(T[id+stride[0]] - T[id])/dx;
//		//	const FLOAT vTy = (v[id] >= 0.0) ? v[id]*(T[id] - T[id-stride[1]])/dx : v[id]*(T[id+stride[1]] - T[id])/dx;
//		//	const FLOAT wTz = (w[id] >= 0.0) ? w[id]*(T[id] - T[id-stride[2]])/dx : w[id]*(T[id+stride[2]] - T[id])/dx;
//		const FLOAT uTx = ( 0.5*(u[id]+u[id-stride[0]])*Txl + 0.5*(u[id]+u[id+stride[0]])*Txr ) * (FLOAT)0.5;
//		const FLOAT vTy = ( 0.5*(v[id]+v[id-stride[1]])*Tyl + 0.5*(v[id]+v[id+stride[1]])*Tyr ) * (FLOAT)0.5;
//		const FLOAT wTz = ( 0.5*(w[id]+w[id-stride[2]])*Tzl + 0.5*(w[id]+w[id+stride[2]])*Tzr ) * (FLOAT)0.5;
//
//		const FLOAT Txx = (Txr - Txl) / (dx);
//		const FLOAT Tyy = (Tyr - Tyl) / (dx);
//		const FLOAT Tzz = (Tzr - Tzl) / (dx);
//
//		const FLOAT coefT = COEF_THERMAL + csvis / PR_SGS * c_ref * (dx);        //* 5.0;       // MOD 2018
//
//		const FLOAT adv_term = - (uTx + vTy + wTz)*c_ref;
//		//const FLOAT adv_term  = - ( u[id]*Tx + v[id]*Ty + w[id]*Tz )*c_ref;
//		const FLOAT diff_term = coefT * ( Txx + Tyy + Tzz ) + (HFxr + HFxl)/dx + (HFyr + HFyl)/dx + (HFzr + HFzl)/dx;
//
//		Tn[id] = T[id] + (adv_term + diff_term)*dt;
//
//
//		/*
//		// heat flux
//		const FLOAT HFxl = (l_obs[id-stride[0]] < 0.0) ? 0.0 : bcTe[id];
//		const FLOAT HFyl = (l_obs[id-stride[1]] < 0.0) ? 0.0 : bcTs[id];
//		const FLOAT HFzl = (l_obs[id-stride[2]] < 0.0) ? 0.0 : bcTr[id];
//		const FLOAT HFxr = (l_obs[id+stride[0]] < 0.0) ? 0.0 : bcTw[id];
//		const FLOAT HFyr = (l_obs[id+stride[1]] < 0.0) ? 0.0 : bcTn[id];
//		const FLOAT HFzr = (l_obs[id+stride[2]] < 0.0) ? 0.0 : 0.0;
//		// advection
//		const FLOAT Fxl = (l_obs[id-stride[0]] < 0.0) ? 0.5 * ( u[id]+u[id-stride[0]] ) * 0.5 * ( T[id]+T[id-stride[0]] ) : 0.0;
//		const FLOAT Fyl = (l_obs[id-stride[1]] < 0.0) ? 0.5 * ( v[id]+v[id-stride[1]] ) * 0.5 * ( T[id]+T[id-stride[1]] ) : 0.0;
//		const FLOAT Fzl = (l_obs[id-stride[2]] < 0.0) ? 0.5 * ( w[id]+w[id-stride[2]] ) * 0.5 * ( T[id]+T[id-stride[2]] ) : 0.0;
//		const FLOAT Fxr = (l_obs[id+stride[0]] < 0.0) ? 0.5 * ( u[id+stride[0]]+u[id] ) * 0.5 * ( T[id+stride[0]]+T[id] ) : 0.0;
//		const FLOAT Fyr = (l_obs[id+stride[1]] < 0.0) ? 0.5 * ( v[id+stride[1]]+v[id] ) * 0.5 * ( T[id+stride[1]]+T[id] ) : 0.0;
//		const FLOAT Fzr = (l_obs[id+stride[2]] < 0.0) ? 0.5 * ( w[id+stride[2]]+w[id] ) * 0.5 * ( T[id+stride[2]]+T[id] ) : 0.0;
//		// diffusion
//		const FLOAT Txl = (l_obs[id-stride[0]] < 0.0) ? (T[id] - T[id-stride[0]])/dx : 0.0;
//		const FLOAT Tyl = (l_obs[id-stride[1]] < 0.0) ? (T[id] - T[id-stride[1]])/dx : 0.0;
//		const FLOAT Tzl = (l_obs[id-stride[2]] < 0.0) ? (T[id] - T[id-stride[2]])/dx : 0.0;
//		const FLOAT Txr = (l_obs[id+stride[0]] < 0.0) ? (T[id+stride[0]] - T[id])/dx : 0.0;
//		const FLOAT Tyr = (l_obs[id+stride[1]] < 0.0) ? (T[id+stride[1]] - T[id])/dx : 0.0;
//		const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;
//		const FLOAT Txx = (Txr - Txl) / (dx);
//		const FLOAT Tyy = (Tyr - Tyl) / (dx);
//		const FLOAT Tzz = (Tzr - Tzl) / (dx);
//		const FLOAT coefT = COEF_THERMAL + csvis / PR_SGS * c_ref * (dx);        //* 5.0;       // MOD 2018
//
//		const FLOAT diff_term = coefT * ( Txx + Tyy + Tzz ) + (HFxr + HFxl)/dx + (HFyr + HFyl)/dx + (HFzr + HFzl)/dx + (-Fxr + Fxl)/dx + (-Fyr + Fyl)/dx + (-Fzr + Fzl)/dx;
//
//		Tn[id] = T[id] + diff_term*dt;
//		 */
//	}
//}


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
		)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// index local //
	const int	nxy = nx*ny;


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// calculation //
		FLOAT	cforce_x = 0.0;
		FLOAT	cforce_y = 0.0;
		FLOAT	cforce_z = (FLOAT)2.0/(nx)/(c_ref*c_ref);

		if (l_obs[id_c0_c0_c0] >= 0.0) {
			cforce_x = 0.0;
			cforce_y = 0.0;
			cforce_z = 0.0;
		}


		// update //
		force_x[id_c0_c0_c0] = cforce_x;
		force_y[id_c0_c0_c0] = cforce_y;
		force_z[id_c0_c0_c0] = cforce_z;
	}
}


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
		)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	// D3Q19:
	// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)


	// D3Q27:
	// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
	// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
	//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)


	// index local //
	const int	nxy = nx*ny;

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos; // variables after streaming step
	FLOAT	fs[NUM_DIRECTION_VEL];

	// force
	FLOAT	force[NUM_DIRECTION_VEL];


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// calculation //
#pragma unroll
		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			fs[ii] = fn[ii*nxyz + id_c0_c0_c0];
		}


		// force //
		const FLOAT	cforce_x = force_x[id_c0_c0_c0];
		const FLOAT	cforce_y = force_y[id_c0_c0_c0];
		const FLOAT	cforce_z = force_z[id_c0_c0_c0];

		mathLib_LBM::device_lbm_rho   (rhos, fs, d3qx_velocity);
		mathLib_LBM::device_lbm_force (rhos, cforce_x, cforce_y, cforce_z, force, d3qx_velocity);

#pragma unroll
		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			fn[nxyz*ii + id_c0_c0_c0] += force[ii];
		}

	}
}


// lbm_function_to_velocity //
	__global__ void
cuda_lbm_function_to_velocity (
		const FLOAT	*f,
		FLOAT	*rho,
		FLOAT	*u,
		FLOAT	*v,
		FLOAT	*w,
		int nx, int ny, int nz,
		int halo,
		LBM_VELOCITY_MODEL	d3qx_velocity
		)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// variables //
	const int	nxy = nx*ny;
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws;
	FLOAT	fs[NUM_DIRECTION_VEL];


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		// calculation //
#pragma unroll
		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			fs[ii] = f[nxyz*ii + id_c0_c0_c0];
		}


		// update BasisVariables //
		// fs -> rho, u,v,w
		mathLib_LBM::device_lbm_rho_velocity (rhos, us, vs, ws, fs, d3qx_velocity);
		//#ifdef D3Q19_MODEL_
		//		mathLib_LBM::Device_LBM_D3Q19_rho_velocity (rhos, us, vs, ws, fs);
		//#endif
		//#ifdef D3Q27_MODEL_
		//		mathLib_LBM::Device_LBM_D3Q27_rho_velocity (rhos, us, vs, ws, fs);
		//#endif


		// update //
		rho[id_c0_c0_c0] = rhos;
		u  [id_c0_c0_c0] = us;
		v  [id_c0_c0_c0] = vs;
		w  [id_c0_c0_c0] = ws;
	}
}


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
		)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	// variables //
	const int	nn = nx*ny*nz;
	FLOAT	feq[NUM_DIRECTION_VEL];

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// index
		const int	id_c0_c0_c0 = id_x + nx*id_y + nx*ny*id_z;

		// calculation *****
		// f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )
		const FLOAT	rhos = rho[id_c0_c0_c0];
		const FLOAT	us   = u  [id_c0_c0_c0];
		const FLOAT	vs   = v  [id_c0_c0_c0];
		const FLOAT	ws   = w  [id_c0_c0_c0];

		// rhos, us, vs, ws -> feq
		mathLib_LBM::device_lbm_feq (rhos, us, vs, ws, feq, d3qx_velocity);
		//#ifdef D3Q19_MODEL_
		//		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
		//#endif
		//#ifdef D3Q27_MODEL_
		//		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
		//#endif


#pragma unroll
		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nn*ii + id_c0_c0_c0] = feq[ii];
		}
	}
}


// lbm_function_to_velocity //
	__global__ void
cuda_velocity_obstacle_filter (
		FLOAT	*u,
		FLOAT	*v,
		FLOAT	*w,
		const FLOAT	*lv,
		int nx,     int ny,     int nz,
		int halo
		)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
		  id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
		  id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// index
		const int	id_c0_c0_c0 = id_x + nx*id_y + nx*ny*id_z;

		if (lv[id_c0_c0_c0] < 0.0)	{ continue; }


		// calculation //
		FLOAT	us   = u[id_c0_c0_c0];
		FLOAT	vs   = v[id_c0_c0_c0];
		FLOAT	ws   = w[id_c0_c0_c0];

		if (lv[id_c0_c0_c0] >= 0.0) {
			us = 0.0;
			vs = 0.0;
			ws = 0.0;
		}


		// updata //
		u[id_c0_c0_c0] = us;
		v[id_c0_c0_c0] = vs;
		w[id_c0_c0_c0] = ws;
	}
}


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
		)
{
#ifdef BOUNDARY_BOUNCE_BACK_
	mathFuncLBM::solid_boundary_bounce_back (
			fs,
			f,
			l_obs,
			idg_lbm,
			direction_lbm,
			num_direction_vel,
			nn
			);
#endif
#ifdef BOUNDARY_BOUZIDI_
	mathFuncLBM::solid_boundary_BOUZIDI (
			fs,
			f,
			l_obs,
			idg_lbm,
			direction_lbm,
			num_direction_vel,
			nn
			);
#endif
}

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
		)
{
	for(int ii=0; ii<=6; ii++){
		if(l_obs[idg_lbm[0]] < 0.0){
			if       (l_obs[idg_lbm[ii]] < 0.0){
				fs[ii] = f[ii*nn + idg_lbm[ii]];
			}else if (l_obs[idg_lbm[ii]] > 0.0){
				fs[ii] = f[direction_lbm_slip[0][ii]*nn + idg_lbm[0]];
			}
			if(isnan(fs[ii])){
				printf("%d\t%d\n",ii,idg_lbm[0]);
				assert(!isnan(fs[ii]));
			}
		}
	}
	for(int ii=7; ii<=10; ii++){
		if(l_obs[idg_lbm[0]] < 0.0){
			FLOAT suml = 0.0;
			FLOAT sumf = 0.0;
			uint8_t status = 0;
			if(         l_obs[ idg_lbm[0]                                                      ] < 0.0){
				sumf += f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
				suml += 1.0;
				status += 1;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii]                                       ] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii]                                       + direction_lbm_slip[2][ii]*nn];
				suml += 1.0;
				status += 2;
			}
			if(         l_obs[ idg_lbm[0]                - nx*eD3Q27_y[ii]                     ] < 0.0){
				sumf += f    [ idg_lbm[0]                - nx*eD3Q27_y[ii]                     + direction_lbm_slip[1][ii]*nn];
				suml += 1.0;
				status += 4;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii]                     ] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii]                     + direction_lbm_slip[7][ii]*nn];
				suml += 1.0;
				status += 8;
			}
			if       ( status == 1+2+4+8){
				fs[ii]= f    [idg_lbm[ii]                                                      + direction_lbm_slip[7][ii]*nn];
			}else if ( status == 1 || status == 1+8){
				fs[ii]= f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
			}else if ( status == 1+2 ){
				fs[ii]= f    [ idg_lbm[0] - eD3Q27_x[ii]                                       + direction_lbm_slip[2][ii]*nn];
			}else if ( status == 1+4 ){
				fs[ii]= f    [ idg_lbm[0]                 - nx*eD3Q27_y[ii]                    + direction_lbm_slip[1][ii]*nn];
			}else if ( status == 1+2+4 || status == 1+2+8 || status == 1+4+8 ){
				fs[ii]= sumf/suml;
				assert(suml>0.0);
			}
			if(isnan(fs[ii])){
				printf("%d\t%d\n",ii,idg_lbm[0]);
				assert(!isnan(fs[ii]));
			}
		}
	}
	for(int ii=11; ii<=14; ii++){
		if(l_obs[idg_lbm[0]] < 0.0){
			FLOAT suml = 0.0;
			FLOAT sumf = 0.0;
			uint8_t status = 0;
			if(         l_obs[ idg_lbm[0]                                                      ] < 0.0){
				sumf += f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
				suml += 1.0;
				status += 1;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii]                                       ] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii]                                       + direction_lbm_slip[4][ii]*nn];
				suml += 1.0;
				status += 2;
			}
			if(         l_obs[ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[1][ii]*nn];
				suml += 1.0;
				status += 4;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii]                   - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii]                   - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[7][ii]*nn];
				suml += 1.0;
				status += 8;
			}
			if       ( status == 1+2+4+8){
				fs[ii]= f    [idg_lbm[ii]                                                      + direction_lbm_slip[7][ii]*nn];
			}else if ( status == 1 || status == 1+8){
				fs[ii]= f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
			}else if ( status == 1+2 ){
				fs[ii]= f    [ idg_lbm[0] - eD3Q27_x[ii]                                       + direction_lbm_slip[4][ii]*nn];
			}else if ( status == 1+4 ){
				fs[ii]= f    [ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[1][ii]*nn];
			}else if ( status == 1+2+4 || status == 1+2+8 || status == 1+4+8 ){
				fs[ii]= sumf/suml;
				assert(suml>0.0);
			}
			if(isnan(fs[ii])){
				printf("%d\t%d\n",ii,idg_lbm[0]);
				assert(!isnan(fs[ii]));
			}
		}
	}
	for(int ii=15; ii<=18; ii++){
		if(l_obs[idg_lbm[0]] < 0.0){
			FLOAT suml = 0.0;
			FLOAT sumf = 0.0;
			uint8_t status = 0;
			if(         l_obs[ idg_lbm[0]                                                      ] < 0.0){
				sumf += f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
				suml += 1.0;
				status += 1;
			}
			if(         l_obs[ idg_lbm[0]                - nx*eD3Q27_y[ii]                     ] < 0.0){
				sumf += f    [ idg_lbm[0]                - nx*eD3Q27_y[ii]                     + direction_lbm_slip[4][ii]*nn];
				suml += 1.0;
				status += 2;
			}
			if(         l_obs[ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[2][ii]*nn];
				suml += 1.0;
				status += 4;
			}
			if(         l_obs[ idg_lbm[0]                - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0]                - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[7][ii]*nn];
				suml += 1.0;
				status += 8;
			}
			if       ( status == 1+2+4+8){
				fs[ii]= f    [idg_lbm[ii]                                                      + direction_lbm_slip[7][ii]*nn];
			}else if ( status == 1 || status == 1+8){
				fs[ii]= f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
			}else if ( status == 1+2 ){
				fs[ii]= f    [ idg_lbm[0]                - nx*eD3Q27_y[ii]                     + direction_lbm_slip[4][ii]*nn];
			}else if ( status == 1+4 ){
				fs[ii]= f    [ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[2][ii]*nn];
			}else if ( status == 1+2+4 || status == 1+2+8 || status == 1+4+8 ){
				fs[ii]= sumf/suml;
				assert(suml>0.0);
			}
			if(isnan(fs[ii])){
				printf("%d\t%d\n",ii,idg_lbm[0]);
				assert(!isnan(fs[ii]));
			}
		}
	}
	for(int ii=19; ii<=26; ii++){
		if(l_obs[idg_lbm[0]] < 0.0){
			FLOAT suml = 0.0;
			FLOAT sumf = 0.0;
			uint8_t status = 0;
			if(         l_obs[ idg_lbm[0]                                                      ] < 0.0){
				sumf += f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
				suml += 1.0;
				status+=1;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii]                                       ] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii]                                       + direction_lbm_slip[6][ii]*nn];
				suml += 1.0;
				status+=2;
			}
			if(         l_obs[ idg_lbm[0]                - nx*eD3Q27_y[ii]                     ] < 0.0){
				sumf += f    [ idg_lbm[0]                - nx*eD3Q27_y[ii]                     + direction_lbm_slip[5][ii]*nn];
				suml += 1.0;
				status+=4;
			}
			if(         l_obs[ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[3][ii]*nn];
				suml += 1.0;
				status+=8;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii]                     ] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii]                     + direction_lbm_slip[4][ii]*nn];
				suml += 1.0;
				status+=16;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii]                   - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii]                   - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[2][ii]*nn];
				suml += 1.0;
				status+=32;
			}
			if(         l_obs[ idg_lbm[0]                - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0]                - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[1][ii]*nn];
				suml += 1.0;
				status+=64;
			}
			if(         l_obs[ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]] < 0.0){
				sumf += f    [ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[7][ii]*nn];
				suml += 1.0;
				status+=128;
			}
			if      ( status == 1+2+4+8+16+32+64+128){
				fs[ii]= f    [idg_lbm[ii]                                                      + direction_lbm_slip[7][ii]*nn];
			}else if((~status&(uint8_t)(2+4+8)) == (uint8_t)2+4+8 ){
				fs[ii]= f	 [ idg_lbm[0]                                                      + direction_lbm_slip[0][ii]*nn];
			}else if( status == (1+2)){
				fs[ii]= f    [ idg_lbm[0] - eD3Q27_x[ii]                                       + direction_lbm_slip[6][ii]*nn];
			}else if( status == (1+4)){
				fs[ii]= f    [ idg_lbm[0]                - nx*eD3Q27_y[ii]                     + direction_lbm_slip[5][ii]*nn];
			}else if( status == (1+8)){
				fs[ii]= f    [ idg_lbm[0]                                  - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[3][ii]*nn];
			}else if( status == (1+2+4+16)){
				fs[ii]= f    [ idg_lbm[0] - eD3Q27_x[ii] - nx*eD3Q27_y[ii]                     + direction_lbm_slip[4][ii]*nn];
			}else if( status == (1+2+8+32)){
				fs[ii]= f    [ idg_lbm[0] - eD3Q27_x[ii]                   - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[2][ii]*nn];
			}else if( status == (1+4+8+64)){
				fs[ii]= f    [ idg_lbm[0]                - nx*eD3Q27_y[ii] - nx*ny*eD3Q27_z[ii]+ direction_lbm_slip[1][ii]*nn];
			}else{
				fs[ii]= sumf / suml;
				assert(suml>0.0);
			}
			if(isnan(fs[ii])){
				printf("%d\t%d\n",ii,idg_lbm[0]);
				assert(!isnan(fs[ii]));
			}
		}
	}
}


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
		)
{
	const int		num_direction_vel = 19;
	FLOAT	force_obs[num_direction_vel];


	// momentum_exchange_on_boundary_D3Q19 //
	mathFuncLBM::momentum_exchange_on_boundary_D3Q19 (
			force_obs,
			l_obs,
			u_obs,
			v_obs,
			w_obs,
			idg_lbm,
			direction_lbm,
			nn
			);


	// moving boundary //
	mathFuncLBM::moving_boundary_BOUZIDI (
			fs,
			f,
			l_obs,
			force_obs,
			idg_lbm,
			direction_lbm,
			num_direction_vel,
			nn
			);
}
// wall function
__device__ FLOAT wall_func(FLOAT ut, FLOAT U, FLOAT y){
	FLOAT k = 0.4;
	FLOAT B = 5.5;
	return U/ut + exp(-k*B)*(exp(k*U/ut) -1.0 -(k*U/ut) -(k*k*U*U/(2.0*ut*ut)) -(k*k*k*U*U*U/(6.0*ut*ut*ut)) )  - (ut*y/coefficient::KVIS_AIR);
}

__device__ FLOAT d_wall_func_dut(FLOAT ut, FLOAT U, FLOAT y){
	FLOAT k = 0.4;
	FLOAT B = 5.5;
	return -U/(ut*ut) + exp(-k*B)*( -k*U*exp(k*U/ut)/(ut*ut) +k*U/(ut*ut) + k*k*U*U/(ut*ut*ut) + (k*k*k*U*U*U/(2.0*ut*ut*ut*ut))  ) - y/coefficient::KVIS_AIR;
}

__device__ FLOAT d_wall_func_dU(FLOAT ut, FLOAT U, FLOAT y){
	FLOAT k = 0.4;
	FLOAT B = 5.5;
	return 1.0/ut -exp(-k*B)*(k/ut + k*k*U/(ut*ut) +k*k*k*U*U/(2.0*ut*ut*ut) -k*expf(k*U/ut)/ut);
}


// (YOKOUCHI 2020)
// Mean velocity GPU ... host function is in "Calculation" class
__global__ void mean_velocity_GPU (
	const FLOAT *u,	const FLOAT *v,	const FLOAT *w,
	      FLOAT *um,      FLOAT *vm,      FLOAT *wm,
	      int   nx, int ny, int nz,
	      int   halo,
	      int   t
	)
{
	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
			id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
			id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	const int	nxy  = nx*ny; 

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;
		// index
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;

		if (t == 0) {
		um[id_c0_c0_c0] = 0.0;
		vm[id_c0_c0_c0] = 0.0;
		wm[id_c0_c0_c0] = 0.0;
		}

		FLOAT 	um_before = um[id_c0_c0_c0];
		FLOAT	vm_before = vm[id_c0_c0_c0];
		FLOAT	wm_before = wm[id_c0_c0_c0];
	
		um[id_c0_c0_c0]	= ((FLOAT)(t) * um_before + u[id_c0_c0_c0]) / (FLOAT)(t+1);
		vm[id_c0_c0_c0]	= ((FLOAT)(t) * vm_before + v[id_c0_c0_c0]) / (FLOAT)(t+1);
		wm[id_c0_c0_c0]	= ((FLOAT)(t) * wm_before + w[id_c0_c0_c0]) / (FLOAT)(t+1);
	}
	
}	

__global__ void tke_LBM_GPU (
	const FLOAT *u, const FLOAT *v, const FLOAT *w,
	      FLOAT *tke_sgs,
	      FLOAT *tke_sgs_old,
	      int   nx, int ny, int nz,
	      int   halo,
	      int   t
	)
{
	// const value //
	const FLOAT	C_kes	= 1.0;

	// cuda index
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
			id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
			id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);

	const int	nxy  = nx*ny; 
	
	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;
		// index
		const int	id = id_x + nx*id_y + nxy*id_z;
	
		// index ... East x+1,  West x-1, North y+1, South y-1, Top z+1, Bottom z-1
		const int id_xp = id_x + 1;	// East
		const int id_xm = id_x - 1;	// West
		const int id_yp = id_y + 1;	// North
		const int id_ym = id_y - 1;	// South
		const int id_zp = id_z + 1;	// Top
		const int id_zm = id_z - 1;	// Bottom

		// index array for average
		const int id_ave[6] = {
			id_xp + nx*id_y  + nxy*id_z, 	// East
			id_xm + nx*id_y  + nxy*id_z,	// West
			id_x  + nx*id_yp + nxy*id_z,	// North 
			id_x  + nx*id_ym + nxy*id_z,	// South
			id_x  + nx*id_y  + nxy*id_zp,	// Top
			id_x  + nx*id_y  + nxy*id_zm };	// Bottom
	
		if (t==0) {
			tke_sgs[id]	= 0.0;
			tke_sgs_old[id]	= 0.0;
		}
		
		const FLOAT	u_ins	= u[id];
		const FLOAT	v_ins	= v[id];
		const FLOAT	w_ins	= w[id];

		      FLOAT	u_ave	= 0.0;
		      FLOAT	v_ave	= 0.0;
		      FLOAT	w_ave	= 0.0;

		for (int i=0; i<6; i++) {
			int id_a = id_ave[i];
			
			u_ave += u[id_a] / 6.0;
			v_ave += v[id_a] / 6.0;
			w_ave += w[id_a] / 6.0;
		}
		
		tke_sgs_old[id]	= tke_sgs[id]; // for temporal derivative

		// difference //
		FLOAT	u_diff	= u_ave - u_ins;
		FLOAT	v_diff	= v_ave - v_ins;
		FLOAT	w_diff	= w_ave - w_ins;
	     

		tke_sgs[id]	= C_kes * (0.5*0.5) * (u_diff*u_diff + v_diff*v_diff + w_diff*w_diff);
	
	}	
}


