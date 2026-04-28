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
	LBM_solid_boundary (
                        fs,
                        f,
                        l_obs,
                        idg_lbm,
                        direction_lbm,
                        27,
                        nxyz
                        );

	// calculate density (rho)
	mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
	// f,F -> u,v,w,rho (for calculation of collistion operator) //
	FLOAT us,vs,ws;
	mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);

	// force : nuoyancy:Boussinesq approximation
	const int id = id_c0_c0_c0;
	const int stride[3] = { 1, nx, nx*ny };
	const FLOAT  rho_gas = coefficient::DENSITY_AIR;
	const FLOAT  gravity = coefficient::GRAVITY;	
	const FLOAT  beta       = (FLOAT)1.0 / BASE_TEMPERATURE;
	const FLOAT  nondim_lbm = ((FLOAT)1.0/c_ref/rho_gas) * dt;
	FLOAT  force_lbm[NUM_DIRECTION_VEL];
	FLOAT  force_x = (FLOAT)0.0;
	FLOAT  force_y = (FLOAT)0.0;
	FLOAT  force_z = (FLOAT)0.0;
// pressure gradient
	if(user_flags::flg_dpdx==1)	{force_x = force_x - user_init::dpdx * nondim_lbm;}	//MOD2019
	if(user_flags::flg_dpdy==1)	{force_y = force_y - user_init::dpdy * nondim_lbm;}	//MOD2019
	if(user_flags::flg_coriolis==1) {
		const FLOAT pi=(FLOAT)3.14159265;
		force_x = force_x + (FLOAT)2.0 * user_init::angular_velocity * vs * c_ref * sin(user_init::latitude/(FLOAT)180.0 * pi) * nondim_lbm ;
		force_y = force_y - (FLOAT)2.0 * user_init::angular_velocity * us * c_ref * sin(user_init::latitude/(FLOAT)180.0 * pi) * nondim_lbm ;
	}
// buoyancy
	if(user_flags::flg_buoyancy==1) {force_z = force_z + rho_gas*fabs(gravity)*beta*(T[id] - BASE_TEMPERATURE) * nondim_lbm;}
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

	// OUTPUT
	u[id] = us;
	v[id] = vs;
	w[id] = ws;
	rho[id] = rhos;

	// source term force_x,y,z -> force_lbm(i)

	// sgs model : fs, feq -> csvis
	const FLOAT     csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
	vis_sgs[id_c0_c0_c0] = csvis;

	// relaxation time
	const FLOAT     tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
	const FLOAT     omega     = (FLOAT)1.0/tau_total;               // 1.0/tau

#pragma unroll
	// add collision and source terms
	//	fnew = fs - 1/tau * (fs - feq)
	for (int ii=0; ii<27; ii++) {
		fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
	}

	// thermal convection (MOD 2018) //

	if(user_flags::flg_scalar==0) { continue; }	//MOD2019

	if (l_obs[id] >= (FLOAT)0.0) { continue; }

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
	const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;

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

__global__ void
cuda_stream_collision_T_D3Q27_wall (
	const FLOAT	*f,
	      FLOAT	*fn,
	const FLOAT	*T,
	      FLOAT	*Tn,
	const FLOAT	*l_obs,
	const int *l_obs_x,
	const int *l_obs_y,
	const int *l_obs_z,
	  FLOAT	*u,
	  FLOAT	*v,
	  FLOAT	*w,
	  FLOAT *rho,
	const FLOAT	*vis,
	      FLOAT	*vis_sgs,
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
	int halo,

	FLOAT *u_bb, 
	FLOAT *v_bb, 
	FLOAT *w_bb
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
	const FLOAT l1 = 1;
	const FLOAT l2 = (FLOAT)sqrtf(2);
	const FLOAT l3 = (FLOAT)sqrtf(3);
	const FLOAT eD3Q27_l[27]={ 0,l1,l1,l1,l1,l1,l1,l2,l2,l2,l2,l2,l2,l2,l2,l2,l2,l2,l2,l3,l3,l3,l3,l3,l3,l3,l3 };
	const FLOAT w1 = (FLOAT)8.0/(FLOAT)27.0;
	const FLOAT w2 = (FLOAT)2.0/(FLOAT)27.0;
	const FLOAT w3 = (FLOAT)1.0/(FLOAT)54.0;
	const FLOAT w4 = (FLOAT)1.0/(FLOAT)216.0;
	const FLOAT eD3Q27_w[27]={w1,w2,w2,w2,w2,w2,w2,w3,w3,w3,w3,w3,w3,w3,w3,w3,w3,w3,w3,w4,w4,w4,w4,w4,w4,w4,w4 };

	// global index
	// lbm stream from
	// local id (e[ii] : depature point )
	const int direction_lbm[27]     = {0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,14,13,12,11,18,17,16,15,26,23,24,25,20,21,22,19};

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
	LBM_solid_boundary (
                        fs,
                        f,
                        l_obs,
                        idg_lbm,
                        direction_lbm,
                        27,
                        nxyz
                        );
	//LBM_slid_slip_boundary(
	//					    fs,
	//						f,
	//						l_obs,
	//						l_obs_x[id_c0_c0_c0],
	//						l_obs_y[id_c0_c0_c0],
	//						l_obs_z[id_c0_c0_c0],
	//						idg_lbm,
	//						direction_lbm_slip,
	//						27,
	//						nx,ny,nz,nxyz
	//);

	// calculate density (rho)
	mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
	// f,F -> u,v,w,rho (for calculation of collistion operator) //
	FLOAT us,vs,ws;
	mathLib_LBM::Device_LBM_D3Q27_rho_velocity(rhos, us, vs, ws, fs);

	// upload streamed u,v,w
	u_bb[id_c0_c0_c0] = us;
	v_bb[id_c0_c0_c0] = vs;
	w_bb[id_c0_c0_c0] = ws;

	FLOAT f_bp_deq[27];
	FLOAT rho_bc;
	//mathLib_LBM::Device_LBM_D3Q27_rho(rho_bc, f);
	//mathLib_LBM::Device_LBM_D3Q27_d_equivalent(f_bp_deq, rho_bc, f);
	//for(int ii=0; ii<27; ii++){
	//	FLOAT rtmp = rho[idg_lbm[ii]];
	//	FLOAT utmp = u[idg_lbm[ii]];
	//	FLOAT vtmp = v[idg_lbm[ii]];
	//	FLOAT wtmp = w[idg_lbm[ii]];
	//	FLOAT ftmp;
	//	mathLib_LBM::Device_LBM_D3Q27_feq(rtmp, utmp, vtmp, wtmp, f_bp_eq);
	//	f_bp_deq[ii] = f[ii*nxyz + idg_lbm[ii]] - f_bp_eq[ii];
	//}

	// wall treatment (ISHIBASHI 2019)
		int		p_ip[3] 	= {0, 0, 0};		// relative position of imaginary point from boundary point
		FLOAT 	p_ip_v[3]	= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0}; 	// unit vector for p_ip
		FLOAT 	p_ip_l 		= (FLOAT)0.0; 				// length of p_ip
		int 	id_ip 		= id_c0_c0_c0; 		// absolute positon of imaginary point (initialized with boundary point
		FLOAT	u_ip[3] 	= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0}; 	// velocity vector for IP in cartesian grid
		FLOAT	u_ip_n[3]	= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0};	// velocity vector for IP normal to the wall
		FLOAT	u_ip_p[3]	= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0};	// velocity vector for IP pararell to the wall
		FLOAT	u_bp_n[3]	= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0};	// velocity vector for BP normal to the wall
		FLOAT	u_bp_p[3]	= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0};	// velocity vector for BP pararell to the wall
		FLOAT	u_bp[3]		= {(FLOAT)0.0, (FLOAT)0.0, (FLOAT)0.0};		// velocity vector for BP aligned with cartesian grid
		FLOAT   f_bp[27];
		FLOAT   f_bp_eq[27];
		//FLOAT   f_bp_deq[27];
		//FLOAT   rho_bc = rhos;
		rho_bc = rhos;
		// calculate Imaginary Point (IP) vector
		int flag_ip = 0;

		if(flag_ip==0){ // when the surface touch to solid
			for (int ii=1; ii<=6; ii++){
				if(l_obs[idg_lbm[ii]] > (FLOAT)0.0){
					p_ip[0] += eD3Q27_x[ii];
					p_ip[1] += eD3Q27_y[ii];
					p_ip[2] += eD3Q27_z[ii];
					flag_ip = 1;
					p_ip_l += (FLOAT)1.0;
				}
			}
		}
		
		if(flag_ip==0){ // when the edge touch to solid
			for (int ii=7; ii<=18; ii++){
				if(l_obs[idg_lbm[ii]] > (FLOAT)0.0){
					p_ip[0] += eD3Q27_x[ii];
					p_ip[1] += eD3Q27_y[ii];
					p_ip[2] += eD3Q27_z[ii];
					flag_ip += 1;
					p_ip_l = (FLOAT)2.0;
				}
			}
			assert(flag_ip<=1);
		}

		if(flag_ip==0){ // when the point touch to solid
			for (int ii=19; ii<=26; ii++){
				if(l_obs[idg_lbm[ii]] > (FLOAT)0.0){
					p_ip[0] += eD3Q27_x[ii];
					p_ip[1] += eD3Q27_y[ii];
					p_ip[2] += eD3Q27_z[ii];
					flag_ip += 1;
					p_ip_l = (FLOAT)3.0;
				}
			}
			assert(flag_ip<=1);
		}

		if(flag_ip && l_obs[id_c0_c0_c0]<(FLOAT)0.0){ // wall treatment
			// define the vector
			id_ip = id_ip + p_ip[0] + nx*p_ip[1] + nxy*p_ip[2];
			assert(l_obs[id_ip]<0);
			for(int ii=0; ii<3; ii++){
				p_ip_v[ii] = (FLOAT)(p_ip[ii]);
			}
			p_ip_l = (FLOAT)sqrtf(p_ip_l);
			assert(p_ip_l>(FLOAT)0.9 && p_ip_l<(FLOAT)1.1);
			assert(p_ip_v[0] < (FLOAT)0.1);
			assert(p_ip_v[1] < (FLOAT)0.1);
			for (int ii=0; ii<3; ii++){
				p_ip_v[ii] = p_ip_v[ii] / p_ip_l;
			}
			// get u,v,w from IP
			u_ip[0] = u_bb[id_ip] * c_ref;
			u_ip[1] = v_bb[id_ip] * c_ref;
			u_ip[2] = w_bb[id_ip] * c_ref;
			// calculate velocity vector for normal and pararell to the wall
			u_ip_n[0] = (u_ip[0]*p_ip_v[0] * u_ip[1]*p_ip_v[1] * u_ip[2]*p_ip_v[2]) * p_ip_v[0];
			u_ip_n[1] = (u_ip[0]*p_ip_v[0] * u_ip[1]*p_ip_v[1] * u_ip[2]*p_ip_v[2]) * p_ip_v[1];
			u_ip_n[2] = (u_ip[0]*p_ip_v[0] * u_ip[1]*p_ip_v[1] * u_ip[2]*p_ip_v[2]) * p_ip_v[2];
			u_ip_p[0] = u_ip[0] - u_ip_n[0];
			u_ip_p[1] = u_ip[1] - u_ip_n[1];
			u_ip_p[2] = u_ip[2] - u_ip_n[2];
			u_bp_n[0] = (us*p_ip_v[0] * vs*p_ip_v[1] * ws*p_ip_v[2]) * p_ip_v[0];
			u_bp_n[1] = (us*p_ip_v[0] * vs*p_ip_v[1] * ws*p_ip_v[2]) * p_ip_v[1];
			u_bp_n[2] = (us*p_ip_v[0] * vs*p_ip_v[1] * ws*p_ip_v[2]) * p_ip_v[2];

			// calculate u,v,w at boundary node using wall model
			FLOAT z1 = p_ip_l * (FLOAT)0.5 * dx;
			FLOAT z2 = p_ip_l * (FLOAT)1.5 * dx;
			assert(z2 > z1);
			FLOAT U_ip_n = (FLOAT)sqrtf(u_ip_n[0]*u_ip_n[0] + u_ip_n[1]*u_ip_n[1] + u_ip_n[2]*u_ip_n[2]);
			FLOAT U_ip_p = (FLOAT)sqrtf(u_ip_p[0]*u_ip_p[0] + u_ip_p[1]*u_ip_p[1] + u_ip_p[2]*u_ip_p[2]);
			assert(U_ip_p >= 0.0);
			//assert(U_ip_p < 0.1);
			//if(U_ip_p > coefficient::NON_ZERO_EP * 1.0e4){
			if(U_ip_p > 1.0e-10){
				// loglaw
				//FLOAT U_bp_p = U_ip_p * logf(z1/user_init::z0) / logf(z2/user_init::z0);

				// Spalding law
				FLOAT U_tau = U_ip_p / 30.0;
				for (int itr=0; itr<30; itr++){
					U_tau  = U_tau - wall_func(U_tau, U_ip_p, z2) / d_wall_func_dut(U_tau, U_ip_p, z2);
				}
				assert(!isnan(U_tau));
				assert(U_tau > 0.0 );

				FLOAT U_bp_p = U_ip_p;
				for (int itr=0; itr<10; itr++){
					U_bp_p = U_bp_p - wall_func(U_tau, U_bp_p, z1) / d_wall_func_dU(U_tau, U_bp_p, z1);
				}
				assert(U_bp_p >= 0.0);
				//assert(U_bp_p < U_ip_p);
				//assert(U_bp_p / U_ip_p > U_ip_p * z1/ z2);
				//if(U_tau > U_bp_p){
				//	printf("U_ip_p = %20.17f\tU_tau = %20.17f\tU_bp_p = %20.17f\n",U_ip_p, U_tau, U_bp_p);
				//	//assert(U_tau < U_bp_p);
				//}
				assert(!isnan(U_bp_p));

				for (int ii=0; ii<3; ii++){
					u_bp_p[ii] = u_ip_p[ii] * U_bp_p / U_ip_p;
					//u_bp_n[ii] = u_ip_n[ii] * z1 / z2;
					//u_bp_n[ii] = -u_bp_n[ii];
				}
				FLOAT U_bp_p_c = (FLOAT)sqrtf(u_bp_p[0]*u_bp_p[0] + u_bp_p[1]*u_bp_p[1] + u_bp_p[2]*u_bp_p[2]);
				FLOAT U_bp_n_c = (FLOAT)sqrtf(u_bp_n[0]*u_bp_n[0] + u_bp_n[1]*u_bp_n[1] + u_bp_n[2]*u_bp_n[2]);
				assert(U_bp_p_c <= U_ip_p);
				assert(U_bp_n_c <= U_ip_n);
				for (int ii=0; ii<3; ii++){
					u_bp[ii] = (u_bp_p[ii] + u_bp_n[ii]) / c_ref;
					assert(!isnan(u_bp[ii]));
				}
				FLOAT U_ip_c = (FLOAT)sqrtf(u_ip[0]*u_ip[0] + u_ip[1]*u_ip[1] + u_ip[2]*u_ip[2]);
				FLOAT U_bp_c = (FLOAT)sqrtf(u_bp[0]*u_bp[0] + u_bp[1]*u_bp[1] + u_bp[2]*u_bp[2]);
				assert(U_bp_c < U_ip_c );
			}else{
				u_bp[0] = us;
				u_bp[1] = vs;
				u_bp[2] = ws;
			}

			// rho_bc
			if(p_ip_v[2]>(FLOAT)0.0){
				rho_bc = (FLOAT)1.0/((FLOAT)1.0-u_bp_n[2]*p_ip_v[2]/c_ref) * (fs[0]+fs[1]+fs[2]+fs[3]+fs[4]+fs[7]+fs[8]+fs[9]+fs[10] + 2*(fs[6]+fs[13]+fs[14]+fs[17]+fs[18]+fs[22]+fs[23]+fs[24]+fs[26]));
			}
			if(p_ip_v[2]<(FLOAT)0.0){
				rho_bc = (FLOAT)1.0/((FLOAT)1.0-u_bp_n[2]*p_ip_v[2]/c_ref) * (fs[0]+fs[1]+fs[2]+fs[3]+fs[4]+fs[7]+fs[8]+fs[9]+fs[10] + 2*(fs[5]+fs[11]+fs[12]+fs[15]+fs[16]+fs[19]+fs[20]+fs[21]+fs[25]));
			}
			assert(!isnan(rho_bc));
			// calculate equilibrium part
			mathLib_LBM::Device_LBM_D3Q27_feq(rho_bc, u_bp[0], u_bp[1], u_bp[2], f_bp_eq);
			// calculate fs_deq from stress
			for (int ii=0; ii<27; ii++){
				f_bp_deq[ii] = fs[ii] - f_bp_eq[ii];
			}
			//mathLib_LBM::Device_LBM_D3Q27_rho(rho_bc, f_bp);
			//mathLib_LBM::Device_LBM_D3Q27_d_equivalent(f_bp_deq, rho_bc, f_bp);
			//for (int ii=0; ii<27; ii++){
			//	if(l_obs[idg_lbm[ii]] > (FLOAT)0.0){
			//		f_bp_deq[ii] = f_bp_deq[direction_lbm[ii]];
			//	}
			//}
			//const FLOAT vis_bc = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rho_bc, vis[id_c0_c0_c0], f_bp_deq);
			//assert(!isnan(vis_bc));
			//const FLOAT lambda_bc = 1 / ( (vis[id_c0_c0_c0] + vis_bc) * 3.0 + 0.5 );
			//assert(!isnan(lambda_bc));
			FLOAT p11=(FLOAT)0.0, p12=(FLOAT)0.0, p13=(FLOAT)0.0, p22=(FLOAT)0.0, p23=(FLOAT)0.0, p33=(FLOAT)0.0;
			for (int ii=0; ii<27; ii++){
				p11 += (eD3Q27_x[ii]*eD3Q27_x[ii] -(FLOAT)1.0/(FLOAT)3.0 )*f_bp_deq[ii];
				p22 += (eD3Q27_y[ii]*eD3Q27_y[ii] -(FLOAT)1.0/(FLOAT)3.0 )*f_bp_deq[ii];
				p33 += (eD3Q27_z[ii]*eD3Q27_z[ii] -(FLOAT)1.0/(FLOAT)3.0 )*f_bp_deq[ii];
				p12 += (eD3Q27_x[ii]*eD3Q27_y[ii]                        )*f_bp_deq[ii];
				p13 += (eD3Q27_x[ii]*eD3Q27_z[ii]                        )*f_bp_deq[ii];
				p23 += (eD3Q27_y[ii]*eD3Q27_z[ii]                        )*f_bp_deq[ii];
			}
			FLOAT sum_deq = (FLOAT)0.0;
			for (int ii=0; ii<27; ii++){
				f_bp_deq[ii] =   (( eD3Q27_x[ii]*eD3Q27_x[ii] - (FLOAT)1.0/(FLOAT)3.0 ) * p11
						        + ( eD3Q27_x[ii]*eD3Q27_y[ii]                         ) * p12
						        + ( eD3Q27_x[ii]*eD3Q27_z[ii]                         ) * p13
						        + ( eD3Q27_y[ii]*eD3Q27_x[ii]                         ) * p12
						        + ( eD3Q27_y[ii]*eD3Q27_y[ii] - (FLOAT)1.0/(FLOAT)3.0 ) * p22
						        + ( eD3Q27_y[ii]*eD3Q27_z[ii]                         ) * p23
						        + ( eD3Q27_z[ii]*eD3Q27_x[ii]                         ) * p13
						        + ( eD3Q27_z[ii]*eD3Q27_y[ii]                         ) * p23
						        + ( eD3Q27_z[ii]*eD3Q27_z[ii] - (FLOAT)1.0/(FLOAT)3.0 ) * p33) * eD3Q27_w[ii] * (FLOAT)9.0 / (FLOAT)2.0;
				sum_deq += f_bp_deq[ii];
			}
			assert( !isnan(sum_deq) );
			assert( sum_deq <  1e-9 );
			assert( sum_deq > -1e-9 );

			// replace distribution function 
			rhos = rho_bc;
			FLOAT fs_sum_tmp = (FLOAT)0.0;
			for (int ii=0; ii<27; ii++){
				fs[ii] = f_bp_eq[ii] + f_bp_deq[ii];
				//fs_sum_tmp += fs[ii];
				assert(!isnan(fs[ii]));
			}
			//assert(fs_sum_tmp < rhos+(FLOAT)0.1 );
		}
		us = u_bp[0];
		vs = u_bp[1];
		ws = u_bp[2];
		

	// force : nuoyancy:Boussinesq approximation
	const int id = id_c0_c0_c0;
	const int stride[3] = { 1, nx, nx*ny };
	const FLOAT  rho_gas = coefficient::DENSITY_AIR;
	const FLOAT  gravity = coefficient::GRAVITY;	
	const FLOAT  beta       = (FLOAT)1.0 / BASE_TEMPERATURE;
	const FLOAT  nondim_lbm = ((FLOAT)1.0/c_ref/rho_gas) * dt;
	FLOAT  force_lbm[NUM_DIRECTION_VEL];
	FLOAT  force_x = (FLOAT)0.0;
	FLOAT  force_y = (FLOAT)0.0;
	FLOAT  force_z = (FLOAT)0.0;
// pressure gradient
	if(user_flags::flg_dpdx==1)	{force_x = force_x - user_init::dpdx * nondim_lbm;}	//MOD2019
	if(user_flags::flg_dpdy==1)	{force_y = force_y - user_init::dpdy * nondim_lbm;}	//MOD2019
	if(user_flags::flg_coriolis==1) {
		const FLOAT pi=(FLOAT)3.14159265;
		force_x = force_x + (FLOAT)2.0 * user_init::angular_velocity * vs * c_ref * sin(user_init::latitude/(FLOAT)180.0 * pi) * nondim_lbm ;
		force_y = force_y - (FLOAT)2.0 * user_init::angular_velocity * us * c_ref * sin(user_init::latitude/(FLOAT)180.0 * pi) * nondim_lbm ;
	}
// buoyancy
	if(user_flags::flg_buoyancy==1) {force_z = force_z + rho_gas*fabs(gravity)*beta*(T[id] - BASE_TEMPERATURE) * nondim_lbm;}

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

	// OUTPUT
	u[id] = us;
	v[id] = vs;
	w[id] = ws;
	rho[id] = rhos;

	// source term force_x,y,z -> force_lbm(i)

	// sgs model : fs, feq -> csvis
	const FLOAT     csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
	vis_sgs[id_c0_c0_c0] = csvis;

	// relaxation time
	const FLOAT     tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
	const FLOAT     omega     = (FLOAT)1.0/tau_total;               // 1.0/tau
// cumulant LBM (collision process)
	// cumulant //
	//FLOAT fs_cum[27], fIs_cum[27];
	//FLOAT fIs[27];

	//fs_cum[0]  = fs[26];
	//fs_cum[1]  = fs[18];
	//fs_cum[2]  = fs[23];
	//fs_cum[3]  = fs[14];
	//fs_cum[4]  = fs[6];
	//fs_cum[5]  = fs[13];
	//fs_cum[6]  = fs[24];
	//fs_cum[7]  = fs[17];
	//fs_cum[8]  = fs[22];
	//fs_cum[9]  = fs[10];
	//fs_cum[10] = fs[4];
	//fs_cum[11] = fs[9];
	//fs_cum[12] = fs[2];
	//fs_cum[13] = fs[0];
	//fs_cum[14] = fs[1];
	//fs_cum[15] = fs[8];
	//fs_cum[16] = fs[3];
	//fs_cum[17] = fs[7];
	//fs_cum[18] = fs[25];
	//fs_cum[19] = fs[16];
	//fs_cum[20] = fs[21];
	//fs_cum[21] = fs[12];
	//fs_cum[22] = fs[5];
	//fs_cum[23] = fs[11];
	//fs_cum[24] = fs[20];
	//fs_cum[25] = fs[15];
	//fs_cum[26] = fs[19];

	//FuncCumulantLBM::fs_cumulant_lbm(fIs_cum, fs_cum, omega, rhos, us,vs,ws);

	//fIs[0]  = fIs_cum[13];
	//fIs[1]  = fIs_cum[14];
	//fIs[2]  = fIs_cum[12];
	//fIs[3]  = fIs_cum[16];
	//fIs[4]  = fIs_cum[10];
	//fIs[5]  = fIs_cum[22];
	//fIs[6]  = fIs_cum[4];
	//fIs[7]  = fIs_cum[17];
	//fIs[8]  = fIs_cum[15];
	//fIs[9]  = fIs_cum[11];
	//fIs[10] = fIs_cum[9];
	//fIs[11] = fIs_cum[23];
	//fIs[12] = fIs_cum[21];
	//fIs[13] = fIs_cum[5];
	//fIs[14] = fIs_cum[3];
	//fIs[15] = fIs_cum[25];
	//fIs[16] = fIs_cum[19];
	//fIs[17] = fIs_cum[7];
	//fIs[18] = fIs_cum[1];
	//fIs[19] = fIs_cum[26];
	//fIs[20] = fIs_cum[24];
	//fIs[21] = fIs_cum[20];
	//fIs[22] = fIs_cum[8];
	//fIs[23] = fIs_cum[2];
	//fIs[24] = fIs_cum[6];
	//fIs[25] = fIs_cum[18];
	//fIs[26] = fIs_cum[0];


	// cumulant //

#pragma unroll
	// add collision and source terms
	//	fnew = fs - 1/tau * (fs - feq)
	for (int ii=0; ii<27; ii++) {
		fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
		//fn[nxyz*ii + id_c0_c0_c0] = fIs[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
	}

	// thermal convection (MOD 2018) //

	if(user_flags::flg_scalar==0) { continue; }	//MOD2019

	if (l_obs[id] >= (FLOAT)0.0) { continue; }

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
	const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;

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

__global__ void
cuda_stream_collision_T_D3Q27_cum (
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
	LBM_solid_boundary (
                        fs,
                        f,
                        l_obs,
                        idg_lbm,
                        direction_lbm,
                        27,
                        nxyz
                        );

	// calculate density (rho)
	mathLib_LBM::Device_LBM_D3Q27_rho (rhos, fs);
	// f,F -> u,v,w,rho (for calculation of collistion operator) //
	FLOAT us,vs,ws;
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
// buoyancy
	if(user_flags::flg_buoyancy==1) {force_z = force_z + rho_gas*fabs(gravity)*beta*(T[id] - BASE_TEMPERATURE) * nondim_lbm;}
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

	// OUTPUT
	u[id] = us;
	v[id] = vs;
	w[id] = ws;
	rho[id] = rhos;

	// source term force_x,y,z -> force_lbm(i)

	// sgs model : fs, feq -> csvis
	const FLOAT     csvis = mathLib_LBM::Device_LBM_D3Q27_SGS_viscosity_deq(Fcs[id_c0_c0_c0], rhos, vis[id_c0_c0_c0], fs_deq);
	vis_sgs[id_c0_c0_c0] = csvis;

	// relaxation time
	const FLOAT     tau_total = (FLOAT)3.0*(vis[id_c0_c0_c0] + csvis) + (FLOAT)0.5;
	const FLOAT     omega     = (FLOAT)1.0/tau_total;               // 1.0/tau

// cumulant LBM (collision process)
	// cumulant //
	FLOAT fs_cum[27], fIs_cum[27];
	FLOAT fIs[27];

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
#pragma unroll
	// add collision and source terms
	//	fnew = fs - 1/tau * (fs - feq)
	for (int ii=0; ii<27; ii++) {
	//	fn[nxyz*ii + id_c0_c0_c0] = fs[ii] - omega*fs_deq[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
		fn[nxyz*ii + id_c0_c0_c0] = fIs[ii] + (FLOAT(1.0)-omega*(FLOAT)0.5)*force_lbm[ii];
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
	const FLOAT Tzr = (l_obs[id+stride[2]] < 0.0) ? (T[id+stride[2]] - T[id])/dx : 0.0;

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
	const int    l_obs_x,
	const int    l_obs_y,
	const int    l_obs_z,
	const int    idg_lbm[],
	const int*   direction_lbm_slip[],
	const int    num_direction_vel,
	const int    nx,
	const int    ny,
	const int    nz,
	const int    nn
	)
{
	for(int ii=0; ii<num_direction_vel; ii++){
		if (l_obs[idg_lbm[ii]] > 0.0 && l_obs[idg_lbm[0] < 0.0]){
			int id = idg_lbm[0] + l_obs_x + nx*l_obs_y + nx*ny*l_obs_z;
			int obs = l_obs_x*l_obs_x + l_obs_y*l_obs_y*2 + l_obs_z*l_obs_z*4;
			int direction = direction_lbm_slip[obs][ii];
			// bounce back
			//fs[ii] = f[direction_lbm[ii]*nn + idg_lbm[0]];
			fs[ii] = f[direction*nn + id];

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
inline __device__ FLOAT wall_func(FLOAT ut, FLOAT U, FLOAT y){
	FLOAT k = 0.4;
	FLOAT B = 5.5;
	FLOAT v = coefficient::KVIS_AIR;
	return U/ut + exp(-k*B)*(exp(k*U/ut) -1.0 -(k*U/ut) -(k*k*U*U/(2.0*ut*ut)) -(k*k*k*U*U*U/(6.0*ut*ut*ut)) )  - (ut*y/v);
}

inline __device__ FLOAT d_wall_func_dut(FLOAT ut, FLOAT U, FLOAT y){
	FLOAT k = 0.4;
	FLOAT B = 5.5;
	FLOAT v = coefficient::KVIS_AIR;
	return -U/(ut*ut) + exp(-k*B)*( -k*U*exp(k*U/ut)/(ut*ut) +k*U/(ut*ut) + k*k*U*U/(ut*ut*ut) + (k*k*k*U*U*U/(2.0*ut*ut*ut*ut))  ) - y/v;
}

inline __device__ FLOAT d_wall_func_dU(FLOAT ut, FLOAT U, FLOAT y){
	FLOAT k = 0.4;
	FLOAT B = 5.5;
	FLOAT v = coefficient::KVIS_AIR;
	return 1.0/ut -exp(-k*B)*(k/ut + k*k*U/(ut*ut) +k*k*k*U*U/(2.0*ut*ut*ut) -k*exp(k*U/ut)/ut);
	//return 1.0/ut +((k*exp(-k*B))/ut)*( exp(k*U/ut) -1.0 -(k*U)/ut - (k*k*U*U)/(2.0*ut*ut) );
}
