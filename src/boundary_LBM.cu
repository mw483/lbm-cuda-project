#include "boundary_LBM.h"

#include "defineCUDA.h"
#include "defineBoundaryFlag.h"
#include "defineLBM.h"
#include "mathLib_particle.h"
#include "mathLib_LBM.h"

#include "defineCoefficient.h"
#include "Define_user.h"

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


__global__ void
boundary_LBM_y_Neumann (
	int		rank_y,
	int		ncpu_y,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_z  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_x >= nx || id_z >= nz)	{ return; }

	// variables *****
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step
	FLOAT	feq[NUM_DIRECTION_VEL];


	// calculation
	if (rank_y == 0) {
		id_g   = id_x + nxy*id_z;

		id_g_r = id_g + nx;
		id_g_w = id_g;

		// r, u, v, w //
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq(rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq(rhos, us, vs, ws, feq);
#endif

		// update //
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	if (rank_y == (ncpu_y-1)) {
		id_g   = id_x + nxy*id_z;

		id_g_r = id_g + nx*(ny-2);
		id_g_w = id_g + nx*(ny-1);

		// r, u, v, w //
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq(rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq(rhos, us, vs, ws, feq);
#endif

		// update //
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
}


__global__ void
boundary_LBM_z_Neumann (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step
	FLOAT	feq[NUM_DIRECTION_VEL];


	// calculation
	if (rank_z == 0) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy;
		id_g_w = id_g;

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	if (rank_z == (ncpu_z-1)) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy*(nz-2);
		id_g_w = id_g + nxy*(nz-1);

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
}


__global__ void
boundary_LBM_z_Slip (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step
	FLOAT	feq[NUM_DIRECTION_VEL];


	// calculation
	if (rank_z == 0) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy;
		id_g_w = id_g;

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   = -w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	if (rank_z == (ncpu_z-1)) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy*(nz-2);
		id_g_w = id_g + nxy*(nz-1);

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   = -w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
}


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
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r;
	int			id_g_w;

	int			id_g_w2;
	int			id_g_w3;
	int			id_g_w4;

	const FLOAT	w_ratio = 0.5;
	const FLOAT	weight2 = 0.5;
	const FLOAT	weight3 = weight2*w_ratio;
	const FLOAT	weight4 = weight3*w_ratio;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step
	FLOAT	Ts;	// 2018
	FLOAT	feq[NUM_DIRECTION_VEL];

	// calculation
	if (rank_z == (ncpu_z-1)) {
		id_g = id_x + nx*id_y;

		id_g_r  = id_g + nxy*(nz-2);
		id_g_w  = id_g + nxy*(nz-1);

		id_g_w2 = id_g + nxy*(nz-2);
		id_g_w3 = id_g + nxy*(nz-3);
		id_g_w4 = id_g + nxy*(nz-4);

		// rho, us, vs, ws
		rhos = r[id_g_r];
//		rhos = rho_ref;			// decrease T at z_top
		us   =  u_ref;
		//us   =  u[id_g_r];
		vs   =  v[id_g_r];
		//vs   = (FLOAT)0.0;
		ws   =  w[id_g_r];
		//ws   = (FLOAT)0.0;
//		Ts   =  315.0;
		Ts   =  T[id_g_r];

/*
		rhos =  rho_ref;
		us   =  u_ref;
//		us   =	u[id_g_w2];
		vs   =  0.0;
		ws   =  0.0;
		// T	2018
		Ts   =  pt_ref;
*/
//		rhos =  r[id_g_r];
//		us   =  2.0*u_ref - u[id_g_r];
//		vs   =  v[id_g_r];
//		ws   = -w[id_g_r];


#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update

		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
//		u[id_g_w2] =  us;
//              u[id_g_w3] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		T[id_g_w]  =  Ts;	// 2018
//		T[id_g_w2] =  Ts;	// 2018
//		T[id_g_w3] =  Ts;	// 2018

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {

		f[nxyz*ii + id_g_w]  = feq[ii];

//			f[nxyz*ii + id_g_w]  = f[nxyz*ii + id_g_r];

// MOD2018 //
/*                // z //
                        f[nxyz*5 + id_g_w] = f[nxyz*6 + id_g_r];
                        f[nxyz*6 + id_g_w] = f[nxyz*5 + id_g_r];

                // x-z //
                        f[nxyz*11 + id_g_w] = f[nxyz*13 + id_g_r];
                        f[nxyz*13 + id_g_w] = f[nxyz*11 + id_g_r];

                // y-z //
                        f[nxyz*12 + id_g_w] = f[nxyz*14 + id_g_r];
                        f[nxyz*14 + id_g_w] = f[nxyz*12 + id_g_r];

                        f[nxyz*15 + id_g_w] = f[nxyz*17 + id_g_r];
                        f[nxyz*17 + id_g_w] = f[nxyz*15 + id_g_r];

                // x-y-z //
                        f[nxyz*19 + id_g_w] = f[nxyz*22 + id_g_r];
                        f[nxyz*22 + id_g_w] = f[nxyz*19 + id_g_r];

                        f[nxyz*20 + id_g_w] = f[nxyz*24 + id_g_r];
                        f[nxyz*24 + id_g_w] = f[nxyz*20 + id_g_r];

                        f[nxyz*21 + id_g_w] = f[nxyz*23 + id_g_r];
                        f[nxyz*23 + id_g_w] = f[nxyz*21 + id_g_r];

                        f[nxyz*25 + id_g_w] = f[nxyz*26 + id_g_r];
                        f[nxyz*26 + id_g_w] = f[nxyz*25 + id_g_r];

*/
// MOD2018 end //


			// boundary layer //
			f[nxyz*ii + id_g_w2]  = weight2*feq[ii] + (1.0 - weight2)*f[nxyz*ii + id_g_w2];
			f[nxyz*ii + id_g_w3]  = weight3*feq[ii] + (1.0 - weight3)*f[nxyz*ii + id_g_w3];
			f[nxyz*ii + id_g_w4]  = weight4*feq[ii] + (1.0 - weight4)*f[nxyz*ii + id_g_w4];

		}
	}
}


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
	)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
//	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step

// D3Q19:
// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)


	// calculation
	if (rank_z == (ncpu_z-1)) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy*(nz-2);
		id_g_w = id_g + nxy*(nz-1);

		// rho, us, vs, ws
		// update
		r[id_g_w] =   r[id_g_r];
		u[id_g_w] =   u[id_g_r];
		v[id_g_w] =   v[id_g_r];
		w[id_g_w] =  -w[id_g_r];

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = f[nxyz*ii + id_g_r];
		}
		// opposite direction //
		// z //
		f[nxyz*5 + id_g_w] = f[nxyz*6 + id_g_r];
		f[nxyz*6 + id_g_w] = f[nxyz*5 + id_g_r];

		// x-z //
		f[nxyz*11 + id_g_w] = f[nxyz*13 + id_g_r];
		f[nxyz*13 + id_g_w] = f[nxyz*11 + id_g_r];

		// y-z //
		f[nxyz*12 + id_g_w] = f[nxyz*14 + id_g_r];
		f[nxyz*14 + id_g_w] = f[nxyz*12 + id_g_r];

		f[nxyz*15 + id_g_w] = f[nxyz*17 + id_g_r];
		f[nxyz*17 + id_g_w] = f[nxyz*15 + id_g_r];
	}
}


__global__ void
boundary_LBM_z_D3Q27_Upper (
	int		rank_z,
	int		ncpu_z,
	FLOAT	*f,
	FLOAT	*r,
	FLOAT	*u,
	FLOAT	*v,
	FLOAT	*w,
	int nx, int ny, int nz
	)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
//	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step

// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)


	// calculation
	if (rank_z == (ncpu_z-1)) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy*(nz-2);
		id_g_w = id_g + nxy*(nz-1);

		// rho, us, vs, ws
		// update
		r[id_g_w] =   r[id_g_r];
		u[id_g_w] =   u[id_g_r];
		v[id_g_w] =   v[id_g_r];
		w[id_g_w] =  -w[id_g_r];

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = f[nxyz*ii + id_g_r];
		}
		// opposite direction //
		// z //
		f[nxyz*5 + id_g_w] = f[nxyz*6 + id_g_r];
		f[nxyz*6 + id_g_w] = f[nxyz*5 + id_g_r];

		// x-z //
		f[nxyz*11 + id_g_w] = f[nxyz*13 + id_g_r];
		f[nxyz*13 + id_g_w] = f[nxyz*11 + id_g_r];

		// y-z //
		f[nxyz*12 + id_g_w] = f[nxyz*14 + id_g_r];
		f[nxyz*14 + id_g_w] = f[nxyz*12 + id_g_r];

		f[nxyz*15 + id_g_w] = f[nxyz*17 + id_g_r];
		f[nxyz*17 + id_g_w] = f[nxyz*15 + id_g_r];

		// x-y-z //
		f[nxyz*19 + id_g_w] = f[nxyz*22 + id_g_r];
		f[nxyz*22 + id_g_w] = f[nxyz*19 + id_g_r];

		f[nxyz*20 + id_g_w] = f[nxyz*24 + id_g_r];
		f[nxyz*24 + id_g_w] = f[nxyz*20 + id_g_r];

		f[nxyz*21 + id_g_w] = f[nxyz*23 + id_g_r];
		f[nxyz*23 + id_g_w] = f[nxyz*21 + id_g_r];

		f[nxyz*25 + id_g_w] = f[nxyz*26 + id_g_r];
		f[nxyz*26 + id_g_w] = f[nxyz*25 + id_g_r];
	}
}


// inflow & outflow //
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
//	FLOAT	dx,		// MOD2019
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_z  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_y >= ny || id_z >= nz)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws,   Ts; // variables after streaming step  2018
	FLOAT	feq[NUM_DIRECTION_VEL];

	FLOAT	const	T0 = BASE_TEMPERATURE;		// K	2018
	FLOAT	const	dTdz = user_init::DTDZ_HIGH;	// K/m	2018
	FLOAT	const	dz = 2.0;			// m	2018
	FLOAT	const	kzi = user_init::ZHIGH;		// grid 2018

	// calculation
	if (rank_x == 0) {
		id_g = nx*id_y + nxy*id_z;

		// read write
		id_g_r = id_g + 1;
		id_g_w = id_g;

		// rho, us, vs, ws
		rhos =  rho_ref;
		us   =  u_ref;
		vs   =  0.0;
		ws   =  0.0;
//		Ts   =  T0 + dTdz * dz * (FLOAT)id_z / 4.0;	// MOD 2018
		Ts   =  T0 + (dTdz * dz * fmax((FLOAT)id_z-(FLOAT)kzi/dz,0)); 	//MOD 2018

//		rhos =  r[id_g_r];
//		us   =  u[id_g_r];
//		vs   =  v[id_g_r];
//		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		T[id_g_w] =  Ts;	// 2018

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	if (rank_x == (ncpu_x-1)) {
		id_g = nx*id_y + nxy*id_z;

		id_g_r = id_g + (nx-2);
		id_g_w = id_g + (nx-1);

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

		Ts   =  T[id_g_r];	// 2018

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		T[id_g_w] =  Ts;	// 2018

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
}


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
	)
{
	const int	nxy = nx*ny;
	const int	id_y  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_z  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_y >= ny || id_z >= nz)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step
	FLOAT	feq[NUM_DIRECTION_VEL];


	// calculation
	if (rank_x == 0) {
		id_g = nx*id_y + nxy*id_z;

		// read write
		id_g_r = id_g + 1;
		id_g_w = id_g;

		// rho, us, vs, ws
		rhos =  rho_ref;
		us   =  u_ref;
		vs   =  0.0;
		ws   =  0.0;

//		rhos =  r[id_g_r];
//		us   =  u[id_g_r];
//		vs   =  v[id_g_r];
//		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	if (rank_x == (ncpu_x-1)) {
		id_g = nx*id_y + nxy*id_z;

		id_g_r = id_g + (nx-2);
		id_g_w = id_g + (nx-1);

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = f[nxyz*ii + id_g_r];
		}
	}
}


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
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_y  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws; // variables after streaming step
	FLOAT	feq[NUM_DIRECTION_VEL];


	// calculation
	if (rank_z == 0) {
		id_g = id_x + nx*id_y;

		// read write
		id_g_r = id_g + nxy;
		id_g_w = id_g;

		// rho, us, vs, ws
		rhos =  rho_ref;
		us   =  0.0;
		vs   =  0.0;
		ws   =  u_ref;

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	if (rank_z == (ncpu_z-1)) {
		id_g = id_x + nx*id_y;

		id_g_r = id_g + nxy*(nz-2);
		id_g_w = id_g + nxy*(nz-1);

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
}

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
//	FLOAT	dx,		// MOD2019
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y  = threadIdx.x + blockDim.x*(blockIdx.x),
				id_z  = threadIdx.y + blockDim.y*(blockIdx.y);
	int			id_g;
	int			id_g_r, id_g_w;

	if (id_y >= ny || id_z >= nz)	{ return; }

	// variables //
	const int	nxyz = nx*ny*nz;
	FLOAT	rhos,  us,  vs,  ws,   Ts; // variables after streaming step  2018
	FLOAT	feq[NUM_DIRECTION_VEL];

	FLOAT	const	T0 = BASE_TEMPERATURE;		// K	2018
	FLOAT	const	dTdz = user_init::DTDZ_LOW;	// K/m	2018
	FLOAT	const	dz = 20.0;			// m	2018
	FLOAT	const	kzi = user_init::ZHIGH;		// grid 2018

	// calculation
	// inflow
	if (rank_x == 0) {
		id_g = nx*id_y + nxy*id_z;

		// read write
		id_g_r = id_g + 60;
		id_g_w = id_g;

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];
		Ts   =  T[id_g_r];
		//for (int ii=0; ii<NUM_DIRECTION_VEL; ii++){
		//	feq[ii] = f[nxyz*ii + id_g_r];
		//}
#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;
		T[id_g_w] =  Ts;	// 2018

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
	
	// outflow
	if (rank_x == (ncpu_x-1)) {
		id_g = nx*id_y + nxy*id_z;

		id_g_r = id_g + (nx-2);
		id_g_w = id_g + (nx-1);

		// rho, us, vs, ws
		rhos =  r[id_g_r];
		us   =  u[id_g_r];
		vs   =  v[id_g_r];
		ws   =  w[id_g_r];

		Ts   =  T[id_g_r];	// 2018

#ifdef D3Q19_MODEL_
		mathLib_LBM::Device_LBM_D3Q19_feq (rhos, us, vs, ws, feq);
#endif
#ifdef D3Q27_MODEL_
		mathLib_LBM::Device_LBM_D3Q27_feq (rhos, us, vs, ws, feq);
#endif

		// update
		r[id_g_w] =  rhos;
		u[id_g_w] =  us;
		v[id_g_w] =  vs;
		w[id_g_w] =  ws;

		T[id_g_w] =  Ts;	// 2018

		for (int ii=0; ii<NUM_DIRECTION_VEL; ii++) {
			f[nxyz*ii + id_g_w] = feq[ii];
		}
	}
}
