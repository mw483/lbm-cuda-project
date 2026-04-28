#include "mathLib_LBM.h"


namespace	
mathLib_LBM {


#include "defineSGS.h"
#include "defineLBM.h"


template <typename T>
inline __device__ void
device_lbm_feq (
	FLOAT	rho,	// input
	FLOAT	us,
	FLOAT	vs,
	FLOAT	ws,
	FLOAT	feq[],	// output
	T		lbm_model
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		Device_LBM_D3Q19_feq (
			rho,
			us,
			vs,
			ws,
			feq
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		Device_LBM_D3Q27_feq (
			rho,
			us,
			vs,
			ws,
			feq
			);
	}
}


// feq
inline __device__ void
Device_LBM_D3Q19_feq (
	FLOAT	rho,	// input
	FLOAT	us,
	FLOAT	vs,
	FLOAT	ws,
	FLOAT	feq[]	// output
	)
{
// D3Q19:
// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)

// f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )

	const FLOAT	vel2   = (us*us + vs*vs + ws*ws);
	const FLOAT	C1D3   = 1.0/ 3.0;
	const FLOAT	C1D18  = 1.0/18.0;
	const FLOAT	C1D36  = 1.0/36.0;

	feq[0 ] = ( (FLOAT)1.0 - (FLOAT)1.5*vel2                                                 ) * C1D3  * rho;

	feq[1 ] = ( (FLOAT)1.0 + (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	feq[2 ] = ( (FLOAT)1.0 - (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	feq[3 ] = ( (FLOAT)1.0 + (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	feq[4 ] = ( (FLOAT)1.0 - (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	feq[5 ] = ( (FLOAT)1.0 + (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	feq[6 ] = ( (FLOAT)1.0 - (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;

	feq[7 ] = ( (FLOAT)1.0 + (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[8 ] = ( (FLOAT)1.0 - (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[9 ] = ( (FLOAT)1.0 + (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[10] = ( (FLOAT)1.0 - (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;

	feq[11] = ( (FLOAT)1.0 + (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[12] = ( (FLOAT)1.0 - (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[13] = ( (FLOAT)1.0 + (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[14] = ( (FLOAT)1.0 - (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;

	feq[15] = ( (FLOAT)1.0 + (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[16] = ( (FLOAT)1.0 - (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[17] = ( (FLOAT)1.0 + (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	feq[18] = ( (FLOAT)1.0 - (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
} 


inline __device__ void
Device_LBM_D3Q27_feq (
	FLOAT	rho,	// input
	FLOAT	us,
	FLOAT	vs,
	FLOAT	ws,
	FLOAT	feq[]	// output
	)
{
// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)

// f_i^{ep} = E_i * rho * (1 + 3 (c_i u_i) + 4.5 (c_i u_i0)^2 - 1.5 (u_i u_i) )

	const FLOAT	vel2 = (us*us + vs*vs + ws*ws);
	const FLOAT	C8D27  = (FLOAT)8.0 / (FLOAT)27.0;
	const FLOAT	C2D27  = (FLOAT)2.0 / (FLOAT)27.0;
	const FLOAT	C1D54  = (FLOAT)1.0 / (FLOAT)54.0;
	const FLOAT	C1D216 = (FLOAT)1.0 / (FLOAT)216.0;

	feq[0 ] = ( (FLOAT)1.0 - (FLOAT)1.5*vel2                                                 ) * C8D27 * rho;

	feq[1 ] = ( (FLOAT)1.0 + (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	feq[2 ] = ( (FLOAT)1.0 - (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	feq[3 ] = ( (FLOAT)1.0 + (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	feq[4 ] = ( (FLOAT)1.0 - (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	feq[5 ] = ( (FLOAT)1.0 + (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	feq[6 ] = ( (FLOAT)1.0 - (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;

	feq[7 ] = ( (FLOAT)1.0 + (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[8 ] = ( (FLOAT)1.0 - (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[9 ] = ( (FLOAT)1.0 + (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[10] = ( (FLOAT)1.0 - (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;

	feq[11] = ( (FLOAT)1.0 + (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[12] = ( (FLOAT)1.0 - (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[13] = ( (FLOAT)1.0 + (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[14] = ( (FLOAT)1.0 - (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;

	feq[15] = ( (FLOAT)1.0 + (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[16] = ( (FLOAT)1.0 - (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[17] = ( (FLOAT)1.0 + (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	feq[18] = ( (FLOAT)1.0 - (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;

	feq[19] = ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs+ws) + (FLOAT)4.5*pow( us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[20] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs+ws) + (FLOAT)4.5*pow(-us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[21] = ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs+ws) + (FLOAT)4.5*pow( us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[22] = ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs-ws) + (FLOAT)4.5*pow( us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[23] = ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs-ws) + (FLOAT)4.5*pow( us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[24] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs-ws) + (FLOAT)4.5*pow(-us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[25] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs+ws) + (FLOAT)4.5*pow(-us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	feq[26] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs-ws) + (FLOAT)4.5*pow(-us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
} 


// rho
template <typename T>
inline __device__ void
device_lbm_rho (
	      FLOAT	&rho,
	const FLOAT	*fs,
	T		lbm_model
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		Device_LBM_D3Q19_rho (
			rho,
			fs
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		Device_LBM_D3Q27_rho (
			rho,
			fs
			);
	}
}


inline __device__ void
Device_LBM_D3Q19_rho (
	      FLOAT	&rho,
	const FLOAT	*fs
	)
{
	rho = (FLOAT)0.0;
#pragma unroll
	for (int i=0; i<19; i++) { rho += fs[i]; } // 19
}


inline __device__ void
Device_LBM_D3Q27_rho (
	      FLOAT	&rho,
	const FLOAT	*fs
	)
{
	rho = (FLOAT)0.0;
#pragma unroll
	for (int i=0; i<27; i++) { rho += fs[i]; } // 27
}


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
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		Device_LBM_D3Q19_velocity (
			rho,
			u, v, w,
			fs
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		Device_LBM_D3Q27_velocity (
			rho,
			u, v, w,
			fs
			);
	}
}


inline __device__ void 
Device_LBM_D3Q19_velocity (
	      FLOAT	rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	)
{
// D3Q19:
// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
	const FLOAT	drho = 1.0/rho;

// D3Q19:
	u = ( fs[1] - fs[2] + fs[7 ] - fs[8 ] + fs[9 ] - fs[10] + fs[11] - fs[12] + fs[13] - fs[14] ) * drho; // 10
	v = ( fs[3] - fs[4] + fs[7 ] + fs[8 ] - fs[9 ] - fs[10] + fs[15] - fs[16] + fs[17] - fs[18] ) * drho;
	w = ( fs[5] - fs[6] + fs[11] + fs[12] - fs[13] - fs[14] + fs[15] + fs[16] - fs[17] - fs[18] ) * drho;
}


inline __device__ void 
Device_LBM_D3Q27_velocity (
	      FLOAT	rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	)
{
// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)
	const FLOAT	drho = 1.0/rho;

// D3Q27:
	u = ( fs[1] - fs[2] + fs[7 ] - fs[8 ] + fs[9 ] - fs[10] + fs[11] - fs[12] + fs[13] - fs[14] + fs[19] - fs[20] + fs[21] + fs[22] + fs[23] - fs[24] - fs[25] - fs[26]) * drho; // 
	v = ( fs[3] - fs[4] + fs[7 ] + fs[8 ] - fs[9 ] - fs[10] + fs[15] - fs[16] + fs[17] - fs[18] + fs[19] + fs[20] - fs[21] + fs[22] - fs[23] + fs[24] - fs[25] - fs[26]) * drho;
	w = ( fs[5] - fs[6] + fs[11] + fs[12] - fs[13] - fs[14] + fs[15] + fs[16] - fs[17] - fs[18] + fs[19] + fs[20] + fs[21] - fs[22] - fs[23] - fs[24] + fs[25] - fs[26]) * drho;

}


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
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		Device_LBM_D3Q19_rho_velocity (
			rho,
			u, v, w,
			fs
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		Device_LBM_D3Q27_rho_velocity (
			rho,
			u, v, w,
			fs
			);
	}
}


inline __device__ void 
Device_LBM_D3Q19_rho_velocity (
	      FLOAT	&rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	)
{
	Device_LBM_D3Q19_rho     (rho, fs);
	Device_LBM_D3Q19_velocity(rho, u, v, w, fs);
}



inline __device__ void 
Device_LBM_D3Q27_rho_velocity (
	      FLOAT	&rho,
	      FLOAT	&u,
	      FLOAT	&v,
	      FLOAT	&w,
	const FLOAT	*fs
	)
{
	Device_LBM_D3Q27_rho     (rho, fs);
	Device_LBM_D3Q27_velocity(rho, u, v, w, fs);
} 


// fs - feq
template <typename T>
inline __device__ void
device_lbm_d_equivalent (
	      FLOAT	fs_deq[],
	      FLOAT	&rho, 
	const FLOAT	fs[],
	T		lbm_model
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		Device_LBM_D3Q19_d_equivalent (
			fs_deq,
			rho, 
			fs
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		Device_LBM_D3Q27_d_equivalent (
			fs_deq,
			rho, 
			fs
			);
	}
}


inline __device__ void
Device_LBM_D3Q19_d_equivalent (
	      FLOAT	fs_deq[],
	      FLOAT	&rho, 
	const FLOAT	fs[]
	)
{
// D3Q19:
// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
	FLOAT	us, vs, ws;

	// rho, u,v,w
	Device_LBM_D3Q19_rho_velocity(rho, us, vs, ws, fs);

	const FLOAT	vel2   = (us*us + vs*vs + ws*ws);
	const FLOAT	C1D3   = 1.0/ 3.0;
	const FLOAT	C1D18  = 1.0/18.0;
	const FLOAT	C1D36  = 1.0/36.0;

	fs_deq[0 ] = fs[0 ] - ( (FLOAT)1.0 - (FLOAT)1.5*vel2                                                 ) * C1D3  * rho;
 
	fs_deq[1 ] = fs[1 ] - ( (FLOAT)1.0 + (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	fs_deq[2 ] = fs[2 ] - ( (FLOAT)1.0 - (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	fs_deq[3 ] = fs[3 ] - ( (FLOAT)1.0 + (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	fs_deq[4 ] = fs[4 ] - ( (FLOAT)1.0 - (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	fs_deq[5 ] = fs[5 ] - ( (FLOAT)1.0 + (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
	fs_deq[6 ] = fs[6 ] - ( (FLOAT)1.0 - (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C1D18 * rho;
 
	fs_deq[7 ] = fs[7 ] - ( (FLOAT)1.0 + (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[8 ] = fs[8 ] - ( (FLOAT)1.0 - (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[9 ] = fs[9 ] - ( (FLOAT)1.0 + (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[10] = fs[10] - ( (FLOAT)1.0 - (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
 
	fs_deq[11] = fs[11] - ( (FLOAT)1.0 + (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[12] = fs[12] - ( (FLOAT)1.0 - (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[13] = fs[13] - ( (FLOAT)1.0 + (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[14] = fs[14] - ( (FLOAT)1.0 - (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
 
	fs_deq[15] = fs[15] - ( (FLOAT)1.0 + (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[16] = fs[16] - ( (FLOAT)1.0 - (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[17] = fs[17] - ( (FLOAT)1.0 + (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
	fs_deq[18] = fs[18] - ( (FLOAT)1.0 - (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D36 * rho;
} //  flops

inline __device__ void
Device_LBM_D3Q27_equivalent (
	FLOAT &rho,
	FLOAT &us,
	FLOAT &vs,
	FLOAT &ws,
	FLOAT fs[]
){
	//FLOAT us = *u;
	//FLOAT vs = *v;
	//FLOAT ws = *w;
	//FLAOT rho = *rhos;
	const FLOAT	vel2 = (us*us + vs*vs + ws*ws);
	const FLOAT	C8D27  = (FLOAT)8.0 / (FLOAT)27.0;
	const FLOAT	C2D27  = (FLOAT)2.0 / (FLOAT)27.0;
	const FLOAT	C1D54  = (FLOAT)1.0 / (FLOAT)54.0;
	const FLOAT	C1D216 = (FLOAT)1.0 / (FLOAT)216.0;

	fs[0 ] = ( (FLOAT)1.0 - (FLOAT)1.5*vel2                                                 ) * C8D27 * rho;
 
	fs[1 ] = ( (FLOAT)1.0 + (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs[2 ] = ( (FLOAT)1.0 - (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs[3 ] = ( (FLOAT)1.0 + (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs[4 ] = ( (FLOAT)1.0 - (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs[5 ] = ( (FLOAT)1.0 + (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs[6 ] = ( (FLOAT)1.0 - (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
 
	fs[7 ] = ( (FLOAT)1.0 + (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[8 ] = ( (FLOAT)1.0 - (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[9 ] = ( (FLOAT)1.0 + (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[10] = ( (FLOAT)1.0 - (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
 
	fs[11] = ( (FLOAT)1.0 + (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[12] = ( (FLOAT)1.0 - (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[13] = ( (FLOAT)1.0 + (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[14] = ( (FLOAT)1.0 - (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
 
	fs[15] = ( (FLOAT)1.0 + (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[16] = ( (FLOAT)1.0 - (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[17] = ( (FLOAT)1.0 + (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs[18] = ( (FLOAT)1.0 - (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
 
	fs[19] = ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs+ws) + (FLOAT)4.5*pow( us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[20] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs+ws) + (FLOAT)4.5*pow(-us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[21] = ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs+ws) + (FLOAT)4.5*pow( us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[22] = ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs-ws) + (FLOAT)4.5*pow( us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[23] = ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs-ws) + (FLOAT)4.5*pow( us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[24] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs-ws) + (FLOAT)4.5*pow(-us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[25] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs+ws) + (FLOAT)4.5*pow(-us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs[26] = ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs-ws) + (FLOAT)4.5*pow(-us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
}


inline __device__ void
Device_LBM_D3Q27_d_equivalent (
	      FLOAT	fs_deq[],
	      FLOAT	&rho, 
	const FLOAT	fs[]
	)
{
// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)
	FLOAT	us, vs, ws;

	// rho, u,v,w
	Device_LBM_D3Q27_rho_velocity(rho, us, vs, ws, fs);


	const FLOAT	vel2 = (us*us + vs*vs + ws*ws);
	const FLOAT	C8D27  = (FLOAT)8.0 / (FLOAT)27.0;
	const FLOAT	C2D27  = (FLOAT)2.0 / (FLOAT)27.0;
	const FLOAT	C1D54  = (FLOAT)1.0 / (FLOAT)54.0;
	const FLOAT	C1D216 = (FLOAT)1.0 / (FLOAT)216.0;

	fs_deq[0 ] = fs[0 ] - ( (FLOAT)1.0 - (FLOAT)1.5*vel2                                                 ) * C8D27 * rho;
 
	fs_deq[1 ] = fs[1 ] - ( (FLOAT)1.0 + (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[2 ] = fs[2 ] - ( (FLOAT)1.0 - (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[3 ] = fs[3 ] - ( (FLOAT)1.0 + (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[4 ] = fs[4 ] - ( (FLOAT)1.0 - (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[5 ] = fs[5 ] - ( (FLOAT)1.0 + (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[6 ] = fs[6 ] - ( (FLOAT)1.0 - (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
 
	fs_deq[7 ] = fs[7 ] - ( (FLOAT)1.0 + (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[8 ] = fs[8 ] - ( (FLOAT)1.0 - (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[9 ] = fs[9 ] - ( (FLOAT)1.0 + (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[10] = fs[10] - ( (FLOAT)1.0 - (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
 
	fs_deq[11] = fs[11] - ( (FLOAT)1.0 + (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[12] = fs[12] - ( (FLOAT)1.0 - (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[13] = fs[13] - ( (FLOAT)1.0 + (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[14] = fs[14] - ( (FLOAT)1.0 - (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
 
	fs_deq[15] = fs[15] - ( (FLOAT)1.0 + (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[16] = fs[16] - ( (FLOAT)1.0 - (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[17] = fs[17] - ( (FLOAT)1.0 + (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[18] = fs[18] - ( (FLOAT)1.0 - (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
 
	fs_deq[19] = fs[19] - ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs+ws) + (FLOAT)4.5*pow( us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[20] = fs[20] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs+ws) + (FLOAT)4.5*pow(-us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[21] = fs[21] - ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs+ws) + (FLOAT)4.5*pow( us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[22] = fs[22] - ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs-ws) + (FLOAT)4.5*pow( us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[23] = fs[23] - ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs-ws) + (FLOAT)4.5*pow( us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[24] = fs[24] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs-ws) + (FLOAT)4.5*pow(-us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[25] = fs[25] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs+ws) + (FLOAT)4.5*pow(-us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
	fs_deq[26] = fs[26] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs-ws) + (FLOAT)4.5*pow(-us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
} //  flops


inline __device__ void
Device_LBM_D3Q27_d_equivalent_force (
		FLOAT	fs_deq[],
		FLOAT	&rho,
	const	FLOAT	fs[],
		FLOAT	&force_x,
		FLOAT	&force_y,
		FLOAT	&force_z
        )
{
// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)
        FLOAT   us, vs, ws;
	Device_LBM_D3Q27_rho_velocity(rho, us, vs, ws, fs);
	us = us + force_x*(FLOAT)0.5;		//MOD2018
	vs = vs + force_y*(FLOAT)0.5;		//MOD2018
	ws = ws + force_z*(FLOAT)0.5;		//MOD2018
	const FLOAT     vel2 = (us*us + vs*vs + ws*ws);
	const FLOAT     C8D27  = (FLOAT)8.0 / (FLOAT)27.0;
	const FLOAT     C2D27  = (FLOAT)2.0 / (FLOAT)27.0;
	const FLOAT     C1D54  = (FLOAT)1.0 / (FLOAT)54.0;
	const FLOAT     C1D216 = (FLOAT)1.0 / (FLOAT)216.0;
	fs_deq[0 ] = fs[0 ] - ( (FLOAT)1.0 - (FLOAT)1.5*vel2                                                 ) * C8D27 * rho;
	fs_deq[1 ] = fs[1 ] - ( (FLOAT)1.0 + (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[2 ] = fs[2 ] - ( (FLOAT)1.0 - (FLOAT)3.0*us      + (FLOAT)4.5*pow(us, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[3 ] = fs[3 ] - ( (FLOAT)1.0 + (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[4 ] = fs[4 ] - ( (FLOAT)1.0 - (FLOAT)3.0*vs      + (FLOAT)4.5*pow(vs, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[5 ] = fs[5 ] - ( (FLOAT)1.0 + (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[6 ] = fs[6 ] - ( (FLOAT)1.0 - (FLOAT)3.0*ws      + (FLOAT)4.5*pow(ws, 2)    - (FLOAT)1.5*vel2 ) * C2D27 * rho;
	fs_deq[7 ] = fs[7 ] - ( (FLOAT)1.0 + (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[8 ] = fs[8 ] - ( (FLOAT)1.0 - (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[9 ] = fs[9 ] - ( (FLOAT)1.0 + (FLOAT)3.0*(us-vs) + (FLOAT)4.5*pow(us-vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[10] = fs[10] - ( (FLOAT)1.0 - (FLOAT)3.0*(us+vs) + (FLOAT)4.5*pow(us+vs, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[11] = fs[11] - ( (FLOAT)1.0 + (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[12] = fs[12] - ( (FLOAT)1.0 - (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[13] = fs[13] - ( (FLOAT)1.0 + (FLOAT)3.0*(us-ws) + (FLOAT)4.5*pow(us-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[14] = fs[14] - ( (FLOAT)1.0 - (FLOAT)3.0*(us+ws) + (FLOAT)4.5*pow(us+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[15] = fs[15] - ( (FLOAT)1.0 + (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[16] = fs[16] - ( (FLOAT)1.0 - (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[17] = fs[17] - ( (FLOAT)1.0 + (FLOAT)3.0*(vs-ws) + (FLOAT)4.5*pow(vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
        fs_deq[18] = fs[18] - ( (FLOAT)1.0 - (FLOAT)3.0*(vs+ws) + (FLOAT)4.5*pow(vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D54 * rho;
	fs_deq[19] = fs[19] - ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs+ws) + (FLOAT)4.5*pow( us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[20] = fs[20] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs+ws) + (FLOAT)4.5*pow(-us+vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[21] = fs[21] - ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs+ws) + (FLOAT)4.5*pow( us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[22] = fs[22] - ( (FLOAT)1.0 + (FLOAT)3.0*( us+vs-ws) + (FLOAT)4.5*pow( us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[23] = fs[23] - ( (FLOAT)1.0 + (FLOAT)3.0*( us-vs-ws) + (FLOAT)4.5*pow( us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[24] = fs[24] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us+vs-ws) + (FLOAT)4.5*pow(-us+vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[25] = fs[25] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs+ws) + (FLOAT)4.5*pow(-us-vs+ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
        fs_deq[26] = fs[26] - ( (FLOAT)1.0 + (FLOAT)3.0*(-us-vs-ws) + (FLOAT)4.5*pow(-us-vs-ws, 2) - (FLOAT)1.5*vel2 ) * C1D216 * rho;
} //  flops


// sgs viscosity
template <typename T>
inline __device__ FLOAT 
device_lbm_SGS_viscosity_deq (
	      FLOAT	Fcs,
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[],
	T		lbm_model
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		return
		Device_LBM_D3Q19_SGS_viscosity_deq (
			Fcs,
			rho,
			vis,
			fs_deq
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		return
		Device_LBM_D3Q27_SGS_viscosity_deq (
			Fcs,
			rho,
			vis,
			fs_deq
			);
	}
	else {
		return	0;
	}
}

inline __device__ FLOAT 
Device_LBM_D3Q19_SGS_viscosity_deq (
	      FLOAT	Fcs,
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	)
{
	const FLOAT	Csgs = COEF_SGS;

	// stress
	const FLOAT	ss = Device_LBM_D3Q19_stress_deq(rho, vis, fs_deq); // 101 flops

	// Coherent-structure Smagorinsky model
//	return	Csgs * sqrt(2.0f*ss);
	return	Csgs * pow(Fcs, FLOAT(1.5f)) * sqrt(2.0f*ss); // 5 flops
}


inline __device__ FLOAT 
Device_LBM_D3Q27_SGS_viscosity_deq (
	      FLOAT	Fcs,
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	)
{
	const FLOAT	Csgs = COEF_SGS;

	// stress
	const FLOAT	ss = Device_LBM_D3Q27_stress_deq(rho, vis, fs_deq); // 101 flops

	// Coherent-structure Smagorinsky model
//	return	Csgs * sqrt(2.0f*ss);
	return	Csgs * pow(Fcs, FLOAT(1.5f)) * sqrt(2.0f*ss); // 5 flops
}


// stress tensor
template <typename T>
inline __device__ FLOAT
device_lbm_stress_deq (
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[],
	T		lbm_model
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		return
		Device_LBM_D3Q19_stress_deq (
			rho,
			vis,
			fs_deq
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		return
		Device_LBM_D3Q27_stress_deq (
			rho,
			vis,
			fs_deq
			);
	}
	else {
		return	0;
	}
}


inline __device__ FLOAT
Device_LBM_D3Q19_stress_deq (
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	)
{
// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)

	const FLOAT	tau = 3.0f*vis + 0.5f; // 2
	const FLOAT	A = -1.5f/(rho*tau); // 3
//	FLOAT	s11, s22, s33,
//			s12, s23, s13;

	const FLOAT	s11 = (fs_deq[1 ] + fs_deq[2 ] + fs_deq[7 ] + fs_deq[8 ] + fs_deq[9 ] + fs_deq[10] + fs_deq[11] + fs_deq[12] + fs_deq[13] + fs_deq[14]);
	const FLOAT	s22 = (fs_deq[3 ] + fs_deq[4 ] + fs_deq[7 ] + fs_deq[8 ] + fs_deq[9 ] + fs_deq[10] + fs_deq[15] + fs_deq[16] + fs_deq[17] + fs_deq[18]);
	const FLOAT	s33 = (fs_deq[5 ] + fs_deq[6 ] + fs_deq[11] + fs_deq[12] + fs_deq[13] + fs_deq[14] + fs_deq[15] + fs_deq[16] + fs_deq[17] + fs_deq[18]);

	const FLOAT	s12 = (fs_deq[7 ] - fs_deq[8 ] - fs_deq[9 ] + fs_deq[10]);
	const FLOAT	s13 = (fs_deq[11] - fs_deq[12] - fs_deq[13] + fs_deq[14]);
	const FLOAT	s23 = (fs_deq[15] - fs_deq[16] - fs_deq[17] + fs_deq[18]);

	// SS
	return	( ( s11*s11 + s22*s22 + s33*s33 ) + 2.0f*( s12*s12 + s23*s23 + s13*s13 ) ) * (A*A); // 14
}


inline __device__ FLOAT
Device_LBM_D3Q27_stress_deq (
	      FLOAT	rho,
	      FLOAT	vis,
	const FLOAT	fs_deq[]
	)
{
// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)

	const FLOAT	tau = 3.0f*vis + 0.5f; // 2
	const FLOAT	A = -1.5f/(rho*tau); // 3
//	FLOAT	s11, s22, s33,
//			s12, s23, s13;

	const FLOAT	sum19t26 = (fs_deq[19] + fs_deq[20] + fs_deq[21] + fs_deq[22] + fs_deq[23] + fs_deq[24] + fs_deq[25] + fs_deq[26] );

	const FLOAT	s11 = (fs_deq[1 ] + fs_deq[2 ] + fs_deq[7 ] + fs_deq[8 ] + fs_deq[9 ] + fs_deq[10] + fs_deq[11] + fs_deq[12] + fs_deq[13] + fs_deq[14]) + sum19t26;
	const FLOAT	s22 = (fs_deq[3 ] + fs_deq[4 ] + fs_deq[7 ] + fs_deq[8 ] + fs_deq[9 ] + fs_deq[10] + fs_deq[15] + fs_deq[16] + fs_deq[17] + fs_deq[18]) + sum19t26;
	const FLOAT	s33 = (fs_deq[5 ] + fs_deq[6 ] + fs_deq[11] + fs_deq[12] + fs_deq[13] + fs_deq[14] + fs_deq[15] + fs_deq[16] + fs_deq[17] + fs_deq[18]) + sum19t26;

	const FLOAT	s12 = (fs_deq[7 ] - fs_deq[8 ] - fs_deq[9 ] + fs_deq[10]) + fs_deq[19] - fs_deq[20] - fs_deq[21] + fs_deq[22] - fs_deq[23] - fs_deq[24] + fs_deq[25] + fs_deq[26];
	const FLOAT	s13 = (fs_deq[11] - fs_deq[12] - fs_deq[13] + fs_deq[14]) + fs_deq[19] - fs_deq[20] + fs_deq[21] - fs_deq[22] - fs_deq[23] + fs_deq[24] - fs_deq[25] + fs_deq[26];
	const FLOAT	s23 = (fs_deq[15] - fs_deq[16] - fs_deq[17] + fs_deq[18]) + fs_deq[19] + fs_deq[20] - fs_deq[21] - fs_deq[22] + fs_deq[23] - fs_deq[24] - fs_deq[25] + fs_deq[26];

	// SS
	return	( ( s11*s11 + s22*s22 + s33*s33 ) + 2.0f*( s12*s12 + s23*s23 + s13*s13 ) ) * (A*A); // 14
}


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
	)
{
	if (lbm_model == D3Q19_VELOCITY) {
		Device_LBM_D3Q19_Force (
			rho,
			force_x,
			force_y,
			force_z,
			force_lbm
			);
	}
	else if (lbm_model == D3Q27_VELOCITY) {
		Device_LBM_D3Q27_Force (
			rho,
			force_x,
			force_y,
			force_z,
			force_lbm
			);
	}
}


inline __device__ void 
Device_LBM_D3Q19_Force (
	FLOAT	rho,
	FLOAT	force_x,
	FLOAT	force_y,
	FLOAT	force_z,
	FLOAT	force_lbm[]
	)
{
// D3Q19:
// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)

	const FLOAT	d_cs2 = 3.0; // 1.0/(1.0/3.0)
	const FLOAT	force[3] = { force_x, force_y, force_z };

	const FLOAT	C1D18  = (FLOAT)1.0 / (FLOAT)18.0;
	const FLOAT	C1D36  = (FLOAT)1.0 / (FLOAT)36.0;
	
	force_lbm[0 ] = 0.0;
	force_lbm[1 ] = (   force[0]            ) * d_cs2 * C1D18 * rho;
	force_lbm[2 ] = ( - force[0]            ) * d_cs2 * C1D18 * rho;
	force_lbm[3 ] = (   force[1]            ) * d_cs2 * C1D18 * rho;
	force_lbm[4 ] = ( - force[1]            ) * d_cs2 * C1D18 * rho;
	force_lbm[5 ] = (   force[2]            ) * d_cs2 * C1D18 * rho;
	force_lbm[6 ] = ( - force[2]            ) * d_cs2 * C1D18 * rho;
                                                     
	force_lbm[7 ] = (   force[0] + force[1] ) * d_cs2 * C1D36 * rho;
	force_lbm[8 ] = ( - force[0] + force[1] ) * d_cs2 * C1D36 * rho;
	force_lbm[9 ] = (   force[0] - force[1] ) * d_cs2 * C1D36 * rho;
	force_lbm[10] = ( - force[0] - force[1] ) * d_cs2 * C1D36 * rho;
                                                     
	force_lbm[11] = (   force[0] + force[2] ) * d_cs2 * C1D36 * rho;
	force_lbm[12] = ( - force[0] + force[2] ) * d_cs2 * C1D36 * rho;
	force_lbm[13] = (   force[0] - force[2] ) * d_cs2 * C1D36 * rho;
	force_lbm[14] = ( - force[0] - force[2] ) * d_cs2 * C1D36 * rho;
                                                     
	force_lbm[15] = (   force[1] + force[2] ) * d_cs2 * C1D36 * rho;
	force_lbm[16] = ( - force[1] + force[2] ) * d_cs2 * C1D36 * rho;
	force_lbm[17] = (   force[1] - force[2] ) * d_cs2 * C1D36 * rho;
	force_lbm[18] = ( - force[1] - force[2] ) * d_cs2 * C1D36 * rho;
}


inline __device__ void 
Device_LBM_D3Q27_Force (
	FLOAT	rho,
	FLOAT	force_x,
	FLOAT	force_y,
	FLOAT	force_z,
	FLOAT	force_lbm[]
	)
{
// D3Q27:
// center:     e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
// x-y   face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
// x-z   face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
// y-z   face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
// x-y-z face: e19( 1, 1, 1), e20(-1, 1, 1), e21( 1,-1, 1), e22( 1, 1,-1)
//           : e23( 1,-1,-1), e24(-1, 1,-1), e25(-1,-1, 1), e26(-1,-1,-1)

	const FLOAT	d_cs2 = 3.0; // 1.0/(1.0/3.0)
	const FLOAT	force[3] = { force_x, force_y, force_z };

	const FLOAT	C2D27  = (FLOAT)2.0 / (FLOAT)27.0;
	const FLOAT	C1D54  = (FLOAT)1.0 / (FLOAT)54.0;
	const FLOAT	C1D216 = (FLOAT)1.0 / (FLOAT)216.0;
	
	force_lbm[0 ] = 0.0;
	force_lbm[1 ] = (   force[0]            ) * d_cs2 * C2D27 * rho;
	force_lbm[2 ] = ( - force[0]            ) * d_cs2 * C2D27 * rho;
	force_lbm[3 ] = (   force[1]            ) * d_cs2 * C2D27 * rho;
	force_lbm[4 ] = ( - force[1]            ) * d_cs2 * C2D27 * rho;
	force_lbm[5 ] = (   force[2]            ) * d_cs2 * C2D27 * rho;
	force_lbm[6 ] = ( - force[2]            ) * d_cs2 * C2D27 * rho;
                                                     
	force_lbm[7 ] = (   force[0] + force[1] ) * d_cs2 * C1D54 * rho;
	force_lbm[8 ] = ( - force[0] + force[1] ) * d_cs2 * C1D54 * rho;
	force_lbm[9 ] = (   force[0] - force[1] ) * d_cs2 * C1D54 * rho;
	force_lbm[10] = ( - force[0] - force[1] ) * d_cs2 * C1D54 * rho;
                                                     
	force_lbm[11] = (   force[0] + force[2] ) * d_cs2 * C1D54 * rho;
	force_lbm[12] = ( - force[0] + force[2] ) * d_cs2 * C1D54 * rho;
	force_lbm[13] = (   force[0] - force[2] ) * d_cs2 * C1D54 * rho;
	force_lbm[14] = ( - force[0] - force[2] ) * d_cs2 * C1D54 * rho;
                                                     
	force_lbm[15] = (   force[1] + force[2] ) * d_cs2 * C1D54 * rho;
	force_lbm[16] = ( - force[1] + force[2] ) * d_cs2 * C1D54 * rho;
	force_lbm[17] = (   force[1] - force[2] ) * d_cs2 * C1D54 * rho;
	force_lbm[18] = ( - force[1] - force[2] ) * d_cs2 * C1D54 * rho;

	force_lbm[19] = (   force[0] + force[1] + force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[20] = ( - force[0] + force[1] + force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[21] = (   force[0] - force[1] + force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[22] = (   force[0] + force[1] - force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[23] = (   force[0] - force[1] - force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[24] = ( - force[0] + force[1] - force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[25] = ( - force[0] - force[1] + force[2] ) * d_cs2 * C1D216 * rho;
	force_lbm[26] = ( - force[0] - force[1] - force[2] ) * d_cs2 * C1D216 * rho;
}


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
	)
{
	if (boundary_model == BOUNCE_BACK) {
		device_boundary_bounce_back (
			fs,
			f,
			l_obs,
			idg_lbm,
			direction_lbm,
			num_direction_vel,
			nn
			);
	}
	else if (boundary_model == BOUNDARY_2ND_ORDER) {
		device_boundary_YU (
			fs,
			f,
			l_obs,
			idg_lbm,
			direction_lbm,
			num_direction_vel,
			nn
			);
	}
}


inline __device__ void 
device_boundary_bounce_back (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		if (l_obs[idg_lbm[ii]] > 0.0) {
			fs[ii] = f[direction_lbm[ii]*nn + idg_lbm[0]];
		}
	}
}


// Bouzidi's model //
inline __device__ void 
device_boundary_BOUZIDI (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		const FLOAT	lv0   = l_obs[idg_lbm[0] ];
		const FLOAT	lv_up = l_obs[idg_lbm[ii]];

//		if (lv_up > (FLOAT)0.0 && lv0 < (FLOAT)0.0) {	// fluid < 0, solid > 0 // // diverge
		if (lv_up > (FLOAT)0.0) {	// fluid < 0, solid > 0 //
			const FLOAT	ep      = coefficient::NON_ZERO_EP;
			const FLOAT	delta   = fabs( lv0 ) / ( fabs(lv0) + fabs(lv_up) + ep );	// level-set function (boundary) //
			const FLOAT	delta_2 = delta * (FLOAT)2.0;

			// fs_tmp //
			FLOAT	fs_tmp;
			if (delta < (FLOAT)0.5) {	// 0.0 < delta < 0.5 //
				fs_tmp =    delta_2               * f[direction_lbm[ii]*nn + idg_lbm[0]                ]
						 + ((FLOAT)1.0 - delta_2) * f[direction_lbm[ii]*nn + idg_lbm[direction_lbm[ii]]];
			}
			else {						// 0.5 < delta < 1.0 //
				fs_tmp = (                          f[direction_lbm[ii]*nn + idg_lbm[0]                ]
						 + (delta_2 - (FLOAT)1.0) * f[              ii *nn + idg_lbm[0]                ]  )
						 /  delta_2;
			}

			fs[ii] = fs_tmp;
		}
	}
}


// Yu's model //
inline __device__ void
device_boundary_YU (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		const FLOAT	lv0   = l_obs[idg_lbm[0] ];
		const FLOAT	lv_up = l_obs[idg_lbm[ii]];

//		if (lv_up > (FLOAT)0.0 && lv0 < (FLOAT)0.0) {	// fluid < 0, solid > 0 // diverge
		if (lv_up > (FLOAT)0.0) {	// fluid < 0, solid > 0
			const FLOAT	ep    = coefficient::NON_ZERO_EP;
			const FLOAT	delta = fabs( lv0 ) / ( fabs(lv0) + fabs(lv_up) + ep );	// level-set function (boundary)

			fs[ii] = (   delta      * f[direction_lbm[ii]*nn + idg_lbm[0]                ]
				   	 +	(1.0-delta) * f[direction_lbm[ii]*nn + idg_lbm[direction_lbm[ii]]]
					 +   delta      * f[              ii *nn + idg_lbm[0]                ]  ) 
					/ ( (FLOAT)1.0 + delta );
		}
	}
}


// 2nd polynomial //
inline __device__ void
device_boundary_2nd_poly (
	      FLOAT	fs[],
	const FLOAT	*f,
	const FLOAT	*l_obs,
	const int	idg_lbm[],
	const int	direction_lbm[],
	      int	num_direction_vel,
	      int	nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		const FLOAT	lv0   = l_obs[idg_lbm[0] ];
		const FLOAT	lv_up = l_obs[idg_lbm[ii]];

//		if (lv_up > (FLOAT)0.0 && lv0 < (FLOAT)0.0) {	// fluid < 0, solid > 0 // diverge
		if (lv_up > (FLOAT)0.0) {	// fluid < 0, solid > 0
			const FLOAT	ep    = coefficient::NON_ZERO_EP;
			const FLOAT	delta = fabs( lv0 ) / ( fabs(lv0) + fabs(lv_up) + ep );	// level-set function (boundary)

			const FLOAT	coef = (1.0 - 2.0*delta) / (1.0 + 2.0*delta);	// coef > 0 (delta  < 0.5), coef < 0 (delta > 0.5)

			fs[ii] = (         f[direction_lbm[ii]*nn + idg_lbm[0]                ]
				   	 +	coef * f[direction_lbm[ii]*nn + idg_lbm[direction_lbm[ii]]]
					 -  coef * f[              ii *nn + idg_lbm[0]                ]  );
		}
	}
}


// index global
inline __device__ int Index_Global(int id_x, int id_y, int id_z, 
	int x_offset, int y_offset, int z_offset,
	int  nx, int  ny, int  nz)
{
	int		id_xc = id_x + x_offset,
			id_yc = id_y + y_offset,
			id_zc = id_z + z_offset;

	// periodic
	if      (id_xc < 0   )	{ id_xc = (id_xc + nx)%nx; }
	else if (id_xc > nx-1)	{ id_xc = (id_xc     )%nx; }

	if      (id_yc < 0   )	{ id_yc = (id_yc + ny)%ny; }
	else if (id_yc > ny-1)	{ id_yc = (id_yc     )%ny; }

	if      (id_zc < 0   )	{ id_zc = (id_zc + nz)%nz; }
	else if (id_zc > nz-1)	{ id_zc = (id_zc     )%nz; }

	
	return	id_xc + nx*id_yc + nx*ny*id_zc;
}

//(YOKOUCHI 2020)
// total TKE for Lagrangian Stochastic Model

inline __device__ FLOAT  
Device_LBM_D3Q27_totalTKE (
	FLOAT	fs_deq[],
	FLOAT	rho
	)
{
//	const FLOAT	drho = 1.0/rho;
	FLOAT	rho_deq	= 0.0;
	for (int i=0; i<27; i++){
		rho_deq += fs_deq[i];
	}
	const FLOAT	drho = 1.0/rho_deq;

// D3Q27:

	FLOAT us = ( fs_deq[1] - fs_deq[2] + fs_deq[7 ] - fs_deq[8 ] + fs_deq[9 ] - fs_deq[10] + fs_deq[11] - fs_deq[12] + fs_deq[13] - fs_deq[14] + fs_deq[19] - fs_deq[20] + fs_deq[21] + fs_deq[22] + fs_deq[23] - fs_deq[24] - fs_deq[25] - fs_deq[26]) * drho;  
	FLOAT vs = ( fs_deq[3] - fs_deq[4] + fs_deq[7 ] + fs_deq[8 ] - fs_deq[9 ] - fs_deq[10] + fs_deq[15] - fs_deq[16] + fs_deq[17] - fs_deq[18] + fs_deq[19] + fs_deq[20] - fs_deq[21] + fs_deq[22] - fs_deq[23] + fs_deq[24] - fs_deq[25] - fs_deq[26]) * drho;
	FLOAT ws = ( fs_deq[5] - fs_deq[6] + fs_deq[11] + fs_deq[12] - fs_deq[13] - fs_deq[14] + fs_deq[15] + fs_deq[16] - fs_deq[17] - fs_deq[18] + fs_deq[19] + fs_deq[20] + fs_deq[21] - fs_deq[22] - fs_deq[23] - fs_deq[24] + fs_deq[25] - fs_deq[26]) * drho;

	

	FLOAT tke = 0.5 * (us*us + vs*vs + ws*ws);

	return tke;	
}

} // namespace //


// mathLib_LBM_inc.h //
