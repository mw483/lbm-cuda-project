#include "mathFuncLBM.h"


namespace
mathFuncLBM {


// bounce-back model for solid boundary //
inline __device__
void
solid_boundary_bounce_back (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		if (l_obs[idg_lbm[ii]] > 0.0) {
			fs[ii] = f[direction_lbm[ii]*nn + idg_lbm[0]];
		}
	}
}


// Bouzidi's model for solid boundary //
inline __device__
void
solid_boundary_BOUZIDI (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		const FLOAT	lv0   = l_obs[idg_lbm[0] ];
		const FLOAT	lv_up = l_obs[idg_lbm[ii]];

//		if (lv_up > (FLOAT)0.0 && lv0 < (FLOAT)0.0) {	// fluid < 0, solid > 0 // // diverge
		if (lv_up > (FLOAT)0.0) {	// fluid < 0, solid > 0 //
			// moving boundary force //


			// boundary condition //
			const FLOAT	ep      = coefficient::NON_ZERO_EP;
			const FLOAT	delta   = fabs( lv0 ) / ( fabs(lv0) + fabs(lv_up) + ep );	// level-set function (boundary) //
			const FLOAT	delta_2 = delta * (FLOAT)2.0;

			const FLOAT	fp_f  = f[direction_lbm[ii]*nn + idg_lbm[0]                ];
			const FLOAT	fp_ff = f[direction_lbm[ii]*nn + idg_lbm[direction_lbm[ii]]];
			const FLOAT	fm_f  = f[              ii *nn + idg_lbm[0]                ];


			// fs_tmp //
			FLOAT	fs_tmp;
			if (delta < (FLOAT)0.5) {	// 0.0 < delta < 0.5 //
				fs_tmp =    delta_2               * fp_f
						 + ((FLOAT)1.0 - delta_2) * fp_ff;
			}
			else {						// 0.5 < delta < 1.0 //
				fs_tmp = (                          fp_f
						 + (delta_2 - (FLOAT)1.0) * fm_f
						 )
						 /  delta_2;
			}

			fs[ii] = fs_tmp;
		}
	}
}


// Bouzidi's model for moving boundary //
inline __device__
void
moving_boundary_BOUZIDI (
	      FLOAT		fs[],
	const FLOAT*	f,
	const FLOAT*	l_obs,
	const FLOAT		force_obs[],
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		num_direction_vel,
	const int		nn
	)
{
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		const FLOAT	lv0   = l_obs[idg_lbm[0] ];
		const FLOAT	lv_up = l_obs[idg_lbm[ii]];

//		if (lv_up > (FLOAT)0.0 && lv0 < (FLOAT)0.0) {	// fluid < 0, solid > 0 // // diverge
		if (lv_up >= (FLOAT)0.0) {	// fluid < 0, solid > 0 //
			// moving boundary force //


			// boundary condition //
			const FLOAT	ep      = coefficient::NON_ZERO_EP;
			const FLOAT	delta   = fabs( lv0 ) / ( fabs(lv0) + fabs(lv_up) + ep );	// level-set function (boundary) //
			const FLOAT	delta_2 = delta * (FLOAT)2.0;

			const FLOAT	fp_f  = f[direction_lbm[ii]*nn + idg_lbm[0]                ];
			const FLOAT	fp_ff = f[direction_lbm[ii]*nn + idg_lbm[direction_lbm[ii]]];
			const FLOAT	fm_f  = f[              ii *nn + idg_lbm[0]                ];


			// fs_tmp //
			FLOAT	fs_tmp;
			if (delta < (FLOAT)0.5) {	// 0.0 < delta < 0.5 //
				fs_tmp =    delta_2               * fp_f
						 + ((FLOAT)1.0 - delta_2) * fp_ff
						 + force_obs[ii];	// force obs //
			}
			else {						// 0.5 < delta < 1.0 //
				fs_tmp = (                          fp_f
						 + (delta_2 - (FLOAT)1.0) * fm_f
						 + force_obs[ii]	// force obs //
						 )
						 /  delta_2;
			}

			fs[ii] = fs_tmp;
		}
	}
}


// momentum exchange on boundary //
inline __device__
void
momentum_exchange_on_boundary_D3Q19 (
	      FLOAT		force_obs[],
	const FLOAT*	l_obs,
	const FLOAT*	u_obs,
	const FLOAT*	v_obs,
	const FLOAT*	w_obs,
	const int		idg_lbm[],
	const int		direction_lbm[],
	const int		nn
	)
{
	// D3Q19 //
	// center:   e0 ( 0, 0, 0), e1 ( 1, 0, 0), e2 (-1, 0, 0), e3 ( 0, 1, 0),  e4( 0,-1, 0),  e5( 0, 0, 1),  e6( 0, 0,-1)
	// x-y face: e7 ( 1, 1, 0), e8 (-1, 1, 0), e9 ( 1,-1, 0), e10(-1,-1, 0)
	// x-z face: e11( 1, 0, 1), e12(-1, 0, 1), e13( 1, 0,-1), e14(-1, 0,-1)
	// y-z face: e15( 0, 1, 1), e16( 0,-1, 1), e17( 0, 1,-1), e18( 0,-1,-1)
	const int	num_direction_vel = 19;

	FLOAT	u[num_direction_vel];
	FLOAT	v[num_direction_vel];
	FLOAT	w[num_direction_vel];

	u[0] = 0.0;
	v[0] = 0.0;
	w[0] = 0.0;
#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		u[ii] = u_obs[idg_lbm[ii]];
		v[ii] = v_obs[idg_lbm[ii]];
		w[ii] = w_obs[idg_lbm[ii]];
	}


	// inner product //
//	const FLOAT	C1D3   = 1.0/ 3.0;
	const FLOAT	C1D18  = 1.0/18.0;
	const FLOAT	C1D36  = 1.0/36.0;

	force_obs[0 ] = 0.0;	// C1D3 //

	force_obs[1 ] = ( u[1 ]) * (FLOAT)3.0*C1D18;
	force_obs[2 ] = (-u[2 ]) * (FLOAT)3.0*C1D18;
	force_obs[3 ] = ( v[3 ]) * (FLOAT)3.0*C1D18;
	force_obs[4 ] = (-v[4 ]) * (FLOAT)3.0*C1D18;
	force_obs[5 ] = ( w[5 ]) * (FLOAT)3.0*C1D18;
	force_obs[6 ] = (-w[6 ]) * (FLOAT)3.0*C1D18;

	force_obs[7 ] = ( u[7 ] + v[7 ]) * (FLOAT)3.0*C1D36;
	force_obs[8 ] = (-u[8 ] + v[8 ]) * (FLOAT)3.0*C1D36;
	force_obs[9 ] = ( u[9 ] - v[9 ]) * (FLOAT)3.0*C1D36;
	force_obs[10] = (-u[10] - v[10]) * (FLOAT)3.0*C1D36;

	force_obs[11] = ( u[11] + w[11]) * (FLOAT)3.0*C1D36;
	force_obs[12] = (-u[12] + w[12]) * (FLOAT)3.0*C1D36;
	force_obs[13] = ( u[13] - w[13]) * (FLOAT)3.0*C1D36;
	force_obs[14] = (-u[14] - w[14]) * (FLOAT)3.0*C1D36;

	force_obs[15] = ( v[15] + w[15]) * (FLOAT)3.0*C1D36;
	force_obs[16] = (-v[16] + w[16]) * (FLOAT)3.0*C1D36;
	force_obs[17] = ( v[17] - w[17]) * (FLOAT)3.0*C1D36;
	force_obs[18] = (-v[18] - w[18]) * (FLOAT)3.0*C1D36;

#pragma unroll
	for (int ii=1; ii<num_direction_vel; ii++) {
		const FLOAT	lv0   = l_obs[idg_lbm[0] ];
		const FLOAT	lv_up = l_obs[idg_lbm[ii]];

		if (lv0 > (FLOAT)0.0 || lv_up < (FLOAT)0.0) {	// fluid < 0, solid > 0 //
			force_obs[ii] = 0.0;
		}
	}
}


} // namespace //
