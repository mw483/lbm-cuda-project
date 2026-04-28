#ifndef STSTRESS_H_
#define STSTRESS_H_


#include "definePrecision.h"
#include "Define_user.h"


// stress tensor
struct Stress {
	FLOAT	*vis_sgs; // sgs model
	FLOAT	*Fcs_sgs;


	FLOAT	*Div;
	FLOAT	*SS, *WW;

	// force //
	FLOAT	*force_x, *force_y, *force_z;

	// sgs at t-1 for LSM //
	// (YOKOUCHI 2020)
//	FLOAT	*vis_sgs_old;		// for temporal derivative
	FLOAT	*TKE_sgs;		// for SGS tke in LBM
	FLOAT 	*TKE_sgs_old;	 	// for SGS tke in LBM (temporal derivative)
	FLOAT 	*u_m, *v_m, *w_m; 	// for GS  tke (mean velocity)
};


#endif
