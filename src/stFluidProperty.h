#ifndef STFLUIDPROPERTY_H_
#define STFLUIDPROPERTY_H_


#include "definePrecision.h"


struct
FluidProperty {
	// fluid or solid //
	char	*status;

	// viscosity //
	FLOAT	*vis_lbm;	// vis  : viscosity //
	FLOAT	vis0_lbm;

	// obstacle //
	int		*id_obs;
	FLOAT	*l_obs;		// fluid < 0, solid > 0 //
	FLOAT	*u_obs, *v_obs, *w_obs;

    // heat flux //
    FLOAT  *hflux_w; // heat flux on west-side surface. ( from west to east ) //   // MOD2018
    FLOAT  *hflux_e;
    FLOAT  *hflux_s;
    FLOAT  *hflux_n;
    FLOAT  *hflux_r; // from bottom to top //
};


#endif
