#ifndef DEFINECOEFFICIENT_H_
#define DEFINECOEFFICIENT_H_


#define COEF_THERMAL 2.207e-5 // k / (rho * Cp) //
#define BASE_TEMPERATURE 300 // k

// MOD 2018 //
#define PR_SGS 0.1 // SGS Prandtl number (normally 0.41)
//#define PR_SGS 0.4 // SGS Prandtl number (normally 0.41)	// << recommended
//#define PR_SGS 0.6 // SGS Prandtl number (normally 0.41)
//#define PR_SGS 1.0 // SGS Prandtl number (normally 0.41)


namespace
coefficient {
	#include "definePrecision.h"

	const FLOAT	cal_length = 3.25;

//	const FLOAT	x_offset   = -0.33;
//	const FLOAT	y_offset   = -3.25;
//	const FLOAT	z_offset   = -3.20;

	const FLOAT	z_offset   = -0.33;
	const FLOAT	y_offset   = -3.25;
	const FLOAT	x_offset   = -3.20;

	const FLOAT	DENSITY_AIR = 1.293;
	const FLOAT KVIS_AIR    = 1.512*1.0e-5;	// kinetic viscosity //
	const FLOAT VIS_AIR     = KVIS_AIR * DENSITY_AIR;	// viscosity //

	const FLOAT	NON_ZERO_EP = 1.0e-16;

    const FLOAT GRAVITY = 9.8;

	// immersed boundary width //
	const FLOAT	IBM_INTERFACE = 0.45; 

	// solidForce width //
	const FLOAT	SOLID_FORCE_WIDTH = 2.0;

};


#endif
