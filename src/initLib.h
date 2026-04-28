#ifndef INITLIB_H_
#define INITLIB_H_


#include "definePrecision.h"


namespace	
initLib {


// sin //
FLOAT
get_sin_1d (
	FLOAT	x);


FLOAT
get_sin_2d (
	FLOAT	x,
	FLOAT	y);


FLOAT
get_sin_3d (
	FLOAT	x,
	FLOAT	y,
	FLOAT	z);


// euler equation //
void
set_sod_1d (
	FLOAT	x,
	FLOAT	x_center,
	FLOAT	&rho,
	FLOAT	&p,
	FLOAT	&t);


void
set_sod (
	FLOAT	x,
	FLOAT	y,
	FLOAT	z,
	FLOAT	x_center,
	FLOAT	y_center,
	FLOAT	z_center,
	FLOAT	&rho,
	FLOAT	&p,
	FLOAT	&t);


} // namespace //


#endif
