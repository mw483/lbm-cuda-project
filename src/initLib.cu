#include "initLib.h"

#include "mathLib.h"
#include "defineCoefficient.h"


FLOAT initLib::
get_sin_1d (
	FLOAT	x)
{
	// domain //
	// x, y, z : 0-1 //
	const FLOAT	wx = 2.0*M_PI;

	return	sin(wx*x);
}


FLOAT initLib::
get_sin_2d (
	FLOAT	x,
	FLOAT	y)
{
	// domain //
	// x, y, z : 0-1 //
	const FLOAT	wx = 2.0*M_PI;
	const FLOAT	wy = 2.0*M_PI;

	return	sin(wx*x) * sin(wy*y);
}


FLOAT initLib::
get_sin_3d (
	FLOAT	x,
	FLOAT	y,
	FLOAT	z)
{
	// domain //
	// x, y, z : 0-1 //
	const FLOAT	wx = 2.0*M_PI;
	const FLOAT	wy = 2.0*M_PI;
	const FLOAT	wz = 2.0*M_PI;

	return	sin(wx*x) * sin(wy*y) * sin(wz*z);
}


void initLib::
set_sod_1d (
	FLOAT	x,
	FLOAT	x_center,
	FLOAT	&rho,
	FLOAT	&p,
	FLOAT	&t)
{
	// domain //
	// x, y, z : 0-1 //

	const FLOAT	gamma = coefficient::GAMMA;
	const FLOAT	rr = fabs(x - x_center);

	if (rr < 0.2) {
		rho = 1.0;
		p   = 1.0;
	}
	else {
		rho = 0.125;
		p   = 0.1;
	}
	t = p / (rho * (gamma - 1.0));

}



void initLib::
set_sod (
	FLOAT	x,
	FLOAT	y,
	FLOAT	z,
	FLOAT	x_center,
	FLOAT	y_center,
	FLOAT	z_center,
	FLOAT	&rho,
	FLOAT	&p,
	FLOAT	&t)
{
	// domain //
	// x, y, z : 0-1 //

	const FLOAT	gamma = coefficient::GAMMA;
	const FLOAT	rr = mathLib::length_point2point (x, y, z, x_center, y_center, z_center);

	if (rr < 0.2) {
		rho = 1.0;
		p   = 1.0;
	}
	else {
		rho = 0.125;
		p   = 0.1;
	}
	t = p / (rho * (gamma - 1.0));

}


// initLib.cu //
