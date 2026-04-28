#include "classSolidData.h"

#include "functionLib.h"
#include "allocateLib.h"
#include "macroCUDA.h"

#include "defineBoundaryFlag.h"
#include "defineCoefficient.h"


// public //
// initialize //
void	classSolidData::
init_classSolidData (
	const MPIinfo	&mpiinfo, 
	const Domain	&domain
	)
{
	// mpiinfo //
	mpiinfo_ = mpiinfo;


	// domain //
	domain_ = domain;


	// initialize //
	set_solidData ();
}


void	classSolidData::
move_SolidData (
	FLOAT	force_x,
	FLOAT	force_y,
	FLOAT	force_z 
	)
{
	const FLOAT	mass = soliddata_.mass_s;

	const FLOAT	acc_x = force_x / mass;
	const FLOAT	acc_y = force_y / mass;
	const FLOAT	acc_z = force_z / mass;

	const FLOAT	dt = domain_.dt;

	// velocity //
	const FLOAT	u0 = soliddata_.u_s;
	const FLOAT	v0 = soliddata_.v_s;
	const FLOAT	w0 = soliddata_.w_s;

	const FLOAT	un = u0 + acc_x * dt;
	const FLOAT	vn = v0 + acc_y * dt;
	const FLOAT	wn = w0 + acc_z * dt;


	// position //
	const FLOAT	x0 = soliddata_.x_s;
	const FLOAT	y0 = soliddata_.y_s;
	const FLOAT	z0 = soliddata_.z_s;

	const FLOAT	xn = x0 + u0 * dt;
	const FLOAT	yn = y0 + v0 * dt;
	const FLOAT	zn = z0 + w0 * dt;


	// update //
	soliddata_.x_s = xn;
	soliddata_.y_s = yn;
	soliddata_.z_s = zn;

	soliddata_.u_s = un;
	soliddata_.v_s = vn;
	soliddata_.w_s = wn;
}


// private //
// initialize //
void	classSolidData::
set_solidData ()
{
	// id //
	soliddata_.stl_solid_id = 0;	// STL データのID (自分自身のIDとは関係ない)

	// position //
	soliddata_.x_s = 0.0;
	soliddata_.y_s = 0.0;
	soliddata_.z_s = 0.0;

	// angle //
	soliddata_.wx_s = 0.0;
	soliddata_.wy_s = 0.0;
	soliddata_.wz_s = 0.0;

	// velocity //
	soliddata_.u_s = 0.0;
	soliddata_.v_s = 0.0;
	soliddata_.w_s = 0.0;

	// rotation //


	// mass //
	soliddata_.mass_s = 1.0;
}
