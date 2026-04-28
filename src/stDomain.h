#ifndef STDOMAIN_H_
#define STDOMAIN_H_


#include "definePrecision.h"


struct
Domain {
	// lattice boltzmann method //
	FLOAT	c_ref;		// sound speed //
	FLOAT	cfl_ref;
	FLOAT	vel_cfl_ref;	// MOD2019

	// number of grid points (global) //
	int		nxg, nyg, nzg;

	// number of grid points (local) //
	int		nx, ny, nz;
	int		nn;
	int		n0;

	// number of step //
	int		step;

	// halo //
	int		halo;

	// time //
	FLOAT	time_end;
	FLOAT	time;
	FLOAT	dt;

	// dx, dy, dz //
	FLOAT	dx;

	// local //
	FLOAT	x_min, y_min, z_min;
	FLOAT	x_max, y_max, z_max;

	// global //
	FLOAT	xg_length, yg_length, zg_length;
	FLOAT	xg_min, yg_min, zg_min;
	FLOAT	xg_max, yg_max, zg_max;

	// region //
	FLOAT	*x, *y, *z;
};


#endif
