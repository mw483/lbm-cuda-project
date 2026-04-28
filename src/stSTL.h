#ifndef STSTL_H_
#define STSTL_H_


#define		_FAR_IN_	-10000.0
#define		_FAR_OUT_	 10000.0


#include "definePrecision.h"


// 固体壁面のSTL情報
struct
stl_solid_info {
	// solid infomation
	int		nx , ny , nz , n;
	int		nxs, nys, nzs;

	FLOAT	dx, dy, dz;
	FLOAT	Lx, Ly, Lz;

	FLOAT	xoff, yoff, zoff;
	FLOAT	ox  , oy  , oz  ;

	FLOAT	hx, hy, hz;
	FLOAT	lx, ly, lz;

	FLOAT	sinhx, coshx;
	FLOAT	sinhy, coshy;
	FLOAT	sinhz, coshz;
};


// 固体壁面のLevelSet
struct 
stl_solid_parts {
	char	*coi;
	float	**fs;
};


// 固体壁面のSTL情報
struct 
stl_solid {
	stl_solid_info	stl_solid_info;
	stl_solid_parts	*stl_solid_parts;
};


#endif
