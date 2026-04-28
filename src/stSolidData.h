#ifndef STSOLIDDATA_H_
#define STSOLIDDATA_H_


#include "definePrecision.h"


struct 
solidData {
	// id //
	int		stl_solid_id;	// STL データのID (自分自身のIDとは関係ない)

	// position //
	FLOAT	x_s, y_s, z_s;

	// angle //
	FLOAT	wx_s, wy_s, wz_s;

	// velocity //
	FLOAT	u_s, v_s, w_s;

	// rotation //


	// mass //
	FLOAT	mass_s;
};


#endif
