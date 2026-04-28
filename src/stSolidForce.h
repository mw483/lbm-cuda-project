#ifndef STSOLIDFORCE_H_
#define STSOLIDFORCE_H_


#include "definePrecision.h"


struct
solidForce_STL {
	// force x,y,z //
	FLOAT	force_x;
	FLOAT	force_y;
	FLOAT	force_z;
};


struct
solidForce {
	int		*id_solidData;

	// force //
	FLOAT	*force_inner_x;	
	FLOAT	*force_inner_y;	
	FLOAT	*force_inner_z;	
};


#endif
