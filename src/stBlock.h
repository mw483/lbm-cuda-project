#ifndef STBLOCK_H_
#define STBLOCK_H_


#include "definePrecision.h"


struct
stBlock {
	int		nx;
	int		ny;
	int		nz;

	int		halo;

	FLOAT	dx;
};


#endif
