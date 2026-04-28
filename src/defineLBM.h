#ifndef DEFINELBM_H_
#define DEFINELBM_H_


// LBM direction
//#define D3Q19_MODEL_
#define D3Q27_MODEL_


#define BOUNDARY_BOUNCE_BACK_
//#define BOUNDARY_BOUZIDI_	// linear //


#ifdef D3Q19_MODEL_
const int	NUM_DIRECTION_VEL = 19;
#endif
#ifdef D3Q27_MODEL_
const int	NUM_DIRECTION_VEL = 27;
#endif


enum
LBM_VELOCITY_MODEL {
	D3Q19_VELOCITY,
	D3Q27_VELOCITY
};


enum
LBM_STREAM_BOUNDARY_MODEL {
	BOUNCE_BACK,
	BOUNDARY_2ND_ORDER,
};


namespace
definelbm {
//	#include "definePrecision.h"

}


#endif
