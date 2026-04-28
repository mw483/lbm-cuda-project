#ifndef SGS_MODEL_GPU_H_
#define SGS_MODEL_GPU_H_


#include <cuda.h>
#include "definePrecision.h"


__global__ void 
cuda_Fcs_CSM (
	const FLOAT	*u, 
	const FLOAT	*v, 
	const FLOAT	*w,  
	const FLOAT	*lv_obs,
	      FLOAT	*Fcs_sgs,
	      FLOAT	*Div,
	      FLOAT	*SS,
	      FLOAT	*WW,
	int nx, int ny, int nz,
	int halo
	);


#endif
