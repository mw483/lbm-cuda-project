#ifndef SGS_LBM_GPU_H_
#define SGS_LBM_GPU_H_


#include <cuda.h>
#include "definePrecision.h"


// SGS Model
__global__ void 
CUDA_Fcs_CSM_Model(
	const FLOAT *u, const FLOAT *v, const FLOAT *w,  
	const char *status,
	FLOAT *Fcs_sgs,
	FLOAT *Div,
	FLOAT *SS, FLOAT *WW,
	int nx,     int ny,     int nz,
	int halo_x, int halo_y, int halo_z);


#endif
