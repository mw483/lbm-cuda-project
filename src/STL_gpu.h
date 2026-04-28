#ifndef STL_GPU_H_
#define STL_GPU_H_


#include <cuda.h>

#include "stSTL.h"
#include "definePrecision.h"


// function
inline __device__ __host__ FLOAT	
bi_interpolation (
	FLOAT gx, FLOAT gy, FLOAT gz, 
	float f0, float f1, float f2, float f3,
	float f4, float f5, float f6, float f7
	);


FLOAT	
ls_parts_v2
(
	FLOAT	x,		/* x-directional position 		*/
	FLOAT	y,		/* y-directional position 		*/
	FLOAT	z,		/* z-directional position 		*/
	      stl_solid_info	 sinfo,		/* level-set data for solid parts	*/
	const stl_solid_parts	*parts		/* level-set data for solid parts	*/
);


FLOAT	
ls_parts_v2
(
	FLOAT	x,		/* x-directional position 		*/
	FLOAT	y,		/* y-directional position 		*/
	FLOAT	z,		/* z-directional position 		*/
	stl_solid_info	 sinfo,		/* level-set data for solid parts	*/
	const char		*coi,
	float			**fs
);


inline __device__ __host__ FLOAT	
ls_parts_v2_gpu
(
	FLOAT	x,		/* x-directional position 		*/
	FLOAT	y,		/* y-directional position 		*/
	FLOAT	z,		/* z-directional position 		*/
	stl_solid_info	 sinfo,		/* level-set data for solid parts	*/
	const char		*coi,
	float			**fs
);


#include "STL_gpu_inc.h"


#endif
