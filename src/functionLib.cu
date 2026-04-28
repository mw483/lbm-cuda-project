#include "functionLib.h"


void functionLib::
set_dim3 (dim3 *dim_cuda, int dim_x, int dim_y, int dim_z)
{
	dim_cuda->x = dim_x;
	dim_cuda->y = dim_y;
	dim_cuda->z = dim_z;
}


FLOAT functionLib::
get_elapsed_time (
	const struct timeval *begin,
	const struct timeval *end)
{
    return	  (end->tv_sec  - begin->tv_sec) * 1000
			+ (end->tv_usec - begin->tv_usec) / 1000.0;
}


// functionLib.cu
