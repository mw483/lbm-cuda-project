#ifndef MATHARRAYLIB_H_
#define MATHARRAYLIB_H_


#include <cmath>
#include "definePrecision.h"
#include "defineCoefficient.h"


namespace	
matharrayLib {


// fx //
inline __host__ __device__ FLOAT
fx_1st_upwind (
	const FLOAT	f[],
	      FLOAT	dx,
		  FLOAT	vel,
		  int	m
	);


inline __host__ __device__ FLOAT
fx_2nd_central (
	const FLOAT	f[],
	      FLOAT	dx,
		  int	m
	);


// advection //
inline __host__ __device__ FLOAT
adv_1st_upwind (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_2nd_central (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_2nd_central_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_3rd_upwind (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_3rd_upwind_tx (
	FLOAT	dx,	
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_3rd_upwind_sl (
	FLOAT	dx,	
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_3rd_weno (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_3rd_weno_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_4th_central (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_4th_central_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_5th_upwind (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_5th_upwind_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_5th_weno (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


inline __host__ __device__ FLOAT
adv_5th_weno_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[]);


} // namespace //


#include "matharrayLib_inc.h"


#endif
