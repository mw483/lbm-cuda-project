#ifndef MATHLIB_H_
#define MATHLIB_H_


#include <cmath>
#include "definePrecision.h"
#include "defineCoefficient.h"


namespace	
mathLib {

// template *****
template <typename T>
T
average (T f, T g)
{
	return	(f + g) / 2;
}

template <typename T>
T
length_point2point (T *f, T *g)
{
	return	sqrt ( pow(f[0]-g[0], 2) 
				+  pow(f[1]-g[1], 2) 
				+  pow(f[2]-g[2], 2) );
}

template <typename T>
T
length_point2point (
	T f0, T f1, T f2, 
	T g0, T g1, T g2)
{
	return	sqrt ( pow(f0 - g0, 2) 
				+  pow(f1 - g1, 2) 
				+  pow(f2 - g2, 2) );
}


// interpolation //
// f //
inline __host__ __device__ FLOAT
average_1st (
	FLOAT	f0,
	FLOAT	f1);


inline __host__ __device__ FLOAT
fs_1st_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	vel);


inline __host__ __device__ FLOAT
fs_1st_central (
	FLOAT	f0,
	FLOAT	f1);


inline __host__ __device__ FLOAT
fs_2nd_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2);


inline __host__ __device__ FLOAT
fs_3rd_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	vel);


inline __host__ __device__ void
fs_3rd_upwind_sl (
	FLOAT	&fs,
	FLOAT	&fx,
	FLOAT	&fxx,
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	vel,
	FLOAT	dx);


inline __host__ __device__ FLOAT
fs_3rd_weno (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	vel);


inline __host__ __device__ FLOAT
fs_4th_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3);


inline __host__ __device__ FLOAT
fs_5th_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	f4,
	FLOAT	f5,
	FLOAT	vel);


inline __host__ __device__ FLOAT
fs_5th_weno (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	f4,
	FLOAT	f5,
	FLOAT	vel);


inline __host__ __device__ FLOAT
fs_6th_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	f4,
	FLOAT	f5);


// average //
inline __host__ __device__ FLOAT
average_3stencil (
	const FLOAT	*f,
	int		m);


// fx //
inline __host__ __device__ FLOAT
fx_1st_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	dx);


inline __host__ __device__ FLOAT
fx_1st_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	dx,
	FLOAT	vel
	);


// u fx //
inline __host__ __device__ FLOAT
ufx_1st_central (
	FLOAT	u0,
	FLOAT	u1,
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	dx);


inline __host__ __device__ FLOAT
fx_2nd_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	dx
	);


// taylor expansion //
inline __host__ __device__ FLOAT
taylor_expansion_dt (
	FLOAT	ft,
	FLOAT	ftt,
	FLOAT	dt);


inline __host__ __device__ FLOAT
taylor_expansion_dt (
	FLOAT	ft,
	FLOAT	ftt,
	FLOAT	ft3,
	FLOAT	dt);


inline __host__ __device__ FLOAT
taylor_expansion_dt (
	FLOAT	ft,
	FLOAT	ftt,
	FLOAT	ft3,
	FLOAT	ft4,
	FLOAT	dt);


} // namespace //


#include "mathLib_inc.h"


#endif
