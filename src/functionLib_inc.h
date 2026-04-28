#include "functionLib.h"


namespace	
functionLib {

// limit //
inline __host__ __device__ FLOAT 
limit_0_to_1 (FLOAT	f)
{
	FLOAT	f_tmp = f;
	if      (f_tmp < 0.0)	{ f_tmp = 0.0; }
	else if (f_tmp > 1.0)	{ f_tmp = 1.0; }

	return	f_tmp;
}


inline __host__ __device__ FLOAT 
set_density_vof (
	FLOAT	f0,
	FLOAT	density_air,
	FLOAT	density_water
	)
{
	const FLOAT	fs = limit_0_to_1(f0);

	return	(1.0 - fs)*density_air 
		   +       fs *density_water;
}


// heaviside_function for free surface (0 - 1)
inline __host__ __device__ FLOAT
heaviside_function (
	FLOAT	l,
	FLOAT	alpha
	)
{
	// l > 0 : wall
	// l < 0 : fluid

	FLOAT	f;
	if      (l >  alpha)	{ f = 0.0; }
	else if (l > -alpha)	{ f = (FLOAT)0.5 * ( (FLOAT)1.0 - l/alpha - sin(M_PI*l/alpha)/M_PI ); }
	else					{ f = 1.0; }

	return	( (FLOAT)1.0 - f );
}


// heaviside_function for surface tension
inline __host__ __device__ FLOAT
heaviside_function_cos (
	FLOAT	l,
	FLOAT	alpha
	)
{
	// l > 0 : wall
	// l < 0 : fluid

	return	(fabs(l) > alpha) ?
			0.0 :
			(FLOAT)0.5 * ( (FLOAT)1.0 + cos(M_PI*l/alpha) ) / alpha;
}



inline __host__ __device__ FLOAT
heaviside_function_cos_fluid (
	FLOAT	l,
	FLOAT	alpha,
	int		sgn
	)
{
	// l > 0 : wall
	// l < 0 : fluid

	return	(fabs(l) > alpha || l*sgn < (FLOAT)0.0) ?
			(FLOAT)0.0 :
			(FLOAT)2.0 * (FLOAT)0.5 / alpha * ( (FLOAT)1.0 + cos(M_PI*l/alpha) );
}



inline __host__ __device__ FLOAT
interpolate_f_in_cell (
	const FLOAT	*f,
	const int	index8[],
	const FLOAT	weight3[]
	)
{
	const FLOAT	f_cell[8] = {
			f[index8[0]],
			f[index8[1]],
			f[index8[2]],
			f[index8[3]],
			f[index8[4]],
			f[index8[5]],
			f[index8[6]],
			f[index8[7]]  };

	const FLOAT mweight3[3] = {
				(FLOAT)1.0 - weight3[0],
				(FLOAT)1.0 - weight3[1],
				(FLOAT)1.0 - weight3[2]  };

	return
			( f_cell[0] * mweight3[0] * mweight3[1] * mweight3[2]
			+ f_cell[1] *  weight3[0] * mweight3[1] * mweight3[2]
			+ f_cell[2] * mweight3[0] *  weight3[1] * mweight3[2]
			+ f_cell[3] *  weight3[0] *  weight3[1] * mweight3[2]
			+ f_cell[4] * mweight3[0] * mweight3[1] *  weight3[2]
			+ f_cell[5] *  weight3[0] * mweight3[1] *  weight3[2]
			+ f_cell[6] * mweight3[0] *  weight3[1] *  weight3[2]
			+ f_cell[7] *  weight3[0] *  weight3[1] *  weight3[2] );
}


} // namespace //


// functionLib_inc.h //
