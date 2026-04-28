#include "mathLib.h"


namespace	
mathLib {


// interpolation //
// f //
inline __host__ __device__ FLOAT
average_1st (
	FLOAT	f0,
	FLOAT	f1)
{
	return	(f0 + f1) * (FLOAT)0.5;
}


inline __host__ __device__ FLOAT
fs_1st_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	vel)
{
	return	vel > 0.0 ? f0 : f1;
}


inline __host__ __device__ FLOAT
fs_1st_central (
	FLOAT	f0,
	FLOAT	f1)
{
	return	(f0 + f1) * (FLOAT)0.5;
}


inline __host__ __device__ FLOAT
fs_2nd_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2)
{
	return	(f0 + 2.0*f1 + f2) * (FLOAT)0.25;
}


inline __host__ __device__ FLOAT
fs_3rd_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	vel)
{
	return	vel > 0.0 ?
		((FLOAT)2.0*f2 + (FLOAT)5.0*f1 - f0) / (FLOAT)6.0 :
		((FLOAT)2.0*f1 + (FLOAT)5.0*f2 - f3) / (FLOAT)6.0;
}


inline __host__ __device__ FLOAT
fs_3rd_weno (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	vel)
{
	const int	p = 1;
	const FLOAT	ep = coefficient::NON_ZERO_EP;

	const FLOAT	sgn_u = vel > 0.0 ? 1.0 : -1.0;

	const FLOAT	fi = vel > 0.0 ? f1 : f2; // 1st-upwind
//	const FLOAT	fi = vel > 0.0 ? (1.5*f1 - 0.5*f0) : (1.5*f2 - 0.5*f3); // 2nd-upwind

	const FLOAT	weno_fi[3] = {
					(FLOAT)1.5*f1 - (FLOAT)0.5*f0,
					(FLOAT)1.5*f2 - (FLOAT)0.5*f3,
					(FLOAT)0.5*f1 + (FLOAT)0.5*f2  } ;

	const FLOAT	weno_fx[3] = {
					f1 - f0,
					f3 - f2,
					f2 - f1  };

	// weno weight //
	const FLOAT	ideal_ww[3] = { 
					((FLOAT)1.0 + sgn_u) / (FLOAT)6.0,
					((FLOAT)1.0 - sgn_u) / (FLOAT)6.0,
					(FLOAT)2.0/(FLOAT)3.0 };

	// smooth indicator //
	FLOAT	ww[3], aa[3];
	for (int ii=0; ii<3; ii++) {
		const FLOAT	IS =  
			  pow(weno_fi[ii] - fi, 2)
			+ pow(weno_fx[ii]     , 2) * 4.0/3.0
			- sgn_u * (weno_fi[ii] - fi) * (weno_fx[ii]);

		aa[ii] = ideal_ww[ii] / (pow(IS, p) + ep);
	}
	const FLOAT	aa_all = aa[0] + aa[1] + aa[2] + ep;

	// weight
	for (int ii=0; ii<3; ii++) {
		ww[ii] = aa[ii] / aa_all;
	}

	return	(ww[0]*weno_fi[0] + ww[1]*weno_fi[1] + ww[2]*weno_fi[2]);
}


inline __host__ __device__ FLOAT
fs_4th_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3)
{
	return	-(f3 - 7.0*f2 - 7.0*f1 + f0) / (FLOAT)12.0;
}


inline __host__ __device__ FLOAT
fs_5th_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	f4,
	FLOAT	f5,
	FLOAT	vel)
{
	return	vel > 0.0 ?
			(2.0*f0 - 13.0*f1 + 47.0*f2 + 27.0*f3 - 3.0*f4) / (FLOAT)60.0 :
			(2.0*f5 - 13.0*f4 + 47.0*f3 + 27.0*f2 - 3.0*f1) / (FLOAT)60.0 ;
}


inline __host__ __device__ FLOAT
fs_5th_weno (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	f4,
	FLOAT	f5,
	FLOAT	vel)
{
	const int	p  = 2;
	const FLOAT	ep = coefficient::NON_ZERO_EP;

	const FLOAT	sgn_u = vel > (FLOAT)0.0 ? (FLOAT)1.0 : -(FLOAT)1.0;

	const FLOAT	fi = vel > (FLOAT)0.0 ? 
						(-f1 + (FLOAT)5.0*f2 + (FLOAT)2.0*f3)/(FLOAT)6.0 : 
						(-f4 + (FLOAT)5.0*f3 + (FLOAT)2.0*f2)/(FLOAT)6.0; // 3rd-upwind

	// weno //
	const FLOAT	weno_fi[4] = {
						( (FLOAT) 2.0*f0 - (FLOAT)7.0*f1 + (FLOAT)11.0*f2)/(FLOAT)6.0,
						(-            f1 + (FLOAT)5.0*f2 + (FLOAT) 2.0*f3)/(FLOAT)6.0,
						( (FLOAT) 2.0*f2 + (FLOAT)5.0*f3 -             f4)/(FLOAT)6.0,
						( (FLOAT)11.0*f3 - (FLOAT)7.0*f4 + (FLOAT) 2.0*f5)/(FLOAT)6.0 };

	const FLOAT	weno_fx[4] = {
						(  (FLOAT)    f0 - (FLOAT)3.0*f1 +  (FLOAT)2.0*f2),
						( -           f2 +            f3),
						( -           f2 +            f3),
						( -(FLOAT)2.0*f3 + (FLOAT)3.0*f4 -             f5) };

	const FLOAT	weno_fxx[4] = {
						( f0 - (FLOAT)2.0*f1 + f2),
						( f1 - (FLOAT)2.0*f2 + f3),
						( f2 - (FLOAT)2.0*f3 + f4),
						( f3 - (FLOAT)2.0*f4 + f5) };

	// weight //
	const FLOAT	ideal_ww[4] = {
						((FLOAT)1.0 +            sgn_u) /(FLOAT)20.0,
						((FLOAT)9.0 + (FLOAT)3.0*sgn_u) /(FLOAT)20.0,
						((FLOAT)9.0 - (FLOAT)3.0*sgn_u) /(FLOAT)20.0,
						((FLOAT)1.0 -            sgn_u) /(FLOAT)20.0 };

	FLOAT	ff     = 0.0;
	FLOAT	aa_all = ep;

	// smooth indicator //
	for (int ii=0; ii<4; ii++) {
		const FLOAT	IS =  
			  ( pow(weno_fi[ii]-fi, 2) + pow(weno_fx[ii], 2) + pow(weno_fxx[ii], 2) )
			- ( (FLOAT)2.0*(weno_fi[ii]-fi)*weno_fx[ii] + (FLOAT)2.0*weno_fx[ii]*weno_fxx[ii] ) /(FLOAT)2.0 * sgn_u
			+ ( pow(weno_fx[ii], 2) + weno_fx[ii]*weno_fxx[ii] + pow(weno_fxx[ii], 2) ) /(FLOAT)3.0
			- ( weno_fx[ii]*weno_fxx[ii] ) /(FLOAT)4.0 * sgn_u
			+ ( (FLOAT)0.25*pow(weno_fxx[ii], 2) ) /(FLOAT)5.0;

		const FLOAT	tmp = ideal_ww[ii] / (pow(IS, p) + ep);

		ff     += tmp*weno_fi[ii];
		aa_all += tmp;
	}

	return	ff / aa_all;
}


inline __host__ __device__ FLOAT
fs_6th_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	f4,
	FLOAT	f5)
{
	return	(f0 - 8.0*f1 + 37.0*f2 + 37.0*f3 - 8.0*f4 + f5) / (FLOAT)60.0;
}


// average //
inline __host__ __device__ FLOAT
average_3stencil (
	const FLOAT	*f,
	int		m)
{
	return	fs_2nd_central (f[-m], f[0], f[+m]);
}


// fx //
inline __host__ __device__ FLOAT
fx_1st_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	dx)
{
	return	(f1 - f0) / dx;
}


inline __host__ __device__ FLOAT
fx_1st_upwind (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	dx,
	FLOAT	vel
	)
{
	return	vel > (FLOAT)0.0 ?
			(f1 - f0) / dx :
			(f2 - f1) / dx;
}


inline __host__ __device__ FLOAT
fx_2nd_central (
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	dx
	)
{
	return	(f2 - f0) / ((FLOAT)2.0*dx);
}


// u fx //
inline __host__ __device__ FLOAT
ufx_1st_central (
	FLOAT	u0,
	FLOAT	u1,
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	dx)
{
	return	fs_1st_central (u0, u1) * fx_1st_central (f0, f1, dx);
}


inline __host__ __device__ FLOAT
usfx_1st_central (
	FLOAT	u0,
	FLOAT	u1,
	FLOAT	u2,
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	dx)
{
	return	fs_2nd_central (u0, u1, u2) * fx_1st_central (f0, f1, dx);
}


// semi lagrangian //
inline __host__ __device__ void
fs_3rd_upwind_poly (
	FLOAT	&fs,
	FLOAT	&fx,
	FLOAT	&fxx,
	FLOAT	f0,
	FLOAT	f1,
	FLOAT	f2,
	FLOAT	f3,
	FLOAT	vel,
	FLOAT	dx)
{
	if (vel > 0.0) {
		fs  = ((FLOAT)2.0*f2 + (FLOAT)5.0*f1 - f0) / (FLOAT)6.0;
		fx  = (f2 - f1) / dx;
		fxx = (f2 - (FLOAT)2.0*f1 + f0) / (dx*dx);
	}
	else {
		fs  = ((FLOAT)2.0*f1 + (FLOAT)5.0*f2 - f3) / (FLOAT)6.0;
		fx  = (f2 - f1) / dx;
		fxx = (f3 - (FLOAT)2.0*f2 + f1) / (dx*dx);
	}
}



// taylor expansion //
inline __host__ __device__ FLOAT
taylor_expansion_dt (
	FLOAT	ft,
	FLOAT	ftt,
	FLOAT	dt)
{
	return	ft + ftt*dt*(FLOAT)0.5;
}


inline __host__ __device__ FLOAT
taylor_expansion_dt (
	FLOAT	ft,
	FLOAT	ftt,
	FLOAT	ft3,
	FLOAT	dt)
{
	return	taylor_expansion_dt (ft, ftt, dt) + ft3*pow(dt, 2)/(FLOAT)6.0;
}


inline __host__ __device__ FLOAT
taylor_expansion_dt (
	FLOAT	ft,
	FLOAT	ftt,
	FLOAT	ft3,
	FLOAT	ft4,
	FLOAT	dt)
{
	return	taylor_expansion_dt (ft, ftt, ft3, dt) + ft4*pow(dt, 3)/(FLOAT)24.0;
}


} // namespace //


// mathLib_inc.h //
