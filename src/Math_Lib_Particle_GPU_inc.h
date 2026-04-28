// (YOKOUCHI 2020)
#include <cmath>
#include <curand_kernel.h>
#include "stDomain.h"
#include "stStress.h"

#include "Math_Lib_Particle_GPU.h"

#include "defineBoundaryFlag.h"


// math function *****
// device
inline __device__ int Check_Particle_Status(
	const char *frg,
	FLOAT x, FLOAT y, FLOAT z, 
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT dx, FLOAT dy, FLOAT dz, 
	int nx, int ny, int nz,
	int halo)
{
	// index
	const FLOAT	xc_min = xs_min + (FLOAT)0.5*dx;
	const FLOAT	yc_min = ys_min + (FLOAT)0.5*dy;
	const FLOAT	zc_min = zs_min + (FLOAT)0.5*dz;

	const int	id_xm = Index_left_collocate_period(x, xc_min, dx, nx, halo);
	const int	id_ym = Index_left_collocate_period(y, yc_min, dy, ny, halo);
	const int	id_zm = Index_left_collocate_period(z, zc_min, dz, nz, halo);

	const int	id_xp = (id_xm + 1)%nx;
	const int	id_yp = (id_ym + 1)%ny;
	const int	id_zp = (id_zm + 1)%nz;

	// value
	int		pfrg = PARTICLE_CAL;

	if (	frg[id_xm + nx*id_ym + nx*ny*id_zm] != STATUS_FLUID
		&&	frg[id_xp + nx*id_ym + nx*ny*id_zm] != STATUS_FLUID
		&&	frg[id_xm + nx*id_yp + nx*ny*id_zm] != STATUS_FLUID
		&&	frg[id_xp + nx*id_yp + nx*ny*id_zm] != STATUS_FLUID
		&&	frg[id_xm + nx*id_ym + nx*ny*id_zp] != STATUS_FLUID
		&&	frg[id_xp + nx*id_ym + nx*ny*id_zp] != STATUS_FLUID
		&&	frg[id_xm + nx*id_yp + nx*ny*id_zp] != STATUS_FLUID
		&&	frg[id_xp + nx*id_yp + nx*ny*id_zp] != STATUS_FLUID )
	{
		pfrg = PARTICLE_NA;
	}

	return	pfrg;
}


inline __device__ __host__ FLOAT 
Interpolate_Particle_Velocity (
	const FLOAT	*f,
	FLOAT	x,
	FLOAT	y,
	FLOAT	z, 
	FLOAT	xs_min,
	FLOAT	ys_min,
	FLOAT	zs_min,
	FLOAT	dx, 
	FLOAT	dy, 
	FLOAT	dz, 
	int nx, int ny, int nz,
	int halo)
{
	// index
	const FLOAT	xc_min = xs_min + (FLOAT)0.5*dx;
	const FLOAT	yc_min = ys_min + (FLOAT)0.5*dy;
	const FLOAT	zc_min = zs_min + (FLOAT)0.5*dz;

	const int	id_xm = Index_left_collocate_period (x, xc_min, dx, nx, halo);
	const int	id_ym = Index_left_collocate_period (y, yc_min, dy, ny, halo);
	const int	id_zm = Index_left_collocate_period (z, zc_min, dz, nz, halo);

	const int	id_xp = (id_xm + 1)%nx;
	const int	id_yp = (id_ym + 1)%ny;
	const int	id_zp = (id_zm + 1)%nz;

	// value
	const FLOAT	fc[8] = {
					f[id_xm + nx*id_ym + nx*ny*id_zm],
					f[id_xp + nx*id_ym + nx*ny*id_zm],
					f[id_xm + nx*id_yp + nx*ny*id_zm],
					f[id_xp + nx*id_yp + nx*ny*id_zm],
					f[id_xm + nx*id_ym + nx*ny*id_zp],
					f[id_xp + nx*id_ym + nx*ny*id_zp],
					f[id_xm + nx*id_yp + nx*ny*id_zp],
					f[id_xp + nx*id_yp + nx*ny*id_zp] };

	// coordinate
	const FLOAT	xm = xc_min + (FLOAT)(Index_left_collocate(x, xc_min, dx, nx, halo) - halo) * dx; 
	const FLOAT	ym = yc_min + (FLOAT)(Index_left_collocate(y, yc_min, dy, ny, halo) - halo) * dy; 
	const FLOAT	zm = zc_min + (FLOAT)(Index_left_collocate(z, zc_min, dz, nz, halo) - halo) * dz; 

	// weight
	const FLOAT	 wx = fabs(x - xm) / dx;
	const FLOAT	 wy = fabs(y - ym) / dy;
	const FLOAT	 wz = fabs(z - zm) / dz;

	const FLOAT	mwx = fabs( (FLOAT)1.0 - wx );
	const FLOAT	mwy = fabs( (FLOAT)1.0 - wy );
	const FLOAT	mwz = fabs( (FLOAT)1.0 - wz );


	// interpolate
	const FLOAT	fint =
			  fc[0] * mwx * mwy * mwz
			+ fc[1] *  wx * mwy * mwz
			+ fc[2] * mwx *  wy * mwz
			+ fc[3] *  wx *  wy * mwz
			+ fc[4] * mwx * mwy *  wz
			+ fc[5] *  wx * mwy *  wz
			+ fc[6] * mwx *  wy *  wz
			+ fc[7] *  wx *  wy *  wz;


	return	fint;
}


inline __host__ FLOAT _Interpolate_Particle_Velocity_cpu(
	const FLOAT *f,
	FLOAT x, FLOAT y, FLOAT z, 
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT dx, FLOAT dy, FLOAT dz, 
	int nx, int ny, int nz,
	int halo)
{
	// index
	const FLOAT	xc_min = xs_min + (FLOAT)0.5*dx;
	const FLOAT	yc_min = ys_min + (FLOAT)0.5*dy;
	const FLOAT	zc_min = zs_min + (FLOAT)0.5*dz;

	const int	id_xm = Index_left_collocate_period(x, xc_min, dx, nx, halo);
	const int	id_ym = Index_left_collocate_period(y, yc_min, dy, ny, halo);
	const int	id_zm = Index_left_collocate_period(z, zc_min, dz, nz, halo);

	const int	id_xp = (id_xm + 1)%nx;
	const int	id_yp = (id_ym + 1)%ny;
	const int	id_zp = (id_zm + 1)%nz;

	// value
	const FLOAT	fc[8] = {
					f[id_xm + nx*id_ym + nx*ny*id_zm],
					f[id_xp + nx*id_ym + nx*ny*id_zm],
					f[id_xm + nx*id_yp + nx*ny*id_zm],
					f[id_xp + nx*id_yp + nx*ny*id_zm],
					f[id_xm + nx*id_ym + nx*ny*id_zp],
					f[id_xp + nx*id_ym + nx*ny*id_zp],
					f[id_xm + nx*id_yp + nx*ny*id_zp],
					f[id_xp + nx*id_yp + nx*ny*id_zp] };


	// coordinate
	const FLOAT	xm = xc_min + (FLOAT)(Index_left_collocate(x, xc_min, dx, nx, halo) - halo) * dx; 
	const FLOAT	ym = yc_min + (FLOAT)(Index_left_collocate(y, yc_min, dy, ny, halo) - halo) * dy; 
	const FLOAT	zm = zc_min + (FLOAT)(Index_left_collocate(z, zc_min, dz, nz, halo) - halo) * dz; 

	// weight
	FLOAT	 wx = (x - xm) / dx;
	FLOAT	 wy = (y - ym) / dy;
	FLOAT	 wz = (z - zm) / dz;

	if (   wx < 0.0 || wx > 1.0
		|| wy < 0.0 || wy > 1.0
		|| wz < 0.0 || wz > 1.0 ) {
		std::cout << "wx, wy, wz = " << wx << ", " << wy << ", " << wz << std::endl; 
		std::cout << "error : Interpolate_Particle_Velocity_cpu\n";
	}


	const FLOAT	mwx = fabs( (FLOAT)1.0 - wx );
	const FLOAT	mwy = fabs( (FLOAT)1.0 - wy );
	const FLOAT	mwz = fabs( (FLOAT)1.0 - wz );


	// interpolate
	const FLOAT	fint =
			  fc[0] * mwx * mwy * mwz
			+ fc[1] *  wx * mwy * mwz
			+ fc[2] * mwx *  wy * mwz
			+ fc[3] *  wx *  wy * mwz
			+ fc[4] * mwx * mwy *  wz
			+ fc[5] *  wx * mwy *  wz
			+ fc[6] * mwx *  wy *  wz
			+ fc[7] *  wx *  wy *  wz;


	return	fint;
}


inline __device__ __host__ int 
Index_left_collocate_period (
	FLOAT	x,
   	FLOAT	xc_min,
   	FLOAT	dx,
   	int		nx,
   	int		offset_x
	)
{
	// MOD2021 YOKOUCHI
	return	( (int)( (int)(x - xc_min)/dx + offset_x + nx ) )%nx; // center
	//return (x - xc_min > 0) ? ( (int)( (int)(x - xc_min)/(int)dx + offset_x + nx ) )%nx :
	//			   ( (int)( (int)floor((x - xc_min)/dx) + offset_x + nx ) )%nx ;
}


inline __device__ __host__ int 
Index_left_collocate (
	FLOAT	x,
	FLOAT	xc_min,
	FLOAT	dx,
	int		nx, 
	int		offset_x
	)
{
	// MOD2021 YOKOUCHI
	return	  (int)( (int)(x - xc_min)/dx + offset_x + nx ) - nx;
	//return (x - xc_min > 0) ? (int)( (int)(x - xc_min)/(int)dx + offset_x + nx ) - nx :
	//			   (int)( (int)floor((x - xc_min)/dx) + offset_x + nx ) - nx ;
}


// 1次元補間
template<typename T>
inline __device__ __host__ T
interpolate_f_1d (
	const T	fc[],
	T	wx)
{
	const T	mwx = (T)1.0 - wx;

	return	  fc[0] * mwx
			+ fc[1] *  wx;
}


// 3次元補間
template<typename T>
inline __device__ __host__ T
interpolate_f_3d (
	const T	fc[],
	T	wx,
	T	wy,
	T	wz)
{
	const T	mwx = (T)1.0 - wx;
	const T	mwy = (T)1.0 - wy;
	const T	mwz = (T)1.0 - wz;

	return    fc[0] * mwx * mwy * mwz
			+ fc[1] *  wx * mwy * mwz
			+ fc[2] * mwx *  wy * mwz
			+ fc[3] *  wx *  wy * mwz
			+ fc[4] * mwx * mwy *  wz
			+ fc[5] *  wx * mwy *  wz
			+ fc[6] * mwx *  wy *  wz
			+ fc[7] *  wx *  wy *  wz;
}


// 粒子の格子点上での左側のindex
template<typename T>
inline __device__ __host__ int 
index_particle_left_collocate (
	T	x_particle,
	T	xmin_center,
	T	dx,
	int	nx,
	int	offset_x)
{
	return	( (int)( (x_particle - xmin_center)/dx + offset_x + nx ) - nx );
}

// (YOKOUCHI 2020)
// Random SGS Velocity
inline __device__ __host__ FLOAT
Random_SGS_Velocity_Energy (
	FLOAT		vel_sgs,
	const FLOAT *tke_sgs, const FLOAT *tke_sgs_old,
	const FLOAT k_GS,
	FLOAT x, FLOAT y, FLOAT z,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT rand,
	FLOAT dx, FLOAT dy, FLOAT dz,
	int nx, int ny, int nz,
	int		halo,
	FLOAT		dt,
	int		dim
	)
{
	const FLOAT	C_0 	= 4.0;
	
	const FLOAT	dx_diff = 0.2; //[m], +-dx_diff
	
	const FLOAT	nondim_l  = dx / dx;
	const FLOAT	vel_s	  = vel_sgs;
	const FLOAT	nondim_dt = dt / dt;
	const FLOAT	nondim_dx = dx_diff / dx;

	// tke (x, y, z, t)
	const FLOAT	tke_s 		= Interpolate_Particle_Velocity(tke_sgs, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

	// tke (x, y, z, t-1)
	const FLOAT	tke_s_old	= Interpolate_Particle_Velocity(tke_sgs_old, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

	// spatial gradient of turbulent viscosity at t
	      FLOAT	tke_nabla;	
	if (dim == 0) {
		FLOAT 	xm = x - dx_diff;
		FLOAT 	xp = x + dx_diff;
		FLOAT	tke_sm	= Interpolate_Particle_Velocity(tke_sgs, xm, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
		FLOAT	tke_sp	= Interpolate_Particle_Velocity(tke_sgs, xp, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

			tke_nabla = (tke_sp - tke_sm)/ (2.0 * nondim_dx);
	} else if (dim == 1) {
		FLOAT 	ym = y - dx_diff;
		FLOAT 	yp = y + dx_diff;
		FLOAT	tke_sm	= Interpolate_Particle_Velocity(tke_sgs, x, ym, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
		FLOAT	tke_sp	= Interpolate_Particle_Velocity(tke_sgs, x, yp, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

			tke_nabla = (tke_sp - tke_sm)/ (2.0 * nondim_dx);	
	} else if (dim == 2) {
		FLOAT 	zm = z - dx_diff;
		FLOAT 	zp = z + dx_diff;
		FLOAT	tke_sm	= Interpolate_Particle_Velocity(tke_sgs, x, y, zm, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
		FLOAT	tke_sp	= Interpolate_Particle_Velocity(tke_sgs, x, y, zp, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

			tke_nabla = (tke_sp - tke_sm)/ (2.0 * nondim_dx);	
	}

	if (tke_s == 0.0) { return 0.0;}
	
	const FLOAT	f_s	= tke_s / (k_GS + tke_s);
	const FLOAT	diss	= local_dissipation_rate(nondim_l, tke_s);

	const FLOAT	tke_dt 	= (tke_s - tke_s_old) / nondim_dt;

	// calculation //
	// first term //
	const FLOAT	tmp_first	= - (3.0*C_0/4.0) * f_s * diss * vel_s / tke_s * nondim_dt;

	// second term //
	const FLOAT	tmp_second	= 0.5 * (1.0/tke_s * tke_dt * vel_s + 2.0/3.0 * tke_nabla) * nondim_dt;
//	const FLOAT	tmp_second 	= 0;
	
	// third term //
//	const FLOAT	tmp_third	= pow(f_s * C_0 * diss, 0.5) * rand;
	const FLOAT	tmp_third	= pow(f_s * C_0 * diss, 0.5) * rand;
//	const FLOAT	tmp_third	= 0.0;

	FLOAT	vel_d 	= tmp_first + tmp_second + tmp_third;

	FLOAT	SD	= pow(tke_s, 0.5);

	FLOAT	vel_new = vel_d + vel_s;

	if (vel_new < -2*SD || 2*SD < vel_new) {vel_d = 0;}

	return vel_d;
//	return tke_s;
}

// the local SGS TKE
inline __device__ __host__ FLOAT
local_SGS_TKE (
	const int	id,
	FLOAT		dx,
	const FLOAT	*vis_sgs,
	FLOAT		c_ref
	)
{
	const FLOAT C_k 	= 0.1;
	
	const FLOAT len_mix 	= dx;

	FLOAT vis_tur		= vis_sgs[id] * c_ref * dx;

	return 	pow((vis_tur / (C_k * len_mix)), 2.0);

}
 
// the local dissipation rate
inline __device__ __host__ FLOAT
local_dissipation_rate (
	FLOAT		dx,
	FLOAT		tke_sgs
	)
{
	const FLOAT 	C_eps 		= 0.93;
	
	const FLOAT 	len_mix		= dx;

	return	C_eps * pow(tke_sgs, 3.0/2.0) / len_mix;
}

// the mean contribution of the SGS TKE to the total TKE
inline __device__ __host__ FLOAT
contribution_SGS_TKE (
	const	FLOAT	vis,
	const	FLOAT	GS_tke,
	const	FLOAT	dx
	)
{
	const	FLOAT	C_k = 0.1;

	FLOAT	sgs_tke	 = pow(vis/C_k/dx, 2);

	return sgs_tke / (GS_tke + sgs_tke) ;
}

inline __device__ __host__ FLOAT
Random_SGS_Velocity_Vis_sgs (
	FLOAT		vel_sgs,
	const FLOAT *vis_sgs, const FLOAT *vis_sgs_old,
//	const FLOAT *total_TKE,
	const FLOAT k_GS,
	FLOAT x, FLOAT y, FLOAT z,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT rand,
	FLOAT dx, FLOAT dy, FLOAT dz,
	int nx, int ny, int nz,
	int		halo,
	FLOAT		dt,
	int		dim	
	)
{
	const FLOAT	C_0 	= 4.0;
	const FLOAT	C_k	= 0.1;
	const FLOAT	C_eps	= 0.93;

	const FLOAT	dx_diff = 0.2; //[m]
	
	const FLOAT	nondim_l  = dx / dx;
	const FLOAT	vel_s	  = vel_sgs;
	const FLOAT	nondim_dt = dt / dt;
	const FLOAT	nondim_dx = dx_diff / dx;

	// turbulent viscosity on particle at t
	const FLOAT	vis_s 		= Interpolate_Particle_Velocity(vis_sgs, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

	// turbulent viscosity on particle at t-1
	const FLOAT	vis_s_old	= Interpolate_Particle_Velocity(vis_sgs_old, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

	// spatial gradient of turbulent viscosity at t
	      FLOAT	vis_nabla;	
	if (dim == 0) {
		FLOAT 	xm = x - dx_diff;
		FLOAT 	xp = x + dx_diff;
		FLOAT	vis_sm	= Interpolate_Particle_Velocity(vis_sgs, xm, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
		FLOAT	vis_sp	= Interpolate_Particle_Velocity(vis_sgs, xp, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

			vis_nabla = (pow(vis_sp, 2.0) - pow(vis_sm, 2.0))/ (2.0 * nondim_dx);
	} else if (dim == 1) {
		FLOAT 	ym = y - dx_diff;
		FLOAT 	yp = y + dx_diff;
		FLOAT	vis_sm	= Interpolate_Particle_Velocity(vis_sgs, x, ym, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
		FLOAT	vis_sp	= Interpolate_Particle_Velocity(vis_sgs, x, yp, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

			vis_nabla = (pow(vis_sp, 2.0) - pow(vis_sm, 2.0))/ (2.0 * nondim_dx);	
	} else if (dim == 2) {
		FLOAT 	zm = z - dx_diff;
		FLOAT 	zp = z + dx_diff;
		FLOAT	vis_sm	= Interpolate_Particle_Velocity(vis_sgs, x, y, zm, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
		FLOAT	vis_sp	= Interpolate_Particle_Velocity(vis_sgs, x, y, zp, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

			vis_nabla = (pow(vis_sp, 2.0) - pow(vis_sm, 2.0))/ (2.0 * nondim_dx);	
	}
/*	// Index
	
	const FLOAT	xc_min = xs_min + (FLOAT)0.5*dx;
	const FLOAT	yc_min = ys_min + (FLOAT)0.5*dy;
	const FLOAT	zc_min = zs_min + (FLOAT)0.5*dz;

	int		id_xm = Index_left_collocate_period(x, xc_min, dx, nx, halo);
	int		id_ym = Index_left_collocate_period(y, yc_min, dy, ny, halo);
	int		id_zm = Index_left_collocate_period(z, zc_min, dz, nz, halo);

	int	id_xp = (id_xm + 1)%nx;
	int	id_yp = (id_ym + 1)%ny;
	int	id_zp = (id_zm + 1)%nz;
	
	// coordinate
	const FLOAT	xm = xc_min + (FLOAT)(Index_left_collocate(x, xc_min, dx, nx, halo) - halo) * dx; 
	const FLOAT	ym = yc_min + (FLOAT)(Index_left_collocate(y, yc_min, dy, ny, halo) - halo) * dy; 
	const FLOAT	zm = zc_min + (FLOAT)(Index_left_collocate(z, zc_min, dz, nz, halo) - halo) * dz; 

	// weight
	const FLOAT	 wx = fabs(x - xm) / dx;
	const FLOAT	 wy = fabs(y - ym) / dy;
	const FLOAT	 wz = fabs(z - zm) / dz;

	const FLOAT	mwx = fabs( (FLOAT)1.0 - wx );
	const FLOAT	mwy = fabs( (FLOAT)1.0 - wy );
	const FLOAT	mwz = fabs( (FLOAT)1.0 - wz );

	// value
	      FLOAT	fc[8] = {
					vis_sgs[id_xm + nx*id_ym + nx*ny*id_zm],
					vis_sgs[id_xp + nx*id_ym + nx*ny*id_zm],
					vis_sgs[id_xm + nx*id_yp + nx*ny*id_zm],
					vis_sgs[id_xp + nx*id_yp + nx*ny*id_zm],
					vis_sgs[id_xm + nx*id_ym + nx*ny*id_zp],
					vis_sgs[id_xp + nx*id_ym + nx*ny*id_zp],
					vis_sgs[id_xm + nx*id_yp + nx*ny*id_zp],
					vis_sgs[id_xp + nx*id_yp + nx*ny*id_zp] };

	// interpolate
	const FLOAT	vis_s =		  fc[0] * mwx * mwy * mwz
					+ fc[1] *  wx * mwy * mwz
					+ fc[2] * mwx *  wy * mwz
					+ fc[3] *  wx *  wy * mwz
					+ fc[4] * mwx * mwy *  wz
					+ fc[5] *  wx * mwy *  wz
					+ fc[6] * mwx *  wy *  wz
					+ fc[7] *  wx *  wy *  wz;

	// value
			fc[0] = 	vis_sgs_old[id_xm + nx*id_ym + nx*ny*id_zm];
			fc[1] = 	vis_sgs_old[id_xp + nx*id_ym + nx*ny*id_zm];
			fc[2] = 	vis_sgs_old[id_xm + nx*id_yp + nx*ny*id_zm];
			fc[3] = 	vis_sgs_old[id_xp + nx*id_yp + nx*ny*id_zm];
			fc[4] = 	vis_sgs_old[id_xm + nx*id_ym + nx*ny*id_zp];
			fc[5] = 	vis_sgs_old[id_xp + nx*id_ym + nx*ny*id_zp];
			fc[6] =		vis_sgs_old[id_xm + nx*id_yp + nx*ny*id_zp];
			fc[7] =		vis_sgs_old[id_xp + nx*id_yp + nx*ny*id_zp];

	// interpolate
	const FLOAT	vis_s_old =	  fc[0] * mwx * mwy * mwz
					+ fc[1] *  wx * mwy * mwz
					+ fc[2] * mwx *  wy * mwz
					+ fc[3] *  wx *  wy * mwz
					+ fc[4] * mwx * mwy *  wz
					+ fc[5] *  wx * mwy *  wz
					+ fc[6] * mwx *  wy *  wz
					+ fc[7] *  wx *  wy *  wz;

	// value
			fc[0] = 	total_TKE[id_xm + nx*id_ym + nx*ny*id_zm];
			fc[1] = 	total_TKE[id_xp + nx*id_ym + nx*ny*id_zm];
			fc[2] = 	total_TKE[id_xm + nx*id_yp + nx*ny*id_zm];
			fc[3] = 	total_TKE[id_xp + nx*id_yp + nx*ny*id_zm];
			fc[4] = 	total_TKE[id_xm + nx*id_ym + nx*ny*id_zp];
			fc[5] = 	total_TKE[id_xp + nx*id_ym + nx*ny*id_zp];
			fc[6] =		total_TKE[id_xm + nx*id_yp + nx*ny*id_zp];
			fc[7] =		total_TKE[id_xp + nx*id_yp + nx*ny*id_zp];

	// interpolate
	const FLOAT	total_tke =	  fc[0] * mwx * mwy * mwz
					+ fc[1] *  wx * mwy * mwz
					+ fc[2] * mwx *  wy * mwz
					+ fc[3] *  wx *  wy * mwz
					+ fc[4] * mwx * mwy *  wz
					+ fc[5] *  wx * mwy *  wz
					+ fc[6] * mwx *  wy *  wz
					+ fc[7] *  wx *  wy *  wz;


	// Index
	if (dim == 0) {
		id_xm = id_xm + 1;
	} else if (dim == 1) {
		id_ym = id_ym + 1;
	} else if (dim == 2) {
		id_zm = id_zm + 1;
	}

			id_xp = (id_xm + 1)%nx;
			id_yp = (id_ym + 1)%ny;
			id_zp = (id_zm + 1)%nz;

	// value
			fc[0] = 	vis_sgs[id_xm + nx*id_ym + nx*ny*id_zm];
			fc[1] = 	vis_sgs[id_xp + nx*id_ym + nx*ny*id_zm];
			fc[2] = 	vis_sgs[id_xm + nx*id_yp + nx*ny*id_zm];
			fc[3] = 	vis_sgs[id_xp + nx*id_yp + nx*ny*id_zm];
			fc[4] = 	vis_sgs[id_xm + nx*id_ym + nx*ny*id_zp];
			fc[5] = 	vis_sgs[id_xp + nx*id_ym + nx*ny*id_zp];
			fc[6] =		vis_sgs[id_xm + nx*id_yp + nx*ny*id_zp];
			fc[7] =		vis_sgs[id_xp + nx*id_yp + nx*ny*id_zp];
		
	// interpolate
	const FLOAT	vis_s_pp =	  fc[0] * mwx * mwy * mwz
					+ fc[1] *  wx * mwy * mwz
					+ fc[2] * mwx *  wy * mwz
					+ fc[3] *  wx *  wy * mwz
					+ fc[4] * mwx * mwy *  wz
					+ fc[5] *  wx * mwy *  wz
					+ fc[6] * mwx *  wy *  wz
					+ fc[7] *  wx *  wy *  wz;
*/
	
	if (vis_s == 0.0) { return 0.0;}
	const FLOAT	f_s		= contribution_SGS_TKE(vis_s, k_GS, nondim_l);
	const FLOAT	vis_dt		= (vis_s - vis_s_old) / nondim_dt;
//	const FLOAT	vis_nabla	= (pow(vis_s_pp, 2.0)  - pow(vis_s, 2.0)) / nondim_dx;

	const FLOAT	tmp_first	= - (3.0*C_0*C_eps/4.0/C_k) * f_s * (vel_s * vis_s) * pow(nondim_l, -2.0) * nondim_dt;
//	const FLOAT	tmp_first	= 0;
	const FLOAT	tmp_second	= 0.5 * (2.0 / vis_s * vis_dt * vel_s + 2.0/3.0 * pow(C_k*nondim_l, -2.0) * vis_nabla) * nondim_dt;
//	const FLOAT	tmp_second	= 0.5 * pow(vis_s, -2.0) * vis_dt * vel_s * nondim_dt;
//	const FLOAT	tmp_second	= 0.5 * 2.0/3.0 * pow(C_k, -2.0) * vis_nabla * nondim_dt;	
//	const FLOAT	tmp_second 	= 0;
	const FLOAT	tmp_third	= pow(f_s*C_0*C_eps, 0.5) * pow(vis_s/C_k, 1.5) * pow(nondim_l, -2) * rand;
//	const FLOAT	tmp_third	= 0;

	FLOAT	vel_d 	= tmp_first + tmp_second + tmp_third;

	FLOAT	SD	= vis_s / C_k / nondim_l;

	FLOAT	vel_new = vel_d + vel_s;

	if (vel_new < -2*SD || 2*SD < vel_new) {vel_d = 0.0;}

	return vel_d;
//	return vis_s_old;
}

inline __device__ void particle_reflection (
	const FLOAT	*l_obs,
	const FLOAT x,		const FLOAT y, 		const FLOAT z,
	      FLOAT &x_new, 	      FLOAT &y_new,	      FLOAT &z_new,
	      FLOAT &u_s,	      FLOAT &v_s,	      FLOAT &w_s,
	      FLOAT xs_min,	      FLOAT ys_min,	      FLOAT zs_min,
	      FLOAT dx,		      FLOAT dy,		      FLOAT dz,
	      int   nx,		      int   ny,		      int   nz,
	      int   halo	
)
{
	// New reflection MOD2021
	// copy
	FLOAT pos[3];
	pos[0]  	= x;
	pos[1] 	 	= y;
	pos[2] 	 	= z;
	
	FLOAT pos_new[3];
	pos_new[0] 	= x_new;
	pos_new[1] 	= y_new;
	pos_new[2] 	= z_new;

	FLOAT vel_s[3];
	vel_s[0]	= u_s;
	vel_s[1] 	= v_s;
	vel_s[2]  	= w_s;

	FLOAT res[3];
	res[0]	 	= dx;
	res[1]	 	= dy;
	res[2] 	 	= dz;

	int ngrid[3];
	ngrid[0]	= nx;
	ngrid[1]   	= ny;
	ngrid[2]   	= nz;

	// Index
	FLOAT c_min[3];
	c_min[0] = xs_min+0.08;//+(FLOAT)0.5*dx; //MOD2021 
	c_min[1] = ys_min+0.08;//+(FLOAT)0.5*dy;
	c_min[2] = zs_min+0.08;//+(FLOAT)0.5*dz;

	// if z < 0 Index_left_collocate() is not correct
	if (pos_new[2] < 0.0) {
		pos_new[2] = -pos_new[2];
		vel_s[2]   = -vel_s[2];
	}

	int id_new[3];
	int id_obs_new;
	for (int i=0; i<3; i++) {
		id_new[i] = Index_left_collocate_period(pos_new[i], c_min[i], res[i], ngrid[i], halo);
	}
	id_new[2]++;
	id_obs_new = id_new[0] + nx*id_new[1] + nx*ny*id_new[2];

	if (l_obs[id_obs_new] < 0.0) {
		x_new = pos_new[0];
		y_new = pos_new[1];
		z_new = pos_new[2];
		u_s   = vel_s[0];
		v_s   = vel_s[1];
		w_s   = vel_s[2];
		return;
	}

	int id_old[3];
	for (int i=0; i<3; i++) { 
		id_old[i] = Index_left_collocate_period(pos[i], c_min[i], res[i], ngrid[i], halo);
	}	
	id_old[2]++;
	const int id_obs_old = id_old[0] + nx*id_old[1] + nx*ny*id_old[2];

	if (id_obs_old == id_obs_new && l_obs[id_obs_new]>0.0) {
		 	x_new = 100;
			y_new = 50;
			z_new = 2;
			return;
	}

	for (int i=0; i<3; i++) {
		if (id_new[i] != id_old[i] && l_obs[id_obs_new] > 0.0) {	
			if (abs(id_new[i] - id_old[i]) >= 2) {
			x_new = 100;
			y_new = 100;
			z_new = 2;
			return;
			}

			double max_pos, min_pos, diff, pos_b;
			max_pos = (pos_new[i] > pos[i])? pos_new[i]:pos[i];
			min_pos = (pos_new[i] < pos[i])? pos_new[i]:pos[i];

			pos_b   = c_min[i] + (FLOAT)(Index_left_collocate(max_pos, c_min[i], res[i], ngrid[i], halo) - halo) * res[i];
			diff   	= pos_new[i] - pos_b;

		      	pos_new[i]     = pos_b - diff; // new position
		      	vel_s[i]       = -vel_s[i];    // new veloctiy

			id_new[i] = Index_left_collocate_period(pos_new[i], c_min[i], res[i], ngrid[i], halo);
			if (i == 2) id_new[i]++;
			id_obs_new = id_new[0] + nx*id_new[1] + nx*ny*id_new[2];
			
			if (l_obs[id_obs_new] < 0.0) {
			x_new = pos_new[0];
			y_new = pos_new[1];
			z_new = pos_new[2];
			u_s   = vel_s[0];
			v_s   = vel_s[1];
			w_s   = vel_s[2];
			return;
			} else if (diff == 0) {
			x_new = pos[0];
			y_new = pos[1];
			z_new = pos[2];
			u_s   = vel_s[0];
			v_s   = vel_s[1];
			w_s   = vel_s[2];
			return;
			}	
		}		
	}	

	int id_new_c[3];
	for (int i=0; i<3; i++) {
		id_new_c[i] = Index_left_collocate_period(pos_new[i], c_min[i], res[i], ngrid[i], halo);
	}
	id_new_c[2]++;
	const int id_obs_new_c  = id_new_c[0] + nx*id_new_c[1] + nx*ny*id_new_c[2];

	if (l_obs[id_obs_new_c] < 0.0) {
	x_new = pos_new[0];
	y_new = pos_new[1];
	z_new = pos_new[2];
	u_s   = vel_s[0];
	v_s   = vel_s[1];
	w_s   = vel_s[2];
	} else {
	x_new = pos[0];
	y_new = pos[1];
	z_new = pos[2];
	u_s   = vel_s[0];
	v_s   = vel_s[1];
	w_s   = vel_s[2];	
	}	
	return;
}

// Math_Lib_Particle_GPU_inc.h
