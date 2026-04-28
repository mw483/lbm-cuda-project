#include "matharrayLib.h"

#include "mathLib.h"


namespace	
matharrayLib {

using namespace	mathLib;

#include "mathLib.h"

// fx //
inline __host__ __device__ FLOAT
fx_1st_upwind (
	const FLOAT	f[],
	      FLOAT	dx,
		  FLOAT	vel,
		  int	m
	)
{
	return	mathLib::fx_1st_upwind (f[-m], f[0], f[m], dx, vel);
}


inline __host__ __device__ FLOAT
fx_2nd_central (
	const FLOAT	f[],
	      FLOAT	dx,
		  int	m
	)
{
	return	mathLib::fx_2nd_central (f[-m], f[0], f[m], dx);
}


// advection //
inline __host__ __device__ FLOAT
adv_1st_upwind (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[2] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx)	};

	const FLOAT	ft = fs_1st_upwind (cft[0], cft[1], vel);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_2nd_central (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[2] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx)	};

	const FLOAT	ft = fs_1st_central (cft[0], cft[1]);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_2nd_central_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[2] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx)	};

	const FLOAT	
		cftt = - usfx_1st_central (u[0], u[1], u[2], cft[0], cft[1], dx);


	const FLOAT	ft  = fs_1st_central (cft[0], cft[1]);
	const FLOAT	ftt = cftt;

	return	taylor_expansion_dt (ft, ftt, dt);
}


inline __host__ __device__ FLOAT
adv_3rd_upwind (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};

	const FLOAT	ft = fs_3rd_upwind (cft[0], cft[1], cft[2], cft[3], vel);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_3rd_upwind_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};

	const FLOAT	
		cftt[3] = {
			- usfx_1st_central (u[0], u[1], u[2], cft[0], cft[1], dx),
			- usfx_1st_central (u[1], u[2], u[3], cft[1], cft[2], dx),
			- usfx_1st_central (u[2], u[3], u[4], cft[2], cft[3], dx)	};

	const FLOAT
		cft3[2] = {
			- ufx_1st_central (u[1], u[2], cftt[0], cftt[1], dx),
			- ufx_1st_central (u[2], u[3], cftt[1], cftt[2], dx)	};


	const FLOAT	ft  = fs_3rd_upwind (cft[0], cft[1], cft[2], cft[3], vel);
	const FLOAT	ftt = cftt[1];
	const FLOAT	ft3 = fs_1st_upwind (cft3[0], cft3[1], vel);


	// df = (ft * dt + 1/2 ftt * dt^2  + 1/6 * ft3 * dt^3) / dt //
	return	taylor_expansion_dt (ft, ftt, ft3, dt);
}


inline __host__ __device__ FLOAT
adv_3rd_upwind_sl (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};

	FLOAT	ft, ftx, ftxx;
	fs_3rd_upwind_poly (
		ft, ftx, ftxx,
		cft[0], cft[1], cft[2], cft[3], vel, dx);

	const FLOAT	ftt = ftx  *     -vel;
	const FLOAT	ft3 = ftxx * pow(-vel, 2);


	// df = (ft * dt + 1/2 ftt * dt^2  + 1/6 * ft3 * dt^3) / dt //
//	return	taylor_expansion_dt (ft, ftt, dt);
	return	taylor_expansion_dt (ft, ftt, ft3, dt);
}


inline __host__ __device__ FLOAT
adv_3rd_weno (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};

	const FLOAT	ft = fs_3rd_weno (cft[0], cft[1], cft[2], cft[3], vel);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_3rd_weno_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};

	const FLOAT	
		cftt[3] = {
			- usfx_1st_central (u[0], u[1], u[2], cft[0], cft[1], dx),
			- usfx_1st_central (u[1], u[2], u[3], cft[1], cft[2], dx),
			- usfx_1st_central (u[2], u[3], u[4], cft[2], cft[3], dx)	};

	const FLOAT
		cft3[2] = {
			- ufx_1st_central (u[1], u[2], cftt[0], cftt[1], dx),
			- ufx_1st_central (u[2], u[3], cftt[1], cftt[2], dx)	};


	const FLOAT	ft  = fs_3rd_weno (cft[0], cft[1], cft[2], cft[3], vel);
	const FLOAT	ftt = cftt[1];
	const FLOAT	ft3 = fs_1st_upwind (cft3[0], cft3[1], vel);


	// df = (ft * dt + 1/2 ftt * dt^2  + 1/6 * ft3 * dt^3) / dt //
//	return	taylor_expansion_dt (ft, ftt, dt);
	return	taylor_expansion_dt (ft, ftt, ft3, dt);
}


inline __host__ __device__ FLOAT
adv_4th_central (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};


	const FLOAT	ft = fs_4th_central (cft[0], cft[1], cft[2], cft[3]);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_4th_central_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[4] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx)	};

	const FLOAT	
		cftt[3] = {
			- usfx_1st_central (u[0], u[1], u[2], cft[0], cft[1], dx),
			- usfx_1st_central (u[1], u[2], u[3], cft[1], cft[2], dx),
			- usfx_1st_central (u[2], u[3], u[4], cft[2], cft[3], dx)	};

	const FLOAT
		cft3[2] = {
			- ufx_1st_central (u[1], u[2], cftt[0], cftt[1], dx),
			- ufx_1st_central (u[2], u[3], cftt[1], cftt[2], dx)	};

	const FLOAT
		cft4 = - usfx_1st_central (u[1], u[2], u[3], cft3[0], cft3[1], dx);


	const FLOAT	ft = fs_4th_central (cft[0], cft[1], cft[2], cft[3]);
	const FLOAT	ftt = cftt[1];
	const FLOAT	ft3 = fs_1st_central (cft3[0], cft3[1]);
	const FLOAT	ft4 = cft4;


	// df = (ft * dt + 1/2 ftt * dt^2  + 1/6 * ft3 * dt^3) / dt //
	return	taylor_expansion_dt (ft, ftt, ft3, ft4, dt);
}


inline __host__ __device__ FLOAT
adv_5th_upwind (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[6] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx),
			- ufx_1st_central (u[4], u[5], f[4], f[5], dx),
			- ufx_1st_central (u[5], u[6], f[5], f[6], dx)	};

	const FLOAT	ft = fs_5th_upwind (cft[0], cft[1], cft[2], cft[3], cft[4], cft[5], vel);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_5th_upwind_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[6] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx),
			- ufx_1st_central (u[4], u[5], f[4], f[5], dx),
			- ufx_1st_central (u[5], u[6], f[5], f[6], dx)	};

	const FLOAT	
		cftt[5] = {
			- usfx_1st_central (u[0], u[1], u[2], cft[0], cft[1], dx),
			- usfx_1st_central (u[1], u[2], u[3], cft[1], cft[2], dx),
			- usfx_1st_central (u[2], u[3], u[4], cft[2], cft[3], dx),
			- usfx_1st_central (u[3], u[4], u[5], cft[3], cft[4], dx),
			- usfx_1st_central (u[4], u[5], u[6], cft[4], cft[5], dx)	};

	const FLOAT
		cft3[4] = {
			- ufx_1st_central (u[1], u[2], cftt[0], cftt[1], dx),
			- ufx_1st_central (u[2], u[3], cftt[1], cftt[2], dx),
			- ufx_1st_central (u[3], u[4], cftt[2], cftt[3], dx),
			- ufx_1st_central (u[4], u[5], cftt[3], cftt[4], dx)	};


//	const FLOAT	
//		cft4[3] = {
//			- usfx_1st_central (u[1], u[2], u[3], cft3[0], cft3[1], dx),
//			- usfx_1st_central (u[2], u[3], u[4], cft3[1], cft3[2], dx),
//			- usfx_1st_central (u[3], u[4], u[5], cft3[2], cft3[3], dx)	};


	const FLOAT	ft  = fs_5th_upwind (cft[0], cft[1], cft[2], cft[3], cft[4], cft[5], vel);
	const FLOAT	ftt = cftt[2];
	const FLOAT	ft3 = fs_3rd_upwind (cft3[0], cft3[1], cft3[2], cft3[3], vel);
//	const FLOAT	ft4 = cft4[1];

	// df = (ft * dt + 1/2 ftt * dt^2  + 1/6 * ft3 * dt^3) / dt //
	return	taylor_expansion_dt (ft, ftt, ft3, dt);
//	return	taylor_expansion_dt (ft, ftt, ft3, ft4, dt);
}


inline __host__ __device__ FLOAT
adv_5th_weno (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[6] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx),
			- ufx_1st_central (u[4], u[5], f[4], f[5], dx),
			- ufx_1st_central (u[5], u[6], f[5], f[6], dx)	};

	const FLOAT	ft = fs_5th_weno (cft[0], cft[1], cft[2], cft[3], cft[4], cft[5], vel);

	return	ft;
}


inline __host__ __device__ FLOAT
adv_5th_weno_tx (
	FLOAT	dx,
	FLOAT	dt,
	FLOAT	vel,
	const FLOAT	u[],
	const FLOAT	f[])
{
	const FLOAT	
		cft[6] = {
			- ufx_1st_central (u[0], u[1], f[0], f[1], dx),
			- ufx_1st_central (u[1], u[2], f[1], f[2], dx),
			- ufx_1st_central (u[2], u[3], f[2], f[3], dx),
			- ufx_1st_central (u[3], u[4], f[3], f[4], dx),
			- ufx_1st_central (u[4], u[5], f[4], f[5], dx),
			- ufx_1st_central (u[5], u[6], f[5], f[6], dx)	};

	const FLOAT	
		cftt[5] = {
			- usfx_1st_central (u[0], u[1], u[2], cft[0], cft[1], dx),
			- usfx_1st_central (u[1], u[2], u[3], cft[1], cft[2], dx),
			- usfx_1st_central (u[2], u[3], u[4], cft[2], cft[3], dx),
			- usfx_1st_central (u[3], u[4], u[5], cft[3], cft[4], dx),
			- usfx_1st_central (u[4], u[5], u[6], cft[4], cft[5], dx)	};

	const FLOAT
		cft3[4] = {
			- ufx_1st_central (u[1], u[2], cftt[0], cftt[1], dx),
			- ufx_1st_central (u[2], u[3], cftt[1], cftt[2], dx),
			- ufx_1st_central (u[3], u[4], cftt[2], cftt[3], dx),
			- ufx_1st_central (u[4], u[5], cftt[3], cftt[4], dx)	};


//	const FLOAT	
//		cft4[3] = {
//			- usfx_1st_central (u[1], u[2], u[3], cft3[0], cft3[1], dx),
//			- usfx_1st_central (u[2], u[3], u[4], cft3[1], cft3[2], dx),
//			- usfx_1st_central (u[3], u[4], u[5], cft3[2], cft3[3], dx)	};


	const FLOAT	ft  = fs_5th_weno (cft[0], cft[1], cft[2], cft[3], cft[4], cft[5], vel);
	const FLOAT	ftt = cftt[2];
//	const FLOAT	ft3 = fs_3rd_weno   (cft3[0], cft3[1], cft3[2], cft3[3], vel);
	const FLOAT	ft3 = fs_3rd_upwind (cft3[0], cft3[1], cft3[2], cft3[3], vel);
//	const FLOAT	ft4 = cft4[1];

	// df = (ft * dt + 1/2 ftt * dt^2  + 1/6 * ft3 * dt^3) / dt //
	return	taylor_expansion_dt (ft, ftt, ft3, dt);
//	return	taylor_expansion_dt (ft, ftt, ft3, ft4, dt);
}


// levelset
inline __host__ __device__ void
normal_vector (
	      FLOAT	nvec3[], 
	const FLOAT	*l,			// index (i + nx*j + nx*ny*k)
		  int	nx,
		  int	ny,
		  int	nz
	)
{
	// l > 0 : 物体内部
	// nvec  : 界面からの法線ベクトル
	const int	stride_x = 1;
	const int	stride_y = nx;
	const int	stride_z = nx*ny;


	const FLOAT	lx = fx_2nd_central (l, (FLOAT)1.0, stride_x);
	const FLOAT	ly = fx_2nd_central (l, (FLOAT)1.0, stride_y);
	const FLOAT	lz = fx_2nd_central (l, (FLOAT)1.0, stride_z);
	const FLOAT	ld = sqrt(lx*lx + ly*ly + lz*lz);
	const FLOAT	ep = coefficient::NON_ZERO_EP;

	// 法線ベクトル
	nvec3[0] = lx / (ld + ep);
	nvec3[1] = ly / (ld + ep);
	nvec3[2] = lz / (ld + ep);
}


inline __host__ __device__ void
normal_vector_upwind (
	      FLOAT	nvec3[], 
	const FLOAT	*l,			// index (i + nx*j + nx*ny*k)
		  int	nx,
		  int	ny,
		  int	nz
	)
{
	// l > 0 : 物体内部
	// nvec  : 界面からの法線ベクトル
	const int	stride_x = 1;
	const int	stride_y = nx;
	const int	stride_z = nx*ny;


	FLOAT	lx = fx_2nd_central (l, (FLOAT)1.0, stride_x);
	FLOAT	ly = fx_2nd_central (l, (FLOAT)1.0, stride_y);
	FLOAT	lz = fx_2nd_central (l, (FLOAT)1.0, stride_z);

	const FLOAT	sgn = l[0] > (FLOAT)0.0 ? 1.0 : -1.0;
	lx = fx_1st_upwind (l, (FLOAT)1.0, sgn*lx, stride_x);
	ly = fx_1st_upwind (l, (FLOAT)1.0, sgn*ly, stride_y);
	lz = fx_1st_upwind (l, (FLOAT)1.0, sgn*lz, stride_z);


	const FLOAT	ld = sqrt(lx*lx + ly*ly + lz*lz);
	const FLOAT	ep = coefficient::NON_ZERO_EP;

	// normal vector
	nvec3[0] = lx / (ld + ep);
	nvec3[1] = ly / (ld + ep);
	nvec3[2] = lz / (ld + ep);
}


inline __host__ __device__ void
normal_vector_from_surface (
	      FLOAT	nvec3[], 
	const FLOAT	*l,			// index (i + nx*j + nx*ny*k)
		  int	nx,
		  int	ny,
		  int	nz
	)
{
	normal_vector_upwind (
		nvec3,
		l,
		nx, ny, nz
		);


	const FLOAT	sgn = l[0] > (FLOAT)0.0 ? 1.0 : -1.0;

	// normal vector
	nvec3[0] *= sgn;
	nvec3[1] *= sgn;
	nvec3[2] *= sgn;
}


} // namespace //


// matharrayLib_inc.h //
