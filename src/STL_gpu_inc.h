#include "STL_gpu.h"



inline __device__ __host__ FLOAT	
bi_interpolation (
	FLOAT gx, FLOAT gy, FLOAT gz, 
	float f0, float f1, float f2, float f3,
	float f4, float f5, float f6, float f7
	)
{
	const FLOAT		px = (FLOAT)1.0 - gx,	
					py = (FLOAT)1.0 - gy,	
					pz = (FLOAT)1.0 - gz;

	return	
		(px*py*f0 + gx*py*f1 + px*gy*f2 + gx*gy*f3)*pz
	+   (px*py*f4 + gx*py*f5 + px*gy*f6 + gx*gy*f7)*gz;
}


inline __device__ __host__ FLOAT	
ls_parts_v2_gpu
(
	FLOAT	x,		/* x-directional position 		*/
	FLOAT	y,		/* y-directional position 		*/
	FLOAT	z,		/* z-directional position 		*/
	stl_solid_info	sinfo,		/* level-set data for solid parts	*/
	const char		*coi,
	float			**fs
)
// -----------------------------------------------------------------------
{
	const FLOAT	dx = sinfo.dx,	
				dy = sinfo.dy,
				dz = sinfo.dz;

	const FLOAT	xoff = sinfo.xoff,
	  			yoff = sinfo.yoff,
				zoff = sinfo.zoff;

	const FLOAT	ox = sinfo.ox,
	  			oy = sinfo.oy,
				oz = sinfo.oz;

	const FLOAT	sinhx = sinfo.sinhx,		coshx = sinfo.coshx,
				sinhy = sinfo.sinhy,		coshy = sinfo.coshy,
				sinhz = sinfo.sinhz,		coshz = sinfo.coshz;

	const int	nx  = sinfo.nx,	
		  		ny  = sinfo.ny;
	const int	nxs = sinfo.nxs,
				nys = sinfo.nys,
				nzs = sinfo.nys;

	const int	ns = (nxs + 1)*(nys + 1);

	FLOAT	xn, yn, zn;
	yn = oy + coshx*(y - oy) - sinhx*(z - oz);
	zn = oz + sinhx*(y - oy) + coshx*(z - oz);
	y = yn;		z = zn;

	xn = ox + coshy*(x - ox) - sinhy*(z - oz);
	zn = oz + sinhy*(x - ox) + coshy*(z - oz);
	x = xn;		z = zn;

	xn = ox + coshz*(x - ox) - sinhz*(y - oy);
	yn = oy + sinhz*(x - ox) + coshz*(y - oy);
	x = xn;		y = yn;


	x -= sinfo.lx;	
	y -= sinfo.ly;
	z -= sinfo.lz;

	if ( x < xoff
	 ||  y < yoff
	 ||  z < zoff )	{ return _FAR_OUT_; }

	if ( xoff + sinfo.Lx - (FLOAT)1.0*dx < x
	 ||  yoff + sinfo.Ly - (FLOAT)1.0*dy < y
	 ||  zoff + sinfo.Lz - (FLOAT)1.0*dz < z )	{ return _FAR_OUT_; }


	FLOAT	xj = (x - xoff)/dx;
	FLOAT	yj = (y - yoff)/dy;
	FLOAT	zj = (z - zoff)/dz;

	const int	jx = (int)xj;
	const int	jy = (int)yj;
	const int	jz = (int)zj;
	const int	j  = nx*ny*jz + nx*jy + jx;

	if		(coi[j] == '+')		return _FAR_OUT_;
	else if	(coi[j] == '-')		return _FAR_IN_;

	const FLOAT	dxs = dx/(FLOAT)nxs;	
	const FLOAT	dys = dy/(FLOAT)nys;
	const FLOAT	dzs = dz/(FLOAT)nzs;

	xj = (x - jx*dx - xoff)/dxs;
	yj = (y - jy*dy - yoff)/dys;
	zj = (z - jz*dz - zoff)/dzs;

	const int	ix = (int)xj;
	const int	iy = (int)yj;
	const int	iz = (int)zj;

	const FLOAT	gx = xj - (FLOAT)ix;
	const FLOAT	gy = yj - (FLOAT)iy;
	const FLOAT	gz = zj - (FLOAT)iz;

	const int	i = (nxs + 1)*(nys + 1)*iz + (nxs + 1)*iy + ix;

	const float	*f = fs[j];

	return	bi_interpolation(
				gx, gy, gz, 
				f[i   ], f[i+1   ], f[i+nxs+1   ], f[i+nxs+1+1   ], 
				f[i+ns], f[i+1+ns], f[i+nxs+1+ns], f[i+nxs+1+1+ns] 
				);
}


// STL_gpu.cu //
