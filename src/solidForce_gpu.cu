#include "solidForce_gpu.h"

#include "defineCUDA.h"
#include "functionLib.h"
#include "matharrayLib.h"


__global__ void
cuda_get_solidForce_tensor_bb (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv,
		  FLOAT	delta,
		  FLOAT	vis,
		  FLOAT	rho_ref,
		  FLOAT	vel_ref,
	      FLOAT	dx,
	int nx, int ny, int nz,
	int halo
	)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// index local //
	const int	nxy  = nx*ny;


	// stride //
	const int	stride_x = 1;
	const int	stride_y = nx;
	const int	stride_z = nx*ny;


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// gradient //
		const FLOAT	ux = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_x);
		const FLOAT	uy = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_y);
		const FLOAT	uz = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_z);

		const FLOAT	vx = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_x);
		const FLOAT	vy = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_y);
		const FLOAT	vz = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_z);

		const FLOAT	wx = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_x);
		const FLOAT	wy = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_y);
		const FLOAT	wz = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_z);


		// reference value //
		const FLOAT	rho_real = fabs(r[id_c0_c0_c0])*rho_ref;
//		const FLOAT	rho_real = rho_ref; 

		const FLOAT	vel = sqrt( pow(u[id_c0_c0_c0], 2) + pow(v[id_c0_c0_c0], 2) + pow(w[id_c0_c0_c0], 2) );
//		const FLOAT	vel_real = vel*vel_ref;

		const FLOAT	cp = 1.0/3.0*rho_real * pow(vel_ref, 2);


		// tensor //
		const FLOAT	p11 = - cp + vis * (2.0*ux)*vel_ref;
		const FLOAT	p22 = - cp + vis * (2.0*vy)*vel_ref;
		const FLOAT	p33 = - cp + vis * (2.0*wz)*vel_ref;

		const FLOAT	p12 =        vis * (vx + uy)*vel_ref;
		const FLOAT	p23 =        vis * (wy + vz)*vel_ref;
		const FLOAT	p13 =        vis * (uz + wx)*vel_ref;

//		const FLOAT	p11 = - cp;
//		const FLOAT	p22 = - cp;
//		const FLOAT	p33 = - cp;
//
//		const FLOAT	p12 =  0.0;
//		const FLOAT	p23 =  0.0;
//		const FLOAT	p13 =  0.0;


		// inner product //
//		const FLOAT	ds = 1.0;
		const FLOAT	ds = dx*dx;

		const FLOAT	clv = lv[id_c0_c0_c0];
		FLOAT	delta_s;
		if (clv >= 0.0 || clv <  -dx)	{ delta_s = 0.0; }
//		if (clv >  0.0 || clv <= -dx)	{ delta_s = 0.0; }
		else							{ delta_s =  ds; }

		// normal vector //
		FLOAT	nvec[3] = { 0.0, 0.0, 0.0 };	// fluid to solid //
		if (lv[id_c0_c0_c0 + stride_x] > (FLOAT)0.0)	{ nvec[0] +=  1.0; }	// solid //
		if (lv[id_c0_c0_c0 - stride_x] > (FLOAT)0.0)	{ nvec[0] += -1.0; }	// solid //

		if (lv[id_c0_c0_c0 + stride_y] > (FLOAT)0.0)	{ nvec[1] +=  1.0; }
		if (lv[id_c0_c0_c0 - stride_y] > (FLOAT)0.0)	{ nvec[1] += -1.0; }

		if (lv[id_c0_c0_c0 + stride_z] > (FLOAT)0.0)	{ nvec[2] +=  1.0; }
		if (lv[id_c0_c0_c0 - stride_z] > (FLOAT)0.0)	{ nvec[2] += -1.0; }


		// nvec = -nvec ( solid to fluid ) //
		const FLOAT	sforce_inner_x = - (p11*nvec[0] + p12*nvec[1] + p13*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_y = - (p12*nvec[0] + p22*nvec[1] + p23*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_z = - (p13*nvec[0] + p23*nvec[1] + p33*nvec[2]) * delta_s;


		// update //
		force_inner_x[id_c0_c0_c0] = sforce_inner_x;
		force_inner_y[id_c0_c0_c0] = sforce_inner_y;
		force_inner_z[id_c0_c0_c0] = sforce_inner_z;
	} 
}


__global__ void
cuda_get_solidForce_tensor_bounce_back (
	      FLOAT*	force_inner_x,
	      FLOAT*	force_inner_y,
	      FLOAT*	force_inner_z,
	const FLOAT*	r,
	const FLOAT*	u,
	const FLOAT*	v,
	const FLOAT*	w,
	const FLOAT*	lv,
	const FLOAT		delta,
	const FLOAT		vis,
	const FLOAT		rho_ref,
	const FLOAT		vel_ref,
	const FLOAT		dx,
	const int		nx, 
	const int		ny, 
	const int		nz,
	const int		halo
	)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// index local //
	const int	nxy  = nx*ny;


	// stride //
	const int	stride_x = 1;
	const int	stride_y = nx;
	const int	stride_z = nx*ny;


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// gradient //
		const FLOAT	ux = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_x);
		const FLOAT	uy = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_y);
		const FLOAT	uz = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_z);

		const FLOAT	vx = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_x);
		const FLOAT	vy = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_y);
		const FLOAT	vz = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_z);

		const FLOAT	wx = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_x);
		const FLOAT	wy = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_y);
		const FLOAT	wz = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_z);


		// reference value //
		const FLOAT	rho_real = fabs(r[id_c0_c0_c0])*rho_ref;

//		const FLOAT	vel      = sqrt( pow(u[id_c0_c0_c0], 2) + pow(v[id_c0_c0_c0], 2) + pow(w[id_c0_c0_c0], 2) );
//		const FLOAT	vel_real = vel*vel_ref;

		const FLOAT	cp = rho_real/3.0 * pow(vel_ref, 2);


		// tensor //
		const FLOAT	p11 = - cp + vis * (2.0*ux)*vel_ref;
		const FLOAT	p22 = - cp + vis * (2.0*vy)*vel_ref;
		const FLOAT	p33 = - cp + vis * (2.0*wz)*vel_ref;

		const FLOAT	p12 =        vis * (vx + uy)*vel_ref;
		const FLOAT	p23 =        vis * (wy + vz)*vel_ref;
		const FLOAT	p13 =        vis * (uz + wx)*vel_ref;

//		const FLOAT	p11 = - cp;
//		const FLOAT	p22 = - cp;
//		const FLOAT	p33 = - cp;
//
//		const FLOAT	p12 =  0.0;
//		const FLOAT	p23 =  0.0;
//		const FLOAT	p13 =  0.0;


		// inner product //
//		const FLOAT	ds = 1.0;
		const FLOAT	ds = dx*dx;

		const FLOAT	clv     = lv[id_c0_c0_c0];
		const FLOAT	delta_s = (clv >= 0.0) ? 0.0 : ds;
//		const FLOAT	delta_s = (clv >= 0.0 || clv < -dx) ? 0.0 : ds;


		// normal vector //
		FLOAT	nvec[3] = { 0.0, 0.0, 0.0 };	// fluid to solid //
		if (lv[id_c0_c0_c0 + stride_x] >= (FLOAT)0.0)	{ nvec[0] +=  1.0; }	// solid //
		if (lv[id_c0_c0_c0 - stride_x] >= (FLOAT)0.0)	{ nvec[0] += -1.0; }	// solid //

		if (lv[id_c0_c0_c0 + stride_y] >= (FLOAT)0.0)	{ nvec[1] +=  1.0; }
		if (lv[id_c0_c0_c0 - stride_y] >= (FLOAT)0.0)	{ nvec[1] += -1.0; }

		if (lv[id_c0_c0_c0 + stride_z] >= (FLOAT)0.0)	{ nvec[2] +=  1.0; }
		if (lv[id_c0_c0_c0 - stride_z] >= (FLOAT)0.0)	{ nvec[2] += -1.0; }


		// nvec = -nvec ( solid to fluid ) //
		const FLOAT	sforce_inner_x = - (p11*nvec[0] + p12*nvec[1] + p13*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_y = - (p12*nvec[0] + p22*nvec[1] + p23*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_z = - (p13*nvec[0] + p23*nvec[1] + p33*nvec[2]) * delta_s;


		// update //
		force_inner_x[id_c0_c0_c0] = sforce_inner_x;
		force_inner_y[id_c0_c0_c0] = sforce_inner_y;
		force_inner_z[id_c0_c0_c0] = sforce_inner_z;
	} 
}


__global__ void
cuda_get_solidForce_tensor_bb_nvec (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv,
		  FLOAT	delta,
		  FLOAT	vis,
		  FLOAT	rho_ref,
		  FLOAT	vel_ref,
	      FLOAT	dx,
	int nx, int ny, int nz,
	int halo
	)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// index local //
	const int	nxy  = nx*ny;


	// stride //
	const int	stride_x = 1;
	const int	stride_y = nx;
	const int	stride_z = nx*ny;


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// normal vector //
		FLOAT	nvec[3];	// fluid to solid
//		matharrayLib::normal_vector (
		matharrayLib::normal_vector_upwind (
			nvec,
			&lv[id_c0_c0_c0],
			nx, ny, nz
			);

		nvec[0] = (nvec[0] >= 0.0) ? 1.0 : -1.0;
		nvec[1] = (nvec[1] >= 0.0) ? 1.0 : -1.0;
		nvec[2] = (nvec[2] >= 0.0) ? 1.0 : -1.0;

		// gradient //
		const FLOAT	ux = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_x);
		const FLOAT	uy = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_y);
		const FLOAT	uz = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_z);

		const FLOAT	vx = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_x);
		const FLOAT	vy = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_y);
		const FLOAT	vz = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_z);

		const FLOAT	wx = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_x);
		const FLOAT	wy = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_y);
		const FLOAT	wz = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_z);


		// reference value //
		const FLOAT	rho_real = r[id_c0_c0_c0]*rho_ref;

		const FLOAT	vel = sqrt( pow(u[id_c0_c0_c0], 2) + pow(v[id_c0_c0_c0], 2) + pow(w[id_c0_c0_c0], 2) );
//		const FLOAT	vel_real = vel*vel_ref;

//		const FLOAT	p_ref     = 101325.0;
//		const FLOAT	cp = rho_real*(p_ref/rho_ref) - 0.5*rho_real * pow(vel_real, 2);
//		const FLOAT	cp = rho_real*(p_ref/rho_ref);
		const FLOAT	cp = 1.0/3.0*rho_real * pow(vel_ref, 2);
//		const FLOAT	cp = p_ref;


		// tensor //
		const FLOAT	p11 = - cp + vis * (2.0*ux)*vel_ref;
		const FLOAT	p22 = - cp + vis * (2.0*vy)*vel_ref;
		const FLOAT	p33 = - cp + vis * (2.0*wz)*vel_ref;

		const FLOAT	p12 =        vis * (vx + uy)*vel_ref;
		const FLOAT	p23 =        vis * (wy + vz)*vel_ref;
		const FLOAT	p13 =        vis * (uz + wx)*vel_ref;

//		const FLOAT	p11 = - cp;
//		const FLOAT	p22 = - cp;
//		const FLOAT	p33 = - cp;
//
//		const FLOAT	p12 =  0.0;
//		const FLOAT	p23 =  0.0;
//		const FLOAT	p13 =  0.0;


		// inner product //
//		const FLOAT	ds = 1.0;
		const FLOAT	ds = dx*dx;


		const FLOAT	clv     = lv[id_c0_c0_c0];
		FLOAT	delta_s;
		if (clv > 0.0 || clv <= -dx)	{ delta_s = 0.0; }
		else							{ delta_s = ds; }

		const int	sgn_nvec_x = (nvec[0] > 0.0) ? 1 : -1;
		const int	sgn_nvec_y = (nvec[1] > 0.0) ? 1 : -1;
		const int	sgn_nvec_z = (nvec[2] > 0.0) ? 1 : -1;

		if (lv[id_c0_c0_c0 + sgn_nvec_x*stride_x] < 0.0)	{ nvec[0] = 0.0; }
		if (lv[id_c0_c0_c0 + sgn_nvec_y*stride_y] < 0.0)	{ nvec[1] = 0.0; }
		if (lv[id_c0_c0_c0 + sgn_nvec_z*stride_z] < 0.0)	{ nvec[2] = 0.0; }


//		const FLOAT	d_nvec3 = 1.0 / (sqrt( pow(nvec[0], 2) + pow(nvec[1], 2) + pow(nvec[2], 2) ) + coefficient::NON_ZERO_EP);
//		nvec[0] *= d_nvec3;
//		nvec[1] *= d_nvec3;
//		nvec[2] *= d_nvec3;


		// nvec = -nvec ( solid to fluid )
		const FLOAT	sforce_inner_x = - (p11*nvec[0] + p12*nvec[1] + p13*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_y = - (p12*nvec[0] + p22*nvec[1] + p23*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_z = - (p13*nvec[0] + p23*nvec[1] + p33*nvec[2]) * delta_s;


		// update //
		force_inner_x[id_c0_c0_c0] = sforce_inner_x;
		force_inner_y[id_c0_c0_c0] = sforce_inner_y;
		force_inner_z[id_c0_c0_c0] = sforce_inner_z;
	} 
}


__global__ void
cuda_get_solidForce_tensor (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv,
		  FLOAT	delta,
		  FLOAT	vis,
		  FLOAT	rho_ref,
		  FLOAT	vel_ref,
	      FLOAT	dx,
	int nx, int ny, int nz,
	int halo
	)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// index local //
	const int	nxy  = nx*ny;


	// stride //
	const int	stride_x = 1;
	const int	stride_y = nx;
	const int	stride_z = nx*ny;


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// normal vector //
		FLOAT	nvec[3];	// fluid to solid
		matharrayLib::normal_vector_upwind (
			nvec,
			&lv[id_c0_c0_c0],
			nx, ny, nz
			);

		// gradient //
		const FLOAT	ux = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_x);
		const FLOAT	uy = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_y);
		const FLOAT	uz = matharrayLib::fx_2nd_central (&u[id_c0_c0_c0], dx, stride_z);

		const FLOAT	vx = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_x);
		const FLOAT	vy = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_y);
		const FLOAT	vz = matharrayLib::fx_2nd_central (&v[id_c0_c0_c0], dx, stride_z);

		const FLOAT	wx = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_x);
		const FLOAT	wy = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_y);
		const FLOAT	wz = matharrayLib::fx_2nd_central (&w[id_c0_c0_c0], dx, stride_z);

		// reference value //
		const FLOAT	rho_real = r[id_c0_c0_c0]*rho_ref;

		const FLOAT	vel = sqrt( pow(u[id_c0_c0_c0], 2) + pow(v[id_c0_c0_c0], 2) + pow(w[id_c0_c0_c0], 2) );
//		const FLOAT	vel_real = vel*vel_ref;

//		const FLOAT	p_ref     = 101325.0;
//		const FLOAT	cp = rho_real*(p_ref/rho_ref) - 0.5*rho_real * pow(vel_real, 2);
//		const FLOAT	cp = rho_real*(p_ref/rho_ref);
		const FLOAT	cp = 1.0/3.0*rho_real * pow(vel_ref, 2);
//		const FLOAT	cp = p_ref;


		// tensor //
//		const FLOAT	p11 = - cp + vis * (2.0*ux)*vel_ref;
//		const FLOAT	p22 = - cp + vis * (2.0*vy)*vel_ref;
//		const FLOAT	p33 = - cp + vis * (2.0*wz)*vel_ref;
//
//		const FLOAT	p12 =        vis * (vx + uy)*vel_ref;
//		const FLOAT	p23 =        vis * (wy + vz)*vel_ref;
//		const FLOAT	p13 =        vis * (uz + wx)*vel_ref;

		const FLOAT	p11 = - cp;
		const FLOAT	p22 = - cp;
		const FLOAT	p33 = - cp;

		const FLOAT	p12 =  0.0;
		const FLOAT	p23 =  0.0;
		const FLOAT	p13 =  0.0;


		// inner product //
		const FLOAT	ds  = dx*dx;

		const FLOAT	clv     = lv[id_c0_c0_c0];
//		const FLOAT	delta_s = functionLib::heaviside_function_cos (clv, delta) * ds;

//		const FLOAT	delta_s = functionLib::heaviside_function_cos       (clv, delta)           *dx * ds;
		const int	sgn_fluid = -1;
		const FLOAT	delta_s = functionLib::heaviside_function_cos_fluid (clv, delta, sgn_fluid)*dx * ds;


		// nvec = -nvec ( solid to fluid )
		const FLOAT	sforce_inner_x = - (p11*nvec[0] + p12*nvec[1] + p13*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_y = - (p12*nvec[0] + p22*nvec[1] + p23*nvec[2]) * delta_s;
		const FLOAT	sforce_inner_z = - (p13*nvec[0] + p23*nvec[1] + p33*nvec[2]) * delta_s;


		// update //
		force_inner_x[id_c0_c0_c0] = sforce_inner_x;
		force_inner_y[id_c0_c0_c0] = sforce_inner_y;
		force_inner_z[id_c0_c0_c0] = sforce_inner_z;
	} 
}


__global__ void
cuda_filter_solidForce (
	      FLOAT	*force_inner_x,
	      FLOAT	*force_inner_y,
	      FLOAT	*force_inner_z,
	const int	*id_solidData,
	int nx, int ny, int nz,
	int halo
	)
{
	// cuda index //
	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	// index local //
	const int	nxy  = nx*ny;

	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z = id_zs + k;

		// index //
		const int	id_c0_c0_c0 = id_x + nx*id_y + nxy*id_z;


		// calculation //
		FLOAT	sforce_inner_x = force_inner_x [id_c0_c0_c0];
		FLOAT	sforce_inner_y = force_inner_y [id_c0_c0_c0];
		FLOAT	sforce_inner_z = force_inner_z [id_c0_c0_c0];

		if (id_solidData[id_c0_c0_c0] != 0) {
			sforce_inner_x = (FLOAT)0.0;
			sforce_inner_y = (FLOAT)0.0;
			sforce_inner_z = (FLOAT)0.0;
		}

		// update //
		force_inner_x[id_c0_c0_c0] = sforce_inner_x;
		force_inner_y[id_c0_c0_c0] = sforce_inner_y;
		force_inner_z[id_c0_c0_c0] = sforce_inner_z;
	} 
}


// solidForce_gpu.cu //
