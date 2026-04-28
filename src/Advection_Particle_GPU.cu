#include "Advection_Particle_GPU.h"

#include "Math_Lib_Particle_GPU.h"
#include "defineReferenceVel.h"

// (YOKOUCHI 2020)
#include <curand_kernel.h>


__global__ void 
CUDA_Particle_Advection (
	ParticlePosition	*ppos,
	const FLOAT			*u, 
	const FLOAT			*v, 
	const FLOAT			*w,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max,
	FLOAT dx,
	FLOAT dt,
	FLOAT c_ref,
	int nx, int ny, int nz,
	int num,
	int halo
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }
	if (ppos[id].state_p != PARTICLE_CAL)	{ return; }

	FLOAT	x = ppos[id].x_p;
	FLOAT	y = ppos[id].y_p;
	FLOAT	z = ppos[id].z_p;


	// velocity //
	// runge-kutta 1
	FLOAT	u_p = Interpolate_Particle_Velocity(u, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	FLOAT	v_p = Interpolate_Particle_Velocity(v, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	FLOAT	w_p = Interpolate_Particle_Velocity(w, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);


	const FLOAT	vel = sqrt( u_p*u_p + v_p*v_p + w_p*w_p );


	// time-integration //
	ppos[id].x_p += (u_p*c_ref * dt);
	ppos[id].y_p += (v_p*c_ref * dt);
	ppos[id].z_p += (w_p*c_ref * dt);

//	ppos[id].x_p += (u_p * dt);
//	ppos[id].y_p += (v_p * dt);
//	ppos[id].z_p += (w_p * dt);

	ppos[id].vel_p = vel;
}

//(YOKOUCHI 2020)
__global__ void 
CUDA_Particle_Advection_LSM (
	ParticlePosition	*ppos,
	const FLOAT			*l_obs,
	const FLOAT			*u, 
	const FLOAT			*v, 
	const FLOAT			*w,
//	const FLOAT			*vis_sgs,
//	const FLOAT			*vis_sgs_old,
	const FLOAT			*tke_sgs,
	const FLOAT			*tke_sgs_old,
	const FLOAT			*um,
	const FLOAT			*vm,
	const FLOAT			*wm,
	FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
	FLOAT xs_max, FLOAT ys_max, FLOAT zs_max,
	FLOAT dx,
	FLOAT dt,
	FLOAT c_ref,
	int nx, int ny, int nz,
	int num,
	int halo,
	curandState				*state
	)
{
	const int	id   = threadIdx.x + blockDim.x*blockIdx.x;

	if (id >= num) 							{ return; }
	if (ppos[id].state_p != PARTICLE_CAL)	{ return; }

	FLOAT	x = ppos[id].x_p;
	FLOAT	y = ppos[id].y_p;
	FLOAT	z = ppos[id].z_p;

	// velocity //
	// runge-kutta 1
	FLOAT	u_p = Interpolate_Particle_Velocity(u, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	FLOAT	v_p = Interpolate_Particle_Velocity(v, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	FLOAT	w_p = Interpolate_Particle_Velocity(w, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	
	// sgs //
	// White noise //
	FLOAT	rnd;

	// mean velocity //
	FLOAT	um_p = Interpolate_Particle_Velocity(um, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	FLOAT	vm_p = Interpolate_Particle_Velocity(vm, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);
	FLOAT	wm_p = Interpolate_Particle_Velocity(wm, x, y, z, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

	// GS TKE //
	const	FLOAT	us_p = u_p - um_p;
	const	FLOAT	vs_p = v_p - vm_p;
	const	FLOAT	ws_p = w_p - wm_p;

	const	FLOAT	k_GS = 0.5 * (us_p*us_p + vs_p*vs_p + ws_p*ws_p);
	//const 	FLOAT	k_GS = um_p;

	// velocity //
	FLOAT	u_s = ppos[id].u_sgs;
	FLOAT	v_s = ppos[id].v_sgs;
	FLOAT	w_s = ppos[id].w_sgs;
	
	// u_sgs //
	int	dim_flg	= 0;
		rnd	= curand_normal(&state[id]);

//		u_s 	+= Random_SGS_Velocity_Vis_sgs(u_s, vis_sgs, vis_sgs_old, k_GS, x, y, z,  xs_min, ys_min, zs_min, rnd, dx, dx, dx, nx, ny, nz, halo, dt, dim_flg);
		u_s	+= Random_SGS_Velocity_Energy(u_s, tke_sgs, tke_sgs_old, k_GS, x, y, z, xs_min, ys_min, zs_min, rnd, dx, dx, dx, nx, ny, nz, halo, dt, dim_flg);
	// v_sgs //
		dim_flg = 1;
		rnd	= curand_normal(&state[id]);
//		v_s 	+= Random_SGS_Velocity_Vis_sgs(v_s, vis_sgs, vis_sgs_old, k_GS, x, y, z,  xs_min, ys_min, zs_min, rnd, dx, dx, dx, nx, ny, nz, halo, dt, dim_flg);
		v_s	+= Random_SGS_Velocity_Energy(v_s, tke_sgs, tke_sgs_old, k_GS, x, y, z, xs_min, ys_min, zs_min, rnd, dx, dx, dx, nx, ny, nz, halo, dt, dim_flg);
	
	// w_sgs //
		dim_flg = 2;
		rnd	= curand_normal(&state[id]);
//		w_s 	+= Random_SGS_Velocity_Vis_sgs(w_s, vis_sgs, vis_sgs_old, k_GS, x, y, z,  xs_min, ys_min, zs_min, rnd, dx, dx, dx, nx, ny, nz, halo, dt, dim_flg);
		w_s	+= Random_SGS_Velocity_Energy(w_s, tke_sgs, tke_sgs_old, k_GS, x, y, z, xs_min, ys_min, zs_min, rnd, dx, dx, dx, nx, ny, nz, halo, dt, dim_flg);

	// merge velocity //
	FLOAT u_new = u_p + u_s;
	FLOAT v_new = v_p + v_s;
	FLOAT w_new = w_p + w_s;

	// new position //
	FLOAT x_new = x + (u_new*c_ref * dt); 
	FLOAT y_new = y + (v_new*c_ref * dt);
	FLOAT z_new = z + (w_new*c_ref * dt);	
	
	// reflection condition //
//	if(z_new<0){
//		z_new = -z_new;
//		w_s = -w_s;	}	// MOD 2021
	particle_reflection(l_obs, x, y, z, x_new, y_new, z_new, u_s, v_s, w_s, xs_min, ys_min, zs_min, dx, dx, dx, nx, ny, nz, halo);

	// new velocity //
	u_p = u_p + u_s;
	v_p = v_p + v_s;
	w_p = w_p + w_s;
	


	const FLOAT	vel = sqrt( u_p*u_p + v_p*v_p + w_p*w_p );



	// save //
	ppos[id].x_p = x_new; 
	ppos[id].y_p = y_new;
	ppos[id].z_p = z_new;

//	ppos[id].x_p += (u_p * dt);
//	ppos[id].y_p += (v_p * dt);
//	ppos[id].z_p += (w_p * dt);

	ppos[id].u_sgs = u_s;
	ppos[id].v_sgs = v_s;
	ppos[id].w_sgs = w_s;

	ppos[id].vel_p = vel;
}

// White noise
// Initilaize curandState
__global__ void setCurand (
	unsigned long long	seed, 
	curandState 		*state,
	const int		nn
	) 
{
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (id >= nn )			{return;}	
	curand_init(seed, id, 0, &state[id]);
}

__global__ void genrand_normal (
	FLOAT		*rnd_d,
	curandState	*state,
	const int	nn
	)
{
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= nn ) 			{return;}
	rnd_d[id] = curand_normal(&state[id]);
	
}

// Advection_Particle_GPU.cu //
