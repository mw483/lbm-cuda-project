#include "Cal_Particle.h"

#include <mpi.h>
#include <fstream>
#include "option_parser.h"
#include "Advection_Particle_GPU.h"
#include "boundary_flag_particle.h"
#include "Generate_Particle.h"
#include "Renumber_Particle.h"

#include "macroCUDA.h"
#include "functionLib.h"
#include "MPI_Particle.h"

// (YOKOUCHI 2020)
#include "Define_user.h"
#include "calculation.h"
#include <curand_kernel.h>

void CalParticle::
set (
	      int			argc,
	      char			*argv[],
	const paramMPI		&pmpi,
	const paramDomain	&pdomain,
	      Particle		*particle
	)
{
	// option parser //
	const char	program_args[] = "[options...]";
	OptionParser parser("CalParticle", program_args);


	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// class //
	rank_        = pmpi.rank();

	gen_step_    = particle->pfrag.gen_step;
	renum_step_  = gen_step_ * 5;

	thread_length_ = 256;

	// domain //
	domain_ = pdomain.domain();

}


void CalParticle::
particleAdvection (
	int				argc,
	char			*argv[], 
	int				t,
	Domain			&domain, 
	Particle		*particle_h,
	Particle		*particle_d,
	BasisVariables	*cbq_d,
	FluidProperty	*cfp_d
	)
{
	// option parser //
	const char	program_args[] = "[options...]";
	OptionParser parser("CalParticle", program_args);


	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// calculation //
	static bool	first_cal = 0;


	// particle generate //
	int		pstart,pnum;
	int		num_g  [3];
	FLOAT		point_g[3];
	FLOAT		vec_g  [3];


	// read 
	// (YOKOUCHI 2020)	
	if (user_flags::flg_particle == 1 || user_flags::flg_particle == 2) {
		particleRead_LSM(
				pstart, pnum,
				num_g,
				point_g,
				vec_g,
				rank_
				);
	} else {
		particleRead_uniform(
				pstart, pnum,
				num_g,
				point_g,
				vec_g,
				rank_
				);
	}
	// particle generate //
	// Generate_Particle.h
	if ( (t - gen_step_* pstart        ) >= 0 
	 &&  (t - gen_step_*(pstart + pnum)) <  0 
	 &&  (t - gen_step_*pstart)%gen_step_ == 0) {
//	if (t%gen_step_ == 0) {
		if (rank_ == 0 && first_cal == 0)	{ std::cout << "Generate_Particle" << std::endl; }

//		const int	flag_pgenerate = parser.flag_particle_generate();
		const int	flag_pgenerate = parser.flag_particle_generate();
		if (flag_pgenerate == 1) {
//			generate_particle (rank_, domain, particle_d, cfp_d);
			
			//(YOKOUCHI 2020)
			if (user_flags::flg_particle == 1 || user_flags::flg_particle == 2) {
				generate_particle_LSM (
					domain, particle_d, cfp_d,
					num_g,
					point_g,
					vec_g,
					t,
					pstart
					);
			} else {
				generate_particle_box_slice (
					domain, particle_d, cfp_d,
					num_g,
					point_g,
					vec_g
					);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


	// Renumber_ParticlePosition.h //
	if (t%renum_step_ == 0) {
		if (rank_ == 0 && first_cal == 0)	{ std::cout << "Renumber_ParticlePosition" << std::endl; }

		Renumber_ParticlePosition (particle_h, particle_d);
	}
	// Renumber_ParticlePosition.h //


	MPI_Barrier(MPI_COMM_WORLD);


	// advection, boundary check (device) //
	if (rank_ == 0 && first_cal == 0)	{ std::cout << "gpu_advection" << std::endl; }
	gpu_advection (particle_d->pgrid, particle_d->ppos, cbq_d);

	boundary_flag (particle_d->pgrid, particle_d->ppos);
	// advection, boundary check (device) //


	MPI_Barrier(MPI_COMM_WORLD);


	// MPI Particle //
	if (rank_ == 0 && first_cal == 0)	{ std::cout << "MPI_Particle" << std::endl; }
	MPI_Particle (rank_, domain, particle_h, particle_d);
	// MPI Particle //


	MPI_Barrier(MPI_COMM_WORLD);

	if (rank_ == 0 && first_cal == 0)	{ std::cout << "particleAdvection" << std::endl; }
	if (first_cal == 0)	{ first_cal = 1; }
}

// (YOKOUCHI 2020)
void CalParticle::
particleAdvection_LSM (
	int				argc,
	char			*argv[], 
	int				t,
	Domain			&domain, 
	Particle		*particle_h,
	Particle		*particle_d,
	BasisVariables	*cbq_d,
	FluidProperty	*cfp_d,
	Stress		*str_d
	)
{
	// option parser //
	const char	program_args[] = "[options...]";
	OptionParser parser("CalParticle", program_args);


	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// calculation //
	static bool	first_cal = 0;


	// particle generate //
	int		pstart,pnum;
	int		num_g  [3];
	FLOAT		point_g[3];
	FLOAT		vec_g  [3];


	// read 
	// (YOKOUCHI 2020)	
	particleRead_LSM(
		pstart, pnum,
		num_g,
		point_g,
		vec_g,
		rank_
		);
	
	// particle generate //
	// Generate_Particle.h
	if ( (t - gen_step_* pstart        ) >= 0 
	 &&  (t - gen_step_*(pstart + pnum)) <  0 
	 &&  (t - gen_step_*pstart)%gen_step_ == 0) {
//	if (t%gen_step_ == 0) {
		if (rank_ == 0 && first_cal == 0)	{ std::cout << "Generate_Particle" << std::endl; }

//		const int	flag_pgenerate = parser.flag_particle_generate();
		const int	flag_pgenerate = parser.flag_particle_generate();
		if (flag_pgenerate == 1) {
//			generate_particle (rank_, domain, particle_d, cfp_d);
			
		//(YOKOUCHI 2020)
		generate_particle_LSM (
			domain, particle_d, cfp_d,
			num_g,
			point_g,
			vec_g,
			t,
			pstart
			);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


	// Renumber_ParticlePosition.h //
	if (t%renum_step_ == 0) {
		if (rank_ == 0 && first_cal == 0)	{ std::cout << "Renumber_ParticlePosition" << std::endl; }

		Renumber_ParticlePosition (particle_h, particle_d);
	}
	// Renumber_ParticlePosition.h //


	MPI_Barrier(MPI_COMM_WORLD);


	// advection, boundary check (device) //
	if (rank_ == 0 && first_cal == 0)	{ std::cout << "gpu_advection" << std::endl; }
	int nn = parser.particle();
	gpu_advection_LSM (particle_d->pgrid, cfp_d,  particle_d->ppos, cbq_d, str_d, t, nn);

/*	// Check result (YOKOUCHI 2020)
	cudaMemcpy(particle_h->ppos, particle_d->ppos, sizeof(ParticlePosition)*nn, cudaMemcpyDeviceToHost);
	std::cout << particle_h->ppos[100].x_p << "  "  << particle_h->ppos[100].y_p <<"  "   << particle_h->ppos[100].z_p << "  " << particle_h->ppos[100].u_sgs << "  " << particle_h->ppos[100].v_sgs << "  " << particle_h->ppos[100].w_sgs << std::endl;
	// Check result //
		
	// Check random number //
	int num = particle_d->pgrid.parray_end;	
	FLOAT av = 0.0;
	FLOAT sigma = 0.0;
	for (int ii=0; ii<num; ii++){
		av +=  particle_h->ppos[ii].u_sgs/(3*num);
		av +=  particle_h->ppos[ii].v_sgs/(3*num);
		av +=  particle_h->ppos[ii].w_sgs/(3*num);
	}
	for (int ii=0; ii<num; ii++) {
		sigma +=  (particle_h->ppos[ii].u_sgs-av)*(particle_h->ppos[ii].u_sgs-av)/(3*num);
		sigma +=  (particle_h->ppos[ii].v_sgs-av)*(particle_h->ppos[ii].v_sgs-av)/(3*num);
		sigma +=  (particle_h->ppos[ii].w_sgs-av)*(particle_h->ppos[ii].w_sgs-av)/(3*num);
	}	
		 
	std::cout << " av = " << av << " sig = " << sigma << std::endl; 
	// Check random number //
*/		


	boundary_flag (particle_d->pgrid, particle_d->ppos);
	// advection, boundary check (device) //


	MPI_Barrier(MPI_COMM_WORLD);


	// MPI Particle //
	if (rank_ == 0 && first_cal == 0)	{ std::cout << "MPI_Particle" << std::endl; }
	MPI_Particle (rank_, domain, particle_h, particle_d);
	// MPI Particle //


	MPI_Barrier(MPI_COMM_WORLD);

	if (rank_ == 0 && first_cal == 0)	{ std::cout << "particleAdvection" << std::endl; }
	if (first_cal == 0)	{ first_cal = 1; }
}

// private //
// 粒子の移流 //
void CalParticle::
gpu_advection (
	const ParticleGrid		&pgrid,
	      ParticlePosition	*ppos,
	const BasisVariables	*const cbq
	)
{
	const FLOAT	*u = cbq->u_n;
	const FLOAT	*v = cbq->v_n;
	const FLOAT	*w = cbq->w_n;

	// domain //
	const int	halo = domain_.halo;

	const int	parray_end   = pgrid.parray_end;
	const int	block_number = parray_end/thread_length_ + 1;

	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;


	dim3	grid,
			block;

	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length_, 1, 1);

	// particle //
	FLOAT	xs_range[2], ys_range[2], zs_range[2];

	functionLib::init2(xs_range, domain_.x_min, domain_.x_max);
	functionLib::init2(ys_range, domain_.y_min, domain_.y_max);
	functionLib::init2(zs_range, domain_.z_min, domain_.z_max);
	// particle //

	const FLOAT	dx    = domain_.dx;
	const FLOAT	c_ref = domain_.c_ref;
	const FLOAT	dt    = dx/c_ref;	// CFL = u * dt / dx
									// 1 = c_ref * dt / dx

	CUDA_Particle_Advection <<< grid, block >>> (
		ppos,
		u, v, w,
		xs_range[0], ys_range[0], zs_range[0],
		xs_range[1], ys_range[1], zs_range[1],
		dx,
		dt,
		c_ref,
		nx, ny, nz,
		parray_end,
		halo
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}

void CalParticle::
gpu_advection_LSM (
	const ParticleGrid		&pgrid,
	      FluidProperty	*cfp,	
	      ParticlePosition	*ppos,
	const BasisVariables	*const cbq,
	const Stress		*const str,
	      int		t,
	      int		nn
	)
{
	const FLOAT	*u = cbq->u_n;
	const FLOAT	*v = cbq->v_n;
	const FLOAT	*w = cbq->w_n;

//	const FLOAT	*vis_sgs 	= str->vis_sgs;
//	const FLOAT	*vis_sgs_old 	= str->vis_sgs_old;
	const FLOAT	*tke_sgs	= str->TKE_sgs;
	const FLOAT	*tke_sgs_old	= str->TKE_sgs_old;
	const FLOAT	*um		= str->u_m;
	const FLOAT	*vm		= str->v_m;
	const FLOAT	*wm		= str->u_m;
	
	const FLOAT	*l_obs		= cfp->l_obs;
	

	// domain //
	const int	halo = domain_.halo;

	const int	parray_end   = pgrid.parray_end;
	const int	block_number = parray_end/thread_length_ + 1;

	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;


	dim3	grid,
			block;

	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length_, 1, 1);

	// particle //
	FLOAT	xs_range[2], ys_range[2], zs_range[2];

	functionLib::init2(xs_range, domain_.x_min, domain_.x_max);
	functionLib::init2(ys_range, domain_.y_min, domain_.y_max);
	functionLib::init2(zs_range, domain_.z_min, domain_.z_max);
	// particle //

	const FLOAT	dx    = domain_.dx;
	const FLOAT	c_ref = domain_.c_ref;
	const FLOAT	dt    = dx/c_ref;	// CFL = u * dt / dx
									// 1 = c_ref * dt / dx
	
	// make random number //
	curandState	*state;
//	FLOAT			*rnd_d;

	nn = parray_end;
//	cudaMalloc((void**)&rnd_d,	nn * sizeof(FLOAT));
	cudaMalloc((void**)&state,	nn * sizeof(curandState));

	genRandNum(state, nn, rank_, t); // set curandState random number //
	
	CUDA_Particle_Advection_LSM <<< grid, block >>> (
		ppos,
		l_obs,
		u, v, w,
//		vis_sgs, vis_sgs_old,
		tke_sgs, tke_sgs_old,
		um, vm, wm,
		xs_range[0], ys_range[0], zs_range[0],
		xs_range[1], ys_range[1], zs_range[1],
		dx,
		dt,
		c_ref,
		nx, ny, nz,
		parray_end,
		halo,
		state
		);


// test output 2021
//	std::cout<<xs_range[0]<<" "<<ys_range[0]<<" "<< zs_range[0]<<std::endl;
//	std::cout<<xs_range[1]<<" "<<ys_range[1]<<" "<< zs_range[1]<<std::endl;
//	std::cout<<" "<<dx<<" "<<dt<<" "<<c_ref<<" "<<nx<<" "<<ny<<" "<<nz<<std::endl;


	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize *****

//	cudaFree(rnd_d);
	cudaFree(state);
}

// 計算外領域判定 //
void CalParticle::
boundary_flag (
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos)
{
	const int	parray_end   = pgrid.parray_end;
	const int	block_number = parray_end/thread_length_ + 1;

	dim3	grid,
			block;
	functionLib::set_dim3(&grid,    block_number,   1, 1);
	functionLib::set_dim3(&block,   thread_length_, 1, 1);


	// region //
	FLOAT	xs_range[2], ys_range[2], zs_range[2];

	// global domain //
//	Init_FLOAT2(xs_range, 0.0, domain_.nxg);
//	Init_FLOAT2(ys_range, 0.0, domain_.nyg);
//	Init_FLOAT2(zs_range, 0.0, domain_.nzg);

	functionLib::init2(xs_range, domain_.xg_min, domain_.xg_max);
	functionLib::init2(ys_range, domain_.yg_min, domain_.yg_max);
	functionLib::init2(zs_range, domain_.zg_min, domain_.zg_max);
	// region //

	boundary_flag_particle_cuda <<< grid, block >>> (
		ppos,
		xs_range[0], ys_range[0], zs_range[0],
		xs_range[1], ys_range[1], zs_range[1],
		parray_end);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}

// (YOKOUCHI 2020)
void genRandNum (
//	FLOAT		*rnd_d,	
	curandState	*state,
	const int	nn,
	int		rank_,
	int		t
	)

{
	const int 	thread_length_ = 256;		
	const int	block_number   = nn/thread_length_ + 1;

	dim3	grid, block;

	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length_, 1, 1);
		
//	cudaMalloc((void**)&state,	nn * sizeof(curandState));

	unsigned long long sd = t * 1000 + rank_;
//	unsigned long long sd = rank_;

	setCurand <<< grid, block >>> (sd, state, nn);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize();

//	genrand_normal <<< grid, block >>> (rnd_d, state, nn);
	
//	CHECK_CUDA_ERROR("CUDA Error\n");
//	cudaThreadSynchronize();

}

void particleRead_uniform ( 
	int		&pstart,
	int 		&pnum,
	int		num_g[],
	FLOAT		point_g[],
	FLOAT		vec_g[],
	int		rank_)
{
	
	std::ifstream	fin;
	fin.open("./read_particle_box/read_particle_box.txt");
	if (!fin.is_open() && rank_ == 0) { std::cout << "File (particle_number.txt) is not opened." << std::endl;}
	fin >> pstart     >> pnum;
	fin >> num_g  [0] >> num_g  [1] >> num_g  [2];
	fin >> point_g[0] >> point_g[1] >> point_g[2];
	fin >> vec_g  [0] >> vec_g  [1] >> vec_g  [2];

	fin.close();
}

void particleRead_LSM (
	int		&pstart,
	int 		&pnum,
	int		num_g[],
	FLOAT		point_g[],
	FLOAT		vec_g[],
	int		rank_)
{	
	std::ifstream	fin;

	fin.open("./read_particle_box/read_particle_box.txt");
	if (!fin.is_open() && rank_ == 0) { std::cout << "File (read_particle_box.txt) is not opened." << std::endl;}
	fin >> pstart     >> pnum;
	fin >> num_g  [0] >> num_g  [1] >> num_g  [2];
	fin >> point_g[0] >> point_g[1] >> point_g[2];
	fin >> vec_g  [0] >> vec_g  [1] >> vec_g  [2];

	fin.close();

	fin.open("./particle_position/particle_number.txt");
	if (!fin.is_open() && rank_ == 0) { std::cout << "File (particle_number.txt) is not opened." << std::endl;}
	fin >> num_g [0]; // Overrides the above num_g
	fin.close();

	num_g[1] = 1;
	num_g[2] = 1;
}

	 
// Cal_Particle.cu //
