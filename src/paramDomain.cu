#include "paramDomain.h"

// C, C++ //
#include <fstream>

// include //
#include "option_parser.h"
#include "defineCUDA.h"

// Lib //
#include "functionLib.h"
#include "allocateLib.h"


// public //
void	paramDomain::
set_paramDomain (
	char			*program_name,
	int				argc,
   	char			*argv[], 
	const MPIinfo	&mpi,
	const defineMemory::FlagHostDevice	flag_memory
	)
{
	// host or device memory //
	flag_memory_ = flag_memory;


	// option parser //
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);


	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// Init Domain //
	// lattice boltzmann method (lbm) //
	const FLOAT	vel_cfl_ref = parser.velocity_lbm(0);
	const FLOAT	cfl_ref     = parser.velocity_lbm(1);
	const FLOAT	c_ref	= (vel_cfl_ref==0) ? 1.0 / cfl_ref : vel_cfl_ref / cfl_ref;
//	const FLOAT	c_ref       = vel_cfl_ref / cfl_ref;



	// Time //
	const FLOAT	time_end = parser.time();

	const int	ncpu_x = mpi.ncpu_x;
	const int	ncpu_y = mpi.ncpu_y;
	const int	ncpu_z = mpi.ncpu_z;

	const int	rank_x = mpi.rank_x;
	const int	rank_y = mpi.rank_y;
	const int	rank_z = mpi.rank_z;


	// domain //
	const int	halo = parser.halo_grid();


	const int	cn      = parser.cnumgrid(0);	// number of grid per reference domain length
	const int	axis    = parser.cnumgrid(1);	// reference domain direction
	const FLOAT	clength = parser.length(axis);	// reference domain length [m]
	const FLOAT	cdx     = clength / (FLOAT)cn;	// grid size [m]


	// nx, ny, nz per GPU
	const FLOAT	fnx = parser.length(0) / ncpu_x / cdx;	// domain length / grids / grid size
	const FLOAT	fny = parser.length(1) / ncpu_y / cdx;
	const FLOAT	fnz = parser.length(2) / ncpu_z / cdx;

	const int	min_block  = MIN_BLOCKDIM;
	const int	blockdim_x = (BLOCKDIM_X > min_block) ? BLOCKDIM_X : min_block;
	const int	blockdim_y = (BLOCKDIM_Y > min_block) ? BLOCKDIM_Y : min_block;
	const int	blockdim_z = (BLOCKDIM_Z > min_block) ? BLOCKDIM_Z : min_block;

	const int	nx = functionLib::cast_value_large (fnx, blockdim_x) + 2*halo;
	const int	ny = functionLib::cast_value_large (fny, blockdim_y) + 2*halo;
	const int	nz = functionLib::cast_value_large (fnz, blockdim_z) + 2*halo;
	const int	nn = nx*ny*nz;


	const int	nxg = (nx - 2*halo) * (ncpu_x);
	const int	nyg = (ny - 2*halo) * (ncpu_y);
	const int	nzg = (nz - 2*halo) * (ncpu_z);

//	const FLOAT	time_coef = parser.time_coef() * 1.0e-0 / ((nxg + nyg + nzg)/3.0);
//	const FLOAT	dt = time * time_coef;
	const FLOAT	dt = cdx / c_ref;	// CFL = u * dt /dx
									// 1 = c_ref * dt/dx

	const FLOAT	xg_length = nxg * cdx;	// domain size [m]
	const FLOAT	yg_length = nyg * cdx;
	const FLOAT	zg_length = nzg * cdx;

	const FLOAT	x_length = xg_length / ncpu_x;	// domain size per GPU
	const FLOAT	y_length = yg_length / ncpu_y;
	const FLOAT	z_length = zg_length / ncpu_z;

	const FLOAT	xg_min = parser.domain_min(0);
	const FLOAT	yg_min = parser.domain_min(1);
	const FLOAT	zg_min = parser.domain_min(2);

	const FLOAT	xg_max = parser.domain_min(0) + xg_length;
	const FLOAT	yg_max = parser.domain_min(1) + yg_length;
	const FLOAT	zg_max = parser.domain_min(2) + zg_length;

	const FLOAT	dx = cdx;


	// definition of member variables //
	// lattice boltzmann method //
	domain_.c_ref   = c_ref;
	domain_.cfl_ref = cfl_ref;

	domain_.vel_cfl_ref = vel_cfl_ref;	// MOD2019a

	// nx, ny, nz //
	domain_.nx = nx;
	domain_.ny = ny;
	domain_.nz = nz;
	domain_.nn = nn;

	domain_.halo = halo;


	// NXG, NYG, NZG //
	domain_.nxg = nxg;
	domain_.nyg = nyg;
	domain_.nzg = nzg;

	// Domain length, dt //
	domain_.time_end = time_end;
	domain_.time     = 0.0;
	domain_.dt   = dt;


	// mean length //
	domain_.n0 = nxg;


	// number of step //
//	domain_.step = (int)(time / domain_.dt);
	domain_.step = (int)(time_end);


	// global //
	domain_.xg_min = xg_min;
	domain_.yg_min = yg_min;
	domain_.zg_min = zg_min;

	domain_.xg_max = xg_max;
	domain_.yg_max = yg_max;
	domain_.zg_max = zg_max;


	// local //
	domain_.x_min = domain_.xg_min + rank_x*x_length;
	domain_.y_min = domain_.yg_min + rank_y*y_length;
	domain_.z_min = domain_.zg_min + rank_z*z_length;

	domain_.x_max = domain_.x_min + x_length;
	domain_.y_max = domain_.y_min + y_length;
	domain_.z_max = domain_.z_min + z_length;

	domain_.xg_length = xg_length;
	domain_.yg_length = yg_length;
	domain_.zg_length = zg_length;

	domain_.dx = dx;


	// axis //
	if (flag_memory == defineMemory::Host_Memory) {
		allocateLib::new_host   (&domain_.x,  nx);
		allocateLib::new_host   (&domain_.y,  ny);
		allocateLib::new_host   (&domain_.z,  nz);
	
		init_axis (domain_.x, domain_.x_min, dx, nx, halo);
		init_axis (domain_.y, domain_.y_min, dx, ny, halo);
		init_axis (domain_.z, domain_.z_min, dx, nz, halo);
	}
	if (flag_memory == defineMemory::Device_Memory) {
		allocateLib::new_device (&domain_.x,  nx);
		allocateLib::new_device (&domain_.y,  ny);
		allocateLib::new_device (&domain_.z,  nz);
	}

//	for (int i=0; i<nx; i++) { domain_.x[i] = domain_.x_min + (i-halo + (FLOAT)0.5)*dx; }
//	for (int i=0; i<ny; i++) { domain_.y[i] = domain_.y_min + (i-halo + (FLOAT)0.5)*dx; }
//	for (int i=0; i<nz; i++) { domain_.z[i] = domain_.z_min + (i-halo + (FLOAT)0.5)*dx; }
}


// memcpy //
void	paramDomain::
memcpy_paramDomain (
	      paramDomain	&pdomain_n,
	const paramDomain	&pdomain
	)
{
	// lattice boltzmann method //
	pdomain_n.domain_.c_ref   = pdomain.c_ref();
	pdomain_n.domain_.cfl_ref = pdomain.cfl_ref();


	pdomain_n.domain_.vel_cfl_ref = pdomain.vel_cfl_ref();     // MOD2019a

	// number of grid points (global) //
	pdomain_n.domain_.nxg = pdomain.nxg();
	pdomain_n.domain_.nyg = pdomain.nyg();
	pdomain_n.domain_.nzg = pdomain.nzg();

	// number of grid points (local) //
	pdomain_n.domain_.nx = pdomain.nx();
	pdomain_n.domain_.ny = pdomain.ny();
	pdomain_n.domain_.nz = pdomain.nz();
	pdomain_n.domain_.nn = pdomain.nn();
	pdomain_n.domain_.n0 = pdomain.n0();

	// number of step //
	pdomain_n.domain_.step = pdomain.step();

	// halo //
	pdomain_n.domain_.halo = pdomain.halo();

	// time // 
	pdomain_n.domain_.time_end = pdomain.time_end();
	pdomain_n.domain_.time     = pdomain.time();
	pdomain_n.domain_.dt       = pdomain.dt();

	// dx, dy, dz //
	pdomain_n.domain_.dx     = pdomain.dx();

	// local //
	pdomain_n.domain_.x_min = pdomain.x_min();
	pdomain_n.domain_.y_min = pdomain.y_min();
	pdomain_n.domain_.z_min = pdomain.z_min();

	pdomain_n.domain_.x_max = pdomain.x_max();
	pdomain_n.domain_.y_max = pdomain.y_max();
	pdomain_n.domain_.z_max = pdomain.z_max();

	// global //
	pdomain_n.domain_.xg_length = pdomain.xg_length();
	pdomain_n.domain_.yg_length = pdomain.yg_length();
	pdomain_n.domain_.zg_length = pdomain.zg_length();

	pdomain_n.domain_.xg_min = pdomain.xg_min();
	pdomain_n.domain_.yg_min = pdomain.yg_min();
	pdomain_n.domain_.zg_min = pdomain.zg_min();

	pdomain_n.domain_.xg_max = pdomain.xg_max();
	pdomain_n.domain_.yg_max = pdomain.yg_max();
	pdomain_n.domain_.zg_max = pdomain.zg_max();


	// region //
	const int	nx = pdomain.nx();
	const int	ny = pdomain.ny();
	const int	nz = pdomain.nz();

	cudaMemcpy(pdomain_n.domain_.x, pdomain.x(), sizeof(FLOAT)*(nx), cudaMemcpyDefault);
	cudaMemcpy(pdomain_n.domain_.y, pdomain.y(), sizeof(FLOAT)*(ny), cudaMemcpyDefault);
	cudaMemcpy(pdomain_n.domain_.z, pdomain.z(), sizeof(FLOAT)*(nz), cudaMemcpyDefault);
}


void	paramDomain::
time_evolution ()
{
//	domain_.time += domain_.dt;
	domain_.time += 1.0;
}


void	paramDomain::
cout_Domain ()
{
	// cout
	std::cout << "Time (dt)         = " << domain_.time_end << " ( " << domain_.dt << " ) "<< std::endl;

	std::cout << "nx,  ny,  nz      = " << domain_.nx  << ", "  << domain_.ny  << ", " << domain_.nz  << std::endl;
	std::cout << "nxg, nyg, nzg     = " << domain_.nxg << ", "  << domain_.nyg << ", " << domain_.nzg << std::endl;

	std::cout << "glength (x, y, z) = " << domain_.xg_length  << ", "  << domain_.yg_length << ", " << domain_.zg_length << std::endl;
	std::cout << std::endl;
}


void	paramDomain::
output_Condition ()
{
	std::ofstream	fout;
	fout.open("./condition.dat", std::ios::out);

	fout << domain_.time << std::endl;

	fout.close();
}


void	paramDomain::
read_Condition ()
{
	std::ifstream	fin;
	fin.open("./condition.dat");

	fin >> domain_.time;

	fin.close();
}


// private //
// cout : nx, ny, nz, nxg, nyg, nzg //
void	paramDomain::
check_Domain ()
{
	if (domain_.nx <= 0 || domain_.ny <= 0 || domain_.nz <= 0) {
		std::cout << "error : number of grid" << std::endl;
		exit(-2);
	}
}


// axis host //
void	paramDomain::
init_axis (
	FLOAT	*x,
	FLOAT	x_min,
	FLOAT	dx,
	int		nx,
	int		halo
	)
{
	for (int i=0; i<nx; i++) { x[i] = x_min + (i - halo + (FLOAT)0.5)*dx; }
}


// paramDomain //
