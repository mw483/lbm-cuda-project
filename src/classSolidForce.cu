#include "classSolidForce.h"

#include <fstream>
#include "functionLib.h"
#include "allocateLib.h"
#include "macroCUDA.h"

#include "defineCoefficient.h"
#include "defineReferenceVel.h"

// gpu //
#include "defineCUDA.h"
#include "solidForce_gpu.h"


// thrust //
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
// thrust //


// public //
void classSolidForce::
init_classSolidForce (
	const MPIinfo		&mpiinfo, 
	const Domain		&domain,
	const FluidProperty	*fluid
	)
{
	// mpiinfo //
	mpiinfo_ = mpiinfo;


	// domain //
	nx_ = domain.nx;
	ny_ = domain.ny;
	nz_ = domain.nz;

	nn_ = domain.nn;

	halo_ = domain.halo;

	dx_ = domain.dx;


	// lbm //
	c_ref_ = domain.c_ref;


	// solid //
	num_solid_ = 1;


	// cuda
	functionLib::set_dim3(&grid_,
			(nx_-2*halo_)/BLOCKDIM_X, 
			(ny_-2*halo_)/BLOCKDIM_Y, 
			(nz_-2*halo_)/BLOCKDIM_Z);


	functionLib::set_dim3(&block_2d_,
			BLOCKDIM_X,
		   	BLOCKDIM_Y,
		   	1);

	allocate ();
}


// memcpy //
void classSolidForce::
memcpy_solidForce_DeviceToHost ()
{
	memcpy_solidForce (
		sforce_h_,
		sforce_d_
		);
}


void classSolidForce::
memcpy_solidForce_HostToDevice ()
{
	memcpy_solidForce (
		sforce_d_,
		sforce_h_
		);
}


void classSolidForce::
memcpy_solidForce (
	      solidForce &sforcen,
   	const solidForce &sforce
	)
{
	const int	nsize = nn_;

	CUDA_SAFE_CALL( cudaMemcpy(sforcen.id_solidData,	sforce.id_solidData,	sizeof(int )*(nsize),	cudaMemcpyDefault) );

	CUDA_SAFE_CALL( cudaMemcpy(sforcen.force_inner_x,	sforce.force_inner_x,	sizeof(int )*(nsize),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(sforcen.force_inner_y,	sforce.force_inner_y,	sizeof(int )*(nsize),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(sforcen.force_inner_z,	sforce.force_inner_z,	sizeof(int )*(nsize),	cudaMemcpyDefault) );

}


void classSolidForce::
set_num_solid (int	num_solid)
{
	num_solid_ = num_solid; 
}


void classSolidForce::
set_id_solidData (
	const int			*id_solidData
	)
{
	const int	nsize = nn_;

	CUDA_SAFE_CALL( cudaMemcpy(sforce_d_.id_solidData,	id_solidData,	sizeof(int )*(nsize),	cudaMemcpyDefault) );
}


void	classSolidForce::
cal_solidForce (
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv
	)
{
	// zero //
	const int	nsize = nn_;
    thrust::device_ptr< FLOAT > thrust_force_inner_x(sforce_d_.force_inner_x);
    thrust::device_ptr< FLOAT > thrust_force_inner_y(sforce_d_.force_inner_y);
    thrust::device_ptr< FLOAT > thrust_force_inner_z(sforce_d_.force_inner_z);

	const FLOAT	value = 0.0;
	thrust::fill(thrust_force_inner_x, thrust_force_inner_x + nsize, value);
	thrust::fill(thrust_force_inner_y, thrust_force_inner_y + nsize, value);
	thrust::fill(thrust_force_inner_z, thrust_force_inner_z + nsize, value);


	// inner product //
	cal_solidForce_tensor (
		r,
		u, v, w,
		lv
		);


	// reduction //
	reduction_solidForce ();
}


void	classSolidForce::
output_solidForce (
	bool	first_cal,
	FLOAT	time
	)
{
	std::ofstream	fout;

	if (first_cal == 0) {
		fout.open("./data_solidForce.txt", std::ios::out);
	}
	else {
		fout.open("./data_solidForce.txt", std::ios::out|std::ios::app);
	}

	fout << time << "\t";
	fout << sforce_stl_[0].force_x << "\t"
		 << sforce_stl_[0].force_y << "\t"
		 << sforce_stl_[0].force_z << "\n";

	fout.close();

}


// private //
// allocate //
void	classSolidForce::
allocate (
	)
{
	// solidForce_STL //
	allocateLib::new_host   (&sforce_stl_,		num_solid_);

	// solidForce mesh //
	allocate_host   ();
	allocate_device ();

	memcpy_solidForce_HostToDevice ();
}


void classSolidForce::
allocate_host ()
{
	const int	nsize = nn_;

	// solidForce //
	allocateLib::new_host   (&sforce_h_.id_solidData,	nsize);
	allocateLib::new_host   (&sforce_h_.force_inner_x,	nsize);
	allocateLib::new_host   (&sforce_h_.force_inner_y,	nsize);
	allocateLib::new_host   (&sforce_h_.force_inner_z,	nsize);

	// fill //
	functionLib::fillArray (sforce_h_.id_solidData,     -1,	nsize);
//	functionLib::fillArray (sforce_h_.id_solidData,      0,	nsize);
	functionLib::fillArray (sforce_h_.force_inner_x,   0.0,	nsize);
	functionLib::fillArray (sforce_h_.force_inner_y,   0.0,	nsize);
	functionLib::fillArray (sforce_h_.force_inner_z,   0.0,	nsize);
}


void classSolidForce::
allocate_device ()
{
	const int	nsize = nn_;

	allocateLib::new_device (&sforce_d_.id_solidData,	nsize);
	allocateLib::new_device (&sforce_d_.force_inner_x,	nsize);
	allocateLib::new_device (&sforce_d_.force_inner_y,	nsize);
	allocateLib::new_device (&sforce_d_.force_inner_z,	nsize);
}


// calculation //
void classSolidForce::
reduction_solidForce ()
{
	filter_solidForce ();

	thrust_reduction ();
}


void classSolidForce::
filter_solidForce ()
{
	// output //
	int*	id_solidData  = sforce_d_.id_solidData;
	FLOAT*	force_inner_x = sforce_d_.force_inner_x;
	FLOAT*	force_inner_y = sforce_d_.force_inner_y;
	FLOAT*	force_inner_z = sforce_d_.force_inner_z;


	// kernel //
	cuda_filter_solidForce <<< grid_, block_2d_ >>>  (
	      force_inner_x,
	      force_inner_y,
	      force_inner_z,
		  id_solidData,
		  nx_, ny_, nz_,
		  halo_
		  );

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void classSolidForce::
thrust_reduction ()
{
	const int	nsize = nn_;

	// output //
    thrust::device_ptr< FLOAT > thrust_force_inner_x(sforce_d_.force_inner_x);
    thrust::device_ptr< FLOAT > thrust_force_inner_y(sforce_d_.force_inner_y);
    thrust::device_ptr< FLOAT > thrust_force_inner_z(sforce_d_.force_inner_z);


	// kernel //
	FLOAT	force_x = thrust::reduce(thrust_force_inner_x, thrust_force_inner_x + nsize);
	FLOAT	force_y = thrust::reduce(thrust_force_inner_y, thrust_force_inner_y + nsize);
	FLOAT	force_z = thrust::reduce(thrust_force_inner_z, thrust_force_inner_z + nsize);


	// mpi //
	FLOAT	force_x_g = 0.0;
	FLOAT	force_y_g = 0.0;
	FLOAT	force_z_g = 0.0;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&force_x,		&force_x_g,		1, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&force_y,		&force_y_g,		1, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&force_z,		&force_z_g,		1, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	// force //
	sforce_stl_[0].force_x = force_x_g;
	sforce_stl_[0].force_y = force_y_g;
	sforce_stl_[0].force_z = force_z_g;
}


void	classSolidForce::
cal_solidForce_tensor (
	const FLOAT	*r,
	const FLOAT	*u,
	const FLOAT	*v,
	const FLOAT	*w,
	const FLOAT	*lv
	)
{
	// output //
	FLOAT	*force_inner_x = sforce_d_.force_inner_x;
	FLOAT	*force_inner_y = sforce_d_.force_inner_y;
	FLOAT	*force_inner_z = sforce_d_.force_inner_z;

	// input //
	const FLOAT	coef  = coefficient::SOLID_FORCE_WIDTH;
	const FLOAT	delta = dx_*coef;
	const FLOAT	vis   = coefficient::VIS_AIR;

	const FLOAT	rho0 = coefficient::DENSITY_AIR;
	const FLOAT	cvel = c_ref_;


	// kernel //
//	cuda_get_solidForce_tensor    <<< grid_, block_2d_ >>>  (
//	cuda_get_solidForce_tensor_bb <<< grid_, block_2d_ >>>  (
	cuda_get_solidForce_tensor_bounce_back <<< grid_, block_2d_ >>>  (
	      force_inner_x,
	      force_inner_y,
	      force_inner_z,
		  r,
		  u, v, w,
		  lv,
		  delta,
		  vis,
		  rho0,
		  cvel,
		  dx_,
		  nx_, ny_, nz_,
		  halo_
		  );
	
	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// classSolidForce.cu //
