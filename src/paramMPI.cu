#include "paramMPI.h"

#include <iostream>
#include <cuda.h>
#include <mpi.h>
#include "option_parser.h"
#include "defineCUDA.h"
#include "indexLib.h"
#include "functionLib.h"
#include "allocateLib.h"
#include "defineLBM.h"


enum MPI_Tag_List { tag_c, tag_xm, tag_xp, tag_ym, tag_yp, tag_zm, tag_zp };


void	paramMPI::
set (
	char	*program_name,
	int		argc,
	char	*argv[])
{
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);

	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// ncpu, rank //
	int		ncpu, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const int	ncpu_x = parser.nummpi(0);
	const int	ncpu_y = parser.nummpi(1);
	const int	ncpu_z = parser.nummpi(2);

	const int	rank_x = indexLib::get_index_x (rank, ncpu_x, ncpu_y, ncpu_z);
	const int	rank_y = indexLib::get_index_y (rank, ncpu_x, ncpu_y, ncpu_z);
	const int	rank_z = indexLib::get_index_z (rank, ncpu_x, ncpu_y, ncpu_z);


	// periodic boundary //
	const int	rank_xm = (rank_x - 1 + ncpu_x)%ncpu_x;	const int	rank_xp = (rank_x + 1)%ncpu_x;
	const int	rank_ym = (rank_y - 1 + ncpu_y)%ncpu_y;	const int	rank_yp = (rank_y + 1)%ncpu_y;
	const int	rank_zm = (rank_z - 1 + ncpu_z)%ncpu_z;	const int	rank_zp = (rank_z + 1)%ncpu_z;


	// index //
	const int	id_rank    = indexLib::get_index (rank_x,  rank_y,  rank_z,  ncpu_x, ncpu_y, ncpu_z);
	const int	id_rank_xm = indexLib::get_index (rank_xm, rank_y,  rank_z,  ncpu_x, ncpu_y, ncpu_z);
	const int	id_rank_xp = indexLib::get_index (rank_xp, rank_y,  rank_z,  ncpu_x, ncpu_y, ncpu_z);
	const int	id_rank_ym = indexLib::get_index (rank_x,  rank_ym, rank_z,  ncpu_x, ncpu_y, ncpu_z);
	const int	id_rank_yp = indexLib::get_index (rank_x,  rank_yp, rank_z,  ncpu_x, ncpu_y, ncpu_z);
	const int	id_rank_zm = indexLib::get_index (rank_x,  rank_y,  rank_zm, ncpu_x, ncpu_y, ncpu_z);
	const int	id_rank_zp = indexLib::get_index (rank_x,  rank_y,  rank_zp, ncpu_x, ncpu_y, ncpu_z);

	// ncpu div //
	ncpu_div_x_ = parser.ncpu_div(0);
	ncpu_div_y_ = parser.ncpu_div(1);
	ncpu_div_z_ = parser.ncpu_div(2);
	ncpu_div_p_ = parser.ncpu_div(3);





	// initialize //
	mpiinfo_.ncpu   = ncpu;
	mpiinfo_.ncpu_x = ncpu_x;
	mpiinfo_.ncpu_y = ncpu_y;
	mpiinfo_.ncpu_z = ncpu_z;

	mpiinfo_.rank   = rank;
	mpiinfo_.rank_x = rank_x;
	mpiinfo_.rank_y = rank_y;
	mpiinfo_.rank_z = rank_z;

	mpiinfo_.rank_xm = rank_xm;	mpiinfo_.rank_xp = rank_xp;
	mpiinfo_.rank_ym = rank_ym;	mpiinfo_.rank_yp = rank_yp;
	mpiinfo_.rank_zm = rank_zm;	mpiinfo_.rank_zp = rank_zp;

	mpiinfo_.id_rank    = id_rank;
	mpiinfo_.id_rank_xm = id_rank_xm;
	mpiinfo_.id_rank_xp = id_rank_xp;
	mpiinfo_.id_rank_ym = id_rank_ym;
	mpiinfo_.id_rank_yp = id_rank_yp;
	mpiinfo_.id_rank_zm = id_rank_zm;
	mpiinfo_.id_rank_zp = id_rank_zp;

	MPI_Tag_List	tag[6] = { tag_xm, tag_xp, tag_ym, tag_yp, tag_zm, tag_zp };

	mpiinfo_.tag_xm = tag[0];	mpiinfo_.tag_xp = tag[1];
	mpiinfo_.tag_ym = tag[2];	mpiinfo_.tag_yp = tag[3];
	mpiinfo_.tag_zm = tag[4];	mpiinfo_.tag_zp = tag[5];
}


void	paramMPI::
set_buffer (
	const Domain	&domain)
{
	nx_ = domain.nx;
	ny_ = domain.ny;
	nz_ = domain.nz;

	halo_ = domain.halo;

	const int	num_halo      = halo_;
//	const int	num_variables = NUM_DIRECTION_VEL + 4;
//	const int	num_variables = NUM_DIRECTION_VEL + 5;
	const int	num_variables = NUM_DIRECTION_VEL + 6;

	// halo, variable //
	buff_h_.num_halo = num_halo;
	buff_d_.num_halo = num_halo;

	buff_h_.num_variables = num_variables;
	buff_d_.num_variables = num_variables;


	// host memory //
	allocate_host   (num_halo, num_variables);

	// device memory //
	allocate_device (num_halo, num_variables);


	// host to device //
	memcpy_buffer (num_halo, num_variables);
}


// check & cout //
void	paramMPI::
check_MPI ()
{
	int		flg = 0;
	if (   (mpiinfo_.ncpu_x%ncpu_div_x_) != 0 
		|| (mpiinfo_.ncpu_y%ncpu_div_y_) != 0 
		|| (mpiinfo_.ncpu_z%ncpu_div_z_) != 0) {
		std::cout << "error : ncpu div x, ncpu div y, ncpu div z\n";
		flg = 1;
	}
	if ( (mpiinfo_.ncpu%ncpu_div_p_) != 0 ) {
		std::cout << "error : ncpu div p\n";
		flg = 1;
	}

	if ((mpiinfo_.ncpu_x*mpiinfo_.ncpu_y*mpiinfo_.ncpu_z != mpiinfo_.ncpu))
	{
		std::cout << "causion !! ncpu is not good\n";
		flg = 1;
	}

	if (flg == 1) { exit(-2); }
}


void	paramMPI::
cout_MPI ()
{
	std::cout << mpiinfo_.rank 
			  << " : ncpu (x, y, z) = " 
			  << mpiinfo_.ncpu << " ( " <<  mpiinfo_.ncpu_x << ", " << mpiinfo_.ncpu_y << ", " << mpiinfo_.ncpu_z << " )" << std::endl;
}


// mpi communication //
void	paramMPI::
mpi_cuda_x (
	FLOAT	**fp,
	int		phy_num)
{
//	const int	nsize = phy_num * ny_*nz_;
	const int	nsize = phy_num * ny_*nz_ * halo_;

	// download //
	buff_download_x (
		fp,
		phy_num,
		nsize);
	cudaThreadSynchronize(); // cudaThreadSynchronize //


	// MPI send recv //
	mpi_Isend_Irecv_x (nsize);


	// upload //
	buff_upload_x (
		fp,
		phy_num,
		nsize);

	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	paramMPI::
mpi_cuda_y (
	FLOAT	**fp,
	int		phy_num)
{
//	const int	nsize = phy_num * nx_*nz_;
	const int	nsize = phy_num * nx_*nz_ * halo_;

	// download //
	buff_download_y (
		fp,
		phy_num,
		nsize);
	cudaThreadSynchronize(); // cudaThreadSynchronize //


	// MPI send recv //
	mpi_Isend_Irecv_y (nsize);


	// upload //
	buff_upload_y (
		fp,
		phy_num,
		nsize);

	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	paramMPI::
mpi_cuda_z (
	FLOAT	**fp,
	int		phy_num)
{
//	const int	nsize = phy_num * nx_*ny_;
	const int	nsize = phy_num * nx_*ny_ * halo_;

	// download //
	buff_download_z (
		fp,
		phy_num,
		nsize);
	cudaThreadSynchronize(); // cudaThreadSynchronize //


	// MPI send recv //
	mpi_Isend_Irecv_z (nsize);


	// upload //
	buff_upload_z (
		fp,
		phy_num,
		nsize);

	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	paramMPI::
mpi_cuda_xyz (
	FLOAT	**fp,
	int		phy_num)
{
	mpi_cuda_x (fp, phy_num);
	mpi_cuda_y (fp, phy_num);
	mpi_cuda_z (fp, phy_num);
}


// private //
// memory //
void	paramMPI::
allocate_host (
	int num_halo,
	int num_variables)
{
	const int	nsize_x = ny_*nz_ * num_halo * num_variables;
	const int	nsize_y = nx_*nz_ * num_halo * num_variables;
	const int	nsize_z = nx_*ny_ * num_halo * num_variables;

	allocateLib::new_pinned (&buff_h_.buff_s_xm,  nsize_x);
	allocateLib::new_pinned (&buff_h_.buff_s_xp,  nsize_x);
	allocateLib::new_pinned (&buff_h_.buff_r_xm,  nsize_x);
	allocateLib::new_pinned (&buff_h_.buff_r_xp,  nsize_x);

	allocateLib::new_pinned (&buff_h_.buff_s_ym,  nsize_y);
	allocateLib::new_pinned (&buff_h_.buff_s_yp,  nsize_y);
	allocateLib::new_pinned (&buff_h_.buff_r_ym,  nsize_y);
	allocateLib::new_pinned (&buff_h_.buff_r_yp,  nsize_y);

	allocateLib::new_pinned (&buff_h_.buff_s_zm,  nsize_z);
	allocateLib::new_pinned (&buff_h_.buff_s_zp,  nsize_z);
	allocateLib::new_pinned (&buff_h_.buff_r_zm,  nsize_z);
	allocateLib::new_pinned (&buff_h_.buff_r_zp,  nsize_z);


	functionLib::fillArray (buff_h_.buff_s_xm,  0.0, nsize_x);
	functionLib::fillArray (buff_h_.buff_s_xp,  0.0, nsize_x);
	functionLib::fillArray (buff_h_.buff_r_xm,  0.0, nsize_x);
	functionLib::fillArray (buff_h_.buff_r_xp,  0.0, nsize_x);

	functionLib::fillArray (buff_h_.buff_s_ym,  0.0, nsize_y);
	functionLib::fillArray (buff_h_.buff_s_yp,  0.0, nsize_y);
	functionLib::fillArray (buff_h_.buff_r_ym,  0.0, nsize_y);
	functionLib::fillArray (buff_h_.buff_r_yp,  0.0, nsize_y);

	functionLib::fillArray (buff_h_.buff_s_zm,  0.0, nsize_z);
	functionLib::fillArray (buff_h_.buff_s_zp,  0.0, nsize_z);
	functionLib::fillArray (buff_h_.buff_r_zm,  0.0, nsize_z);
	functionLib::fillArray (buff_h_.buff_r_zp,  0.0, nsize_z);
}


void	paramMPI::
allocate_device  (
	int num_halo,
	int num_variables)
{
	const int	nsize_x = ny_*nz_ * num_halo * num_variables;
	const int	nsize_y = nx_*nz_ * num_halo * num_variables;
	const int	nsize_z = nx_*ny_ * num_halo * num_variables;

	allocateLib::new_device (&buff_d_.buff_s_xm,  nsize_x);
	allocateLib::new_device (&buff_d_.buff_s_xp,  nsize_x);
	allocateLib::new_device (&buff_d_.buff_r_xm,  nsize_x);
	allocateLib::new_device (&buff_d_.buff_r_xp,  nsize_x);

	allocateLib::new_device (&buff_d_.buff_s_ym,  nsize_y);
	allocateLib::new_device (&buff_d_.buff_s_yp,  nsize_y);
	allocateLib::new_device (&buff_d_.buff_r_ym,  nsize_y);
	allocateLib::new_device (&buff_d_.buff_r_yp,  nsize_y);

	allocateLib::new_device (&buff_d_.buff_s_zm,  nsize_z);
	allocateLib::new_device (&buff_d_.buff_s_zp,  nsize_z);
	allocateLib::new_device (&buff_d_.buff_r_zm,  nsize_z);
	allocateLib::new_device (&buff_d_.buff_r_zp,  nsize_z);
}


void	paramMPI::
memcpy_buffer (
	int num_halo,
	int num_variables)
{
	const int	nsize_x = ny_*nz_ * num_halo * num_variables;
	const int	nsize_y = nx_*nz_ * num_halo * num_variables;
	const int	nsize_z = nx_*ny_ * num_halo * num_variables;

	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_s_xp,  buff_h_.buff_s_xp,  sizeof(FLOAT)*(nsize_x), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_r_xp,  buff_h_.buff_r_xp,  sizeof(FLOAT)*(nsize_x), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_s_xm,  buff_h_.buff_s_xm,  sizeof(FLOAT)*(nsize_x), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_r_xm,  buff_h_.buff_r_xm,  sizeof(FLOAT)*(nsize_x), cudaMemcpyDefault) );

	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_s_yp,  buff_h_.buff_s_yp,  sizeof(FLOAT)*(nsize_y), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_r_yp,  buff_h_.buff_r_yp,  sizeof(FLOAT)*(nsize_y), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_s_ym,  buff_h_.buff_s_ym,  sizeof(FLOAT)*(nsize_y), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_r_ym,  buff_h_.buff_r_ym,  sizeof(FLOAT)*(nsize_y), cudaMemcpyDefault) );

	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_s_zp,  buff_h_.buff_s_zp,  sizeof(FLOAT)*(nsize_z), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_r_zp,  buff_h_.buff_r_zp,  sizeof(FLOAT)*(nsize_z), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_s_zm,  buff_h_.buff_s_zm,  sizeof(FLOAT)*(nsize_z), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(buff_d_.buff_r_zm,  buff_h_.buff_r_zm,  sizeof(FLOAT)*(nsize_z), cudaMemcpyDefault) );
}


// mpi communication //
void	paramMPI::
mpi_Isend_Irecv (
	FLOAT	*buff_rm,
	FLOAT	*buff_rp,
	FLOAT	*buff_sm,
	FLOAT	*buff_sp,
	int		id_rank_m,
	int		id_rank_p,
	int		tag1,
	int		tag2,
	int		nsize)
{
	MPI_Status	stat;
	MPI_Request	requ_s[2],
				requ_r[2];

	if (tag1 == tag2)	{ std::cout << "error : mpi tag\n"; }

	// Irecv & Isend
	MPI_Irecv(buff_rm, nsize, MFLOAT, id_rank_m, tag1, MPI_COMM_WORLD, &requ_r[0]);
	MPI_Irecv(buff_rp, nsize, MFLOAT, id_rank_p, tag2, MPI_COMM_WORLD, &requ_r[1]);

	MPI_Isend(buff_sm, nsize, MFLOAT, id_rank_m, tag2, MPI_COMM_WORLD, &requ_s[0]);
	MPI_Isend(buff_sp, nsize, MFLOAT, id_rank_p, tag1, MPI_COMM_WORLD, &requ_s[1]);

	MPI_Wait(&requ_s[0], &stat);
	MPI_Wait(&requ_s[1], &stat);
	MPI_Wait(&requ_r[0], &stat);
	MPI_Wait(&requ_r[1], &stat);
}


void	paramMPI::
mpi_Isend_Irecv_x (
	int		nsize)
{
	mpi_Isend_Irecv (
		buff_h_.buff_r_xm,
		buff_h_.buff_r_xp,
		buff_h_.buff_s_xm,
		buff_h_.buff_s_xp,
		mpiinfo_.id_rank_xm,
		mpiinfo_.id_rank_xp,
		mpiinfo_.tag_xm,
		mpiinfo_.tag_xp,
		nsize);
}


void	paramMPI::
mpi_Isend_Irecv_y (
	int		nsize)
{
	mpi_Isend_Irecv (
		buff_h_.buff_r_ym,
		buff_h_.buff_r_yp,
		buff_h_.buff_s_ym,
		buff_h_.buff_s_yp,
		mpiinfo_.id_rank_ym,
		mpiinfo_.id_rank_yp,
		mpiinfo_.tag_ym,
		mpiinfo_.tag_yp,
		nsize);
}


void	paramMPI::
mpi_Isend_Irecv_z (
	int		nsize)
{
	mpi_Isend_Irecv (
		buff_h_.buff_r_zm,
		buff_h_.buff_r_zp,
		buff_h_.buff_s_zm,
		buff_h_.buff_s_zp,
		mpiinfo_.id_rank_zm,
		mpiinfo_.id_rank_zp,
		mpiinfo_.tag_zm,
		mpiinfo_.tag_zp,
		nsize);
}


// download //
void	paramMPI::
buff_download_x (
   	FLOAT	**fp,
	int		phy_num,
	int		nsize)
{
	// buffer //
	FLOAT	*buff_dm = buff_d_.buff_s_xm; // device //
   	FLOAT	*buff_dp = buff_d_.buff_s_xp;
   	FLOAT	*buff_hm = buff_h_.buff_s_xm; // host //
   	FLOAT	*buff_hp = buff_h_.buff_s_xp;

	const int	nx = nx_,
		  		ny = ny_,
				nz = nz_;
	const int	offset = ny*nz * halo_;

	const int	ny_cast = functionLib::cast_value_large(ny, BLOCKDIM_Y);
	const int	nz_cast = functionLib::cast_value_large(nz, BLOCKDIM_Z);

	dim3	grid_yz  (ny_cast/BLOCKDIM_Y, nz_cast/BLOCKDIM_Z, 1),
			block_yz (BLOCKDIM_Y, BLOCKDIM_Z, 1);


	// gpu to gpu_buff (send) //
	for (int i=0; i<phy_num; i++) {
		if (halo_ == 1) {
			copy_global_to_buff_x1 <<< grid_yz, block_yz >>> (
				fp[i],
			   	&buff_dm[offset*i],
				&buff_dp[offset*i],
			   	nx, ny, nz);
		}
		else if (halo_ == 2) {
			copy_global_to_buff_x2 <<< grid_yz, block_yz >>> (
				fp[i],
			   	&buff_dm[offset*i],
				&buff_dp[offset*i],
			   	nx, ny, nz);
		}
		else if (halo_ == 3) {
			copy_global_to_buff_x3 <<< grid_yz, block_yz >>> (
				fp[i],
			   	&buff_dm[offset*i],
				&buff_dp[offset*i],
			   	nx, ny, nz);
		}
	}
	cudaThreadSynchronize(); // cudaThreadSynchronize //


	// gpu_buff to cpu buff (send)
	cudaMemcpy(buff_hm, buff_dm,  nsize*sizeof(FLOAT), cudaMemcpyDeviceToHost);
	cudaMemcpy(buff_hp, buff_dp,  nsize*sizeof(FLOAT), cudaMemcpyDeviceToHost);
}


void	paramMPI::
buff_download_y (
   	FLOAT	**fp,
	int		phy_num,
	int		nsize)
{
	// buffer //
	FLOAT	*buff_dm = buff_d_.buff_s_ym; // device //
   	FLOAT	*buff_dp = buff_d_.buff_s_yp;
   	FLOAT	*buff_hm = buff_h_.buff_s_ym; // host //
   	FLOAT	*buff_hp = buff_h_.buff_s_yp;

	const int	nx = nx_,
		  		ny = ny_,
				nz = nz_;
	const int	offset = nx*nz * halo_;

	const int	nx_cast = functionLib::cast_value_large(nx, BLOCKDIM_X);
	const int	nz_cast = functionLib::cast_value_large(nz, BLOCKDIM_Z);

	dim3	grid_zx  (nx_cast/BLOCKDIM_X, nz_cast/BLOCKDIM_Z, 1),
			block_zx (BLOCKDIM_X, BLOCKDIM_Z, 1);


	// gpu to gpu_buff (send)
	for (int i=0; i<phy_num; i++) {
		if (halo_ == 1) {
			copy_global_to_buff_y1 <<< grid_zx, block_zx >>> (
				fp[i],
			   	&buff_dm[offset*i],
				&buff_dp[offset*i],
			   	nx, ny, nz);
		}
		else if (halo_ == 2) {
			copy_global_to_buff_y2 <<< grid_zx, block_zx >>> (
				fp[i],
			   	&buff_dm[offset*i],
				&buff_dp[offset*i],
			   	nx, ny, nz);
		}
		else if (halo_ == 3) {
			copy_global_to_buff_y3 <<< grid_zx, block_zx >>> (
				fp[i],
			   	&buff_dm[offset*i],
				&buff_dp[offset*i],
			   	nx, ny, nz);
		}
	}
	cudaThreadSynchronize(); // cudaThreadSynchronize //


	// gpu_buff to cpu buff (send)
	cudaMemcpy(buff_hm, buff_dm,  nsize*sizeof(FLOAT), cudaMemcpyDeviceToHost);
	cudaMemcpy(buff_hp, buff_dp,  nsize*sizeof(FLOAT), cudaMemcpyDeviceToHost);
}


void	paramMPI::
buff_download_z (
   	FLOAT	**fp,
	int		phy_num,
	int		nsize)
{
	// buffer //
	FLOAT	*buff_dm = buff_d_.buff_s_zm; // device //
   	FLOAT	*buff_dp = buff_d_.buff_s_zp;
   	FLOAT	*buff_hm = buff_h_.buff_s_zm; // host //
   	FLOAT	*buff_hp = buff_h_.buff_s_zp;

	const int	nx = nx_,
		  		ny = ny_,
				nz = nz_;
	const int	offset = nx*ny * halo_;

	const int	nx_cast = functionLib::cast_value_large(nx, BLOCKDIM_X);
	const int	ny_cast = functionLib::cast_value_large(ny, BLOCKDIM_Y);

	dim3	grid_xy  (nx_cast/BLOCKDIM_X, ny_cast/BLOCKDIM_Y, 1),
			block_xy (BLOCKDIM_X, BLOCKDIM_Y, 1);


	// gpu to gpu_buff (send)
	for (int i=0; i<phy_num; i++) {
		if (halo_ == 1) {
			copy_global_to_buff_z1 <<< grid_xy, block_xy >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 2) {
			copy_global_to_buff_z2 <<< grid_xy, block_xy >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 3) {
			copy_global_to_buff_z3 <<< grid_xy, block_xy >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
	}
	cudaThreadSynchronize(); // cudaThreadSynchronize *****


	// gpu_buff to cpu buff (send)
	cudaMemcpy(buff_hm, buff_dm,  nsize*sizeof(FLOAT), cudaMemcpyDeviceToHost);
	cudaMemcpy(buff_hp, buff_dp,  nsize*sizeof(FLOAT), cudaMemcpyDeviceToHost);
}


// upload //
void	paramMPI::
buff_upload_x (
   	FLOAT	**fp,
	int		phy_num,
	int		nsize)
{
	// buffer //
	FLOAT	*buff_dm = buff_d_.buff_r_xm; // device //
   	FLOAT	*buff_dp = buff_d_.buff_r_xp;
   	FLOAT	*buff_hm = buff_h_.buff_r_xm; // host //
   	FLOAT	*buff_hp = buff_h_.buff_r_xp;

	const int	nx = nx_,
		  		ny = ny_,
				nz = nz_;
	const int	offset = ny*nz * halo_;


	const int	ny_cast = functionLib::cast_value_large(ny, BLOCKDIM_Y);
	const int	nz_cast = functionLib::cast_value_large(nz, BLOCKDIM_Z);

	dim3	grid_yz  (ny_cast/BLOCKDIM_Y, nz_cast/BLOCKDIM_Z, 1),
			block_yz (BLOCKDIM_Y, BLOCKDIM_Z, 1);


	// cpu_buff to gpu buff (receive)
	cudaMemcpy(buff_dm, buff_hm,  nsize*sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(buff_dp, buff_hp,  nsize*sizeof(FLOAT), cudaMemcpyHostToDevice);


	// gpu buff to gpu (receive)
	for (int i=0; i<phy_num; i++) {
		if (halo_ == 1) {
			copy_buff_to_global_x1 <<< grid_yz, block_yz >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 2) {
			copy_buff_to_global_x2 <<< grid_yz, block_yz >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 3) {
			copy_buff_to_global_x3 <<< grid_yz, block_yz >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
	}
}


void	paramMPI::
buff_upload_y (
   	FLOAT	**fp,
	int		phy_num,
	int		nsize)
{
	// buffer //
	FLOAT	*buff_dm = buff_d_.buff_r_ym; // device //
   	FLOAT	*buff_dp = buff_d_.buff_r_yp;
   	FLOAT	*buff_hm = buff_h_.buff_r_ym; // host //
   	FLOAT	*buff_hp = buff_h_.buff_r_yp;

	const int	nx = nx_,
		  		ny = ny_,
				nz = nz_;
	const int	offset = nx*nz * halo_;


	const int	nx_cast = functionLib::cast_value_large(nx, BLOCKDIM_X);
	const int	nz_cast = functionLib::cast_value_large(nz, BLOCKDIM_Z);

	dim3	grid_zx (nx_cast/BLOCKDIM_X, nz_cast/BLOCKDIM_Z, 1),
			block_x (BLOCKDIM_X, BLOCKDIM_Z, 1);


	// cpu_buff to gpu buff (receive)
	cudaMemcpy(buff_dm, buff_hm,  nsize*sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(buff_dp, buff_hp,  nsize*sizeof(FLOAT), cudaMemcpyHostToDevice);


	// gpu buff to gpu (receive)
	for (int i=0; i<phy_num; i++) {
		if (halo_ == 1) {
			copy_buff_to_global_y1 <<< grid_zx, block_x >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 2) {
			copy_buff_to_global_y2 <<< grid_zx, block_x >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 3) {
			copy_buff_to_global_y3 <<< grid_zx, block_x >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
	}
}


void	paramMPI::
buff_upload_z (
   	FLOAT	**fp,
	int		phy_num,
	int		nsize)
{
	// buffer //
	FLOAT	*buff_dm = buff_d_.buff_r_zm; // device //
   	FLOAT	*buff_dp = buff_d_.buff_r_zp;
   	FLOAT	*buff_hm = buff_h_.buff_r_zm; // host //
   	FLOAT	*buff_hp = buff_h_.buff_r_zp;

	const int	nx = nx_,
		  		ny = ny_,
				nz = nz_;
	const int	offset = nx*ny * halo_;


	const int	nx_cast = functionLib::cast_value_large(nx, BLOCKDIM_X);
	const int	ny_cast = functionLib::cast_value_large(ny, BLOCKDIM_Y);

	dim3	grid_xy  (nx_cast/BLOCKDIM_X, ny_cast/BLOCKDIM_Y, 1),
			block_xy (BLOCKDIM_X, BLOCKDIM_Y, 1);


	// cpu_buff to gpu buff (receive)
	cudaMemcpy(buff_dm, buff_hm,  nsize*sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(buff_dp, buff_hp,  nsize*sizeof(FLOAT), cudaMemcpyHostToDevice);


	// gpu buff to gpu (receive)
	for (int i=0; i<phy_num; i++) {
		if (halo_ == 1) {
			copy_buff_to_global_z1 <<< grid_xy, block_xy >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 2) {
			copy_buff_to_global_z2 <<< grid_xy, block_xy >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
		else if (halo_ == 3) {
			copy_buff_to_global_z3 <<< grid_xy, block_xy >>> (
				fp[i],
				&buff_dm[offset*i],
				&buff_dp[offset*i],
				nx, ny, nz);
		}
	}
}


// Host <--> Device //
// device to host buffer //
// halo = 1 //
__global__ void
copy_global_to_buff_x1 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = nx*id_y + nxy*id_z,
		  		id_b =    id_y + ny *id_z;

	if (id_y >= ny || id_z >= nz)	{ return; }

	// x (send)
	buff_dm[id_b] = f_d[id + 1];
	buff_dp[id_b] = f_d[id + (nx-2)];
}


__global__ void
copy_global_to_buff_y1 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = id_x + nxy*id_z,
		  		id_b = id_x + nx *id_z;

	if (id_x >= nx || id_z >= nz)	{ return; }

	// y (send)
	buff_dm[id_b] = f_d[id + nx*1];
	buff_dp[id_b] = f_d[id + nx*(ny-2)];
}


__global__ void
copy_global_to_buff_z1 (
	const FLOAT *f_d,
	      FLOAT *buff_dm,
	      FLOAT *buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_y = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id = id_x + nx*id_y;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// z (send) //
	buff_dm[id] =  f_d[id + nxy*1];
	buff_dp[id] =  f_d[id + nxy*(nz-2)];
}


// halo = 2 //
__global__ void
copy_global_to_buff_x2 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = nx*id_y + nxy*id_z,
		  		id_b =    id_y + ny *id_z;

	if (id_y >= ny || id_z >= nz)	{ return; }
	const int	nyz = ny*nz;

	// x (send) //
	buff_dm[id_b        ] = f_d[id + 2];
	buff_dm[id_b + nyz  ] = f_d[id + 3];

	buff_dp[id_b        ] = f_d[id + (nx-4)];
	buff_dp[id_b + nyz  ] = f_d[id + (nx-3)];
}


__global__ void
copy_global_to_buff_y2 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = id_x + nxy*id_z,
		  		id_b = id_x + nx *id_z;

	if (id_x >= nx || id_z >= nz)	{ return; }
	const int	nxz = nx*nz;

	// y (send) //
	buff_dm[id_b        ] = f_d[id + nx*2];
	buff_dm[id_b + nxz  ] = f_d[id + nx*3];

	buff_dp[id_b        ] = f_d[id + nx*(ny-4)];
	buff_dp[id_b + nxz  ] = f_d[id + nx*(ny-3)];
}


__global__ void
copy_global_to_buff_z2 (
	const FLOAT *f_d,
	      FLOAT *buff_dm,
	      FLOAT *buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_y = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id = id_x + nx*id_y;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// z (send) //
	buff_dm[id        ] =  f_d[id + nxy*2];
	buff_dm[id + nxy  ] =  f_d[id + nxy*3];

	buff_dp[id        ] =  f_d[id + nxy*(nz-4)];
	buff_dp[id + nxy  ] =  f_d[id + nxy*(nz-3)];
}


// halo = 3 //
__global__ void
copy_global_to_buff_x3 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = nx*id_y + nxy*id_z,
		  		id_b =    id_y + ny *id_z;

	if (id_y >= ny || id_z >= nz)	{ return; }
	const int	nyz = ny*nz;

	// x (send) //
	buff_dm[id_b        ] = f_d[id + 3];
	buff_dm[id_b + nyz  ] = f_d[id + 4];
	buff_dm[id_b + nyz*2] = f_d[id + 5];

	buff_dp[id_b        ] = f_d[id + (nx-6)];
	buff_dp[id_b + nyz  ] = f_d[id + (nx-5)];
	buff_dp[id_b + nyz*2] = f_d[id + (nx-4)];
}


__global__ void
copy_global_to_buff_y3 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = id_x + nxy*id_z,
		  		id_b = id_x + nx *id_z;

	if (id_x >= nx || id_z >= nz)	{ return; }
	const int	nxz = nx*nz;

	// y (send) //
	buff_dm[id_b        ] = f_d[id + nx*3];
	buff_dm[id_b + nxz  ] = f_d[id + nx*4];
	buff_dm[id_b + nxz*2] = f_d[id + nx*5];

	buff_dp[id_b        ] = f_d[id + nx*(ny-6)];
	buff_dp[id_b + nxz  ] = f_d[id + nx*(ny-5)];
	buff_dp[id_b + nxz*2] = f_d[id + nx*(ny-4)];
}


__global__ void
copy_global_to_buff_z3 (
	const FLOAT *f_d,
	      FLOAT *buff_dm,
	      FLOAT *buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_y = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id = id_x + nx*id_y;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// z (send) //
	buff_dm[id        ] =  f_d[id + nxy*3];
	buff_dm[id + nxy  ] =  f_d[id + nxy*4];
	buff_dm[id + nxy*2] =  f_d[id + nxy*5];

	buff_dp[id        ] =  f_d[id + nxy*(nz-6)];
	buff_dp[id + nxy  ] =  f_d[id + nxy*(nz-5)];
	buff_dp[id + nxy*2] =  f_d[id + nxy*(nz-4)];
}


// host buffer to device //
// halo = 1 //
__global__ void
copy_buff_to_global_x1 (
	      FLOAT	*f_d,
   	const FLOAT	*buff_dm,
	const FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = nx*id_y + nxy*id_z,
		  		id_b =    id_y + ny *id_z;

	if (id_y >= ny || id_z >= nz)	{ return; }

	// x (receive)
	f_d[id + 0]      = buff_dm[id_b];
	f_d[id + (nx-1)] = buff_dp[id_b];
}


__global__ void
copy_buff_to_global_y1 (
	      FLOAT	*f_d,
	const FLOAT	*buff_dm, 
	const FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = id_x + nxy*id_z,
		  		id_b = id_x + nx *id_z;

	if (id_x >= nx || id_z >= nz)	{ return; }

	// y (receive) //
	f_d[id + nx*0]      = buff_dm[id_b];
	f_d[id + nx*(ny-1)] = buff_dp[id_b];
}


__global__ void 
copy_buff_to_global_z1 (
	      FLOAT *f_d,
	const FLOAT *buff_dm,
	const FLOAT *buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x =     threadIdx.x + blockDim.x*blockIdx.x,
		  		id_y =     threadIdx.y + blockDim.y*blockIdx.y;
	const int	id = id_x + nx*id_y;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// z (receive) //
	f_d[id + nxy*0]      = buff_dm[id];	
	f_d[id + nxy*(nz-1)] = buff_dp[id];
}


// halo = 2 //
__global__ void
copy_buff_to_global_x2 (
	      FLOAT	*f_d,
   	const FLOAT	*buff_dm,
	const FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = nx*id_y + nxy*id_z,
		  		id_b =    id_y + ny *id_z;

	if (id_y >= ny || id_z >= nz)	{ return; }
	const int	nyz = ny*nz;

	// x (receive)
	f_d[id + 0]      = buff_dm[id_b        ];
	f_d[id + 1]      = buff_dm[id_b + nyz  ];

	f_d[id + (nx-2)] = buff_dp[id_b        ];
	f_d[id + (nx-1)] = buff_dp[id_b + nyz  ];
}


__global__ void
copy_buff_to_global_y2 (
	      FLOAT	*f_d,
	const FLOAT	*buff_dm, 
	const FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = id_x + nxy*id_z,
		  		id_b = id_x + nx *id_z;

	if (id_x >= nx || id_z >= nz)	{ return; }
	const int	nxz = nx*nz;

	// y (receive) //
	f_d[id + nx*0]      = buff_dm[id_b        ];
	f_d[id + nx*1]      = buff_dm[id_b + nxz  ];

	f_d[id + nx*(ny-2)] = buff_dp[id_b        ];
	f_d[id + nx*(ny-1)] = buff_dp[id_b + nxz  ];
}


__global__ void 
copy_buff_to_global_z2 (
	      FLOAT *f_d,
	const FLOAT *buff_dm,
	const FLOAT *buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x =     threadIdx.x + blockDim.x*blockIdx.x,
		  		id_y =     threadIdx.y + blockDim.y*blockIdx.y;
	const int	id = id_x + nx*id_y;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// z (receive) //
	f_d[id + nxy*0]      = buff_dm[id        ];	
	f_d[id + nxy*1]      = buff_dm[id + nxy  ];	

	f_d[id + nxy*(nz-2)] = buff_dp[id        ];
	f_d[id + nxy*(nz-1)] = buff_dp[id + nxy  ];
}


// halo = 3 //
__global__ void
copy_buff_to_global_x3 (
	      FLOAT	*f_d,
   	const FLOAT	*buff_dm,
	const FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_y = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = nx*id_y + nxy*id_z,
		  		id_b =    id_y + ny *id_z;

	if (id_y >= ny || id_z >= nz)	{ return; }
	const int	nyz = ny*nz;

	// x (receive)
	f_d[id + 0]      = buff_dm[id_b        ];
	f_d[id + 1]      = buff_dm[id_b + nyz  ];
	f_d[id + 2]      = buff_dm[id_b + nyz*2];

	f_d[id + (nx-3)] = buff_dp[id_b        ];
	f_d[id + (nx-2)] = buff_dp[id_b + nyz  ];
	f_d[id + (nx-1)] = buff_dp[id_b + nyz*2];
}


__global__ void
copy_buff_to_global_y3 (
	      FLOAT	*f_d,
	const FLOAT	*buff_dm, 
	const FLOAT	*buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x = threadIdx.x + blockDim.x*blockIdx.x,
		  		id_z = threadIdx.y + blockDim.y*blockIdx.y;
	const int	id   = id_x + nxy*id_z,
		  		id_b = id_x + nx *id_z;

	if (id_x >= nx || id_z >= nz)	{ return; }
	const int	nxz = nx*nz;

	// y (receive) //
	f_d[id + nx*0]      = buff_dm[id_b        ];
	f_d[id + nx*1]      = buff_dm[id_b + nxz  ];
	f_d[id + nx*2]      = buff_dm[id_b + nxz*2];

	f_d[id + nx*(ny-3)] = buff_dp[id_b        ];
	f_d[id + nx*(ny-2)] = buff_dp[id_b + nxz  ];
	f_d[id + nx*(ny-1)] = buff_dp[id_b + nxz*2];
}


__global__ void 
copy_buff_to_global_z3 (
	      FLOAT *f_d,
	const FLOAT *buff_dm,
	const FLOAT *buff_dp,
	int nx, int ny, int nz)
{
	const int	nxy = nx*ny;
	const int	id_x =     threadIdx.x + blockDim.x*blockIdx.x,
		  		id_y =     threadIdx.y + blockDim.y*blockIdx.y;
	const int	id = id_x + nx*id_y;

	if (id_x >= nx || id_y >= ny)	{ return; }

	// z (receive) //
	f_d[id + nxy*0]      = buff_dm[id        ];	
	f_d[id + nxy*1]      = buff_dm[id + nxy  ];	
	f_d[id + nxy*2]      = buff_dm[id + nxy*2];	

	f_d[id + nxy*(nz-3)] = buff_dp[id        ];
	f_d[id + nxy*(nz-2)] = buff_dp[id + nxy  ];
	f_d[id + nxy*(nz-1)] = buff_dp[id + nxy*2];
}


// MPI_CUDA_Lib.cu //
