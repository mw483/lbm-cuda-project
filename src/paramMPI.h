#ifndef PARAMMPI_H_
#define PARAMMPI_H_


#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "definePrecision.h"
#include "stMPIinfo.h"
#include "stDomain.h"


class
paramMPI {
private:
	MPIinfo	mpiinfo_;

	Buffer	buff_h_,
			buff_d_;

	// domain
	int		nx_,  ny_,  nz_;
	int		halo_;


	// ncpu div //
	int		ncpu_div_x_, ncpu_div_y_, ncpu_div_z_, ncpu_div_p_;

public:
	paramMPI () {}

	paramMPI (
		char	*program_name,
		int		argc,
	   	char	*argv[])
	{
		set (
			program_name,
			argc,
			argv);

		check_MPI ();
	}

	~paramMPI () {}


	int		ncpu  ()	const { return	mpiinfo_.ncpu; }
	int		ncpu_x()	const { return	mpiinfo_.ncpu_x; }
	int		ncpu_y()	const { return	mpiinfo_.ncpu_y; }
	int		ncpu_z()	const { return	mpiinfo_.ncpu_z; }

	int		rank  ()	const { return	mpiinfo_.rank; }
	int		rank_x()	const { return	mpiinfo_.rank_x; }
	int		rank_y()	const { return	mpiinfo_.rank_y; }
	int		rank_z()	const { return	mpiinfo_.rank_z; }

	int		rank_xm()	const { return	mpiinfo_.rank_xm; }
	int		rank_ym()	const { return	mpiinfo_.rank_ym; }
	int		rank_zm()	const { return	mpiinfo_.rank_zm; }

	int		rank_xp()	const { return	mpiinfo_.rank_xp; }
	int		rank_yp()	const { return	mpiinfo_.rank_yp; }
	int		rank_zp()	const { return	mpiinfo_.rank_zp; }

	int		id_rank   ()	const { return	mpiinfo_.id_rank; }
	int		id_rank_xm()	const { return	mpiinfo_.id_rank_xm; }
	int		id_rank_ym()	const { return	mpiinfo_.id_rank_ym; }
	int		id_rank_zm()	const { return	mpiinfo_.id_rank_zm; }
	int		id_rank_xp()	const { return	mpiinfo_.id_rank_xp; }
	int		id_rank_yp()	const { return	mpiinfo_.id_rank_yp; }
	int		id_rank_zp()	const { return	mpiinfo_.id_rank_zp; }

	const MPIinfo	&mpiinfo() const { return mpiinfo_; }

public:
	// initialize //
	void
	set (
		char	*program_name,
		int		argc,
	   	char	*argv[]);

	void	set_buffer (const Domain	&domain);


	// cout //
	void	cout_MPI  ();


	// mpi communication //
	void
	mpi_cuda_x (
		FLOAT	**fp,
		int		phy_num);


	void
	mpi_cuda_y (
		FLOAT	**fp,
		int		phy_num);


	void
	mpi_cuda_z (
		FLOAT	**fp,
		int		phy_num);


	void
	mpi_cuda_xyz (
		FLOAT	**fp,
		int		phy_num);

private:
	// check //
	void	check_MPI ();


	// memory //
	void
	allocate_host (
		int num_halo,
		int num_variables);

	void
	allocate_device  (
		int num_halo,
		int num_variables);

	void
	memcpy_buffer (
		int num_halo,
		int num_variables);


	// mpi communication //
	void
	mpi_Isend_Irecv (
		FLOAT	*buff_rm,	// recv buffer -
		FLOAT	*buff_rp,	// recv buffer +
		FLOAT	*buff_sm,	// send buffer -
		FLOAT	*buff_sp,	// send buffer +
		int		id_rank_m,
		int		id_rank_p,
		int		tag1,
		int		tag2,
		int		nsize);


	void	mpi_Isend_Irecv_x (int	nsize);
	void	mpi_Isend_Irecv_y (int	nsize);
	void	mpi_Isend_Irecv_z (int	nsize);


	// download //
	void
	buff_download_x (
	   	FLOAT	**fp,
		int		phy_num,
		int		nsize);


	void
	buff_download_y (
	   	FLOAT	**fp,
		int		phy_num,
		int		nsize);


	void
	buff_download_z (
	   	FLOAT	**fp,
		int		phy_num,
		int		nsize);


	// upload //
	void
	buff_upload_x (
	   	FLOAT	**fp,
		int		phy_num,
		int		nsize);


	void
	buff_upload_y (
	   	FLOAT	**fp,
		int		phy_num,
		int		nsize);


	void
	buff_upload_z (
	   	FLOAT	**fp,
		int		phy_num,
		int		nsize);
};


// Host <--> Device //
// device to host buffer //
// halo = 1 //
__global__ void
copy_global_to_buff_x1 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_global_to_buff_y1 (
	const FLOAT	*f_d,
   	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_global_to_buff_z1 (
	const FLOAT *f_d,
	      FLOAT *buff_dm,
	      FLOAT *buff_dp,
	int nx, int ny, int nz);


// halo = 2 //
__global__ void
copy_global_to_buff_x2 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_global_to_buff_y2 (
	const FLOAT	*f_d,
   	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_global_to_buff_z2 (
	const FLOAT *f_d,
	      FLOAT *buff_dm,
	      FLOAT *buff_dp,
	int nx, int ny, int nz);


// halo = 3 //
__global__ void
copy_global_to_buff_x3 (
	const FLOAT	*f_d,
	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_global_to_buff_y3 (
	const FLOAT	*f_d,
   	      FLOAT	*buff_dm,
	      FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_global_to_buff_z3 (
	const FLOAT *f_d,
	      FLOAT *buff_dm,
	      FLOAT *buff_dp,
	int nx, int ny, int nz);


// host buffer to device //
// halo = 1 //
__global__ void
copy_buff_to_global_x1 (
	      FLOAT	*f_d,
   	const FLOAT	*buff_dm,
	const FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_buff_to_global_y1 (
	      FLOAT	*f_d,
	const FLOAT	*buff_dm, 
	const FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void 
copy_buff_to_global_z1 (
	      FLOAT *f_d,
	const FLOAT *buff_dm,
	const FLOAT *buff_dp,
	int nx, int ny, int nz);


// halo = 2 //
__global__ void
copy_buff_to_global_x2 (
	      FLOAT	*f_d,
   	const FLOAT	*buff_dm,
	const FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_buff_to_global_y2 (
	      FLOAT	*f_d,
	const FLOAT	*buff_dm, 
	const FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void 
copy_buff_to_global_z2 (
	      FLOAT *f_d,
	const FLOAT *buff_dm,
	const FLOAT *buff_dp,
	int nx, int ny, int nz);


// halo = 3 //
__global__ void
copy_buff_to_global_x3 (
	      FLOAT	*f_d,
   	const FLOAT	*buff_dm,
	const FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void
copy_buff_to_global_y3 (
	      FLOAT	*f_d,
	const FLOAT	*buff_dm, 
	const FLOAT	*buff_dp,
	int nx, int ny, int nz);


__global__ void 
copy_buff_to_global_z3 (
	      FLOAT *f_d,
	const FLOAT *buff_dm,
	const FLOAT *buff_dp,
	int nx, int ny, int nz);


#endif
