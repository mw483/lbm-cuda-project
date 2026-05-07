#include "MPI_Particle.h"

#include <mpi.h>
#include "boundary_flag_particle.h"
#include "defineParticleFlag.h"
#include "mathLib_particle.h"
#include "macroCUDA.h"
#include "functionLib.h"
// (YOKOUCHI 2020)
#include "Define_user.h"


// Calculation
void
MPI_Particle (
	int				index_num,
	const Domain	&domain,
	Particle		*particle_h,
	Particle		*particle_d)
{
	// frag (mpi)
	MPI_Frag_Particle (domain, particle_d->pgrid, particle_d->ppos);


	// compaction on gpu (send buffer)
	CompactParticle_on_gpu (particle_d->pgrid,        particle_d->ppos,
						    particle_d->pgrid_buff_s, particle_d->ppos_buff_s);

	// device to host
	particle_h->pgrid_buff_s.parray_end = particle_d->pgrid_buff_s.parray_end; 
	CUDA_SAFE_CALL( cudaMemcpy( particle_h->ppos_buff_s, particle_d->ppos_buff_s, (particle_h->pgrid_buff_s.parray_end)*sizeof(ParticlePosition), cudaMemcpyDefault) );


	// send buffer sort
	Sort_ParticleBuffer_on_cpu (particle_h->ppos_buff_s, particle_h->pgrid_buff_s.parray_end);


	// find
	Find_ParticleBuffer_on_cpu (particle_h->pmpi_host, particle_h->ppos_buff_s, particle_h->pgrid_buff_s.parray_end);
	MPI_Barrier(MPI_COMM_WORLD);


	// MPI communication (host : send -> recv)
	MPI_Comm_Particle (
			particle_h->pmpi_host,
			particle_h->pgrid_buff_s, particle_h->ppos_buff_s,
			particle_h->pgrid_buff_r, particle_h->ppos_buff_r);

	MPI_Barrier(MPI_COMM_WORLD);


	// host to device
	particle_d->pgrid_buff_r.parray_end = particle_h->pgrid_buff_r.parray_end; 
	CUDA_SAFE_CALL( cudaMemcpy( particle_d->ppos_buff_r, particle_h->ppos_buff_r, (particle_h->pgrid_buff_r.parray_end)*sizeof(ParticlePosition), cudaMemcpyDefault) );

	// update (read buffer)
	Update_MPI_Particle (particle_d->pgrid,        particle_d->ppos,
						 particle_d->pgrid_buff_r, particle_d->ppos_buff_r,
						 index_num);


	// boundary
	Check_Boundary_Particle (domain, particle_d->pgrid, particle_d->ppos);
}


// function
// MPI frag (outside domain)
void
MPI_Frag_Particle (
	const Domain		&domain,
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos)
{
	dim3	grid,
			block;

	const int	parray_end = pgrid.parray_end;
	
	const int	thread_length  = 256;
	const int	block_number   = parray_end/thread_length + 1;

	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length, 1, 1);

	// particle //
	FLOAT	xs_range[2], ys_range[2], zs_range[2];

	functionLib::init2(xs_range, domain.x_min, domain.x_max);
	functionLib::init2(ys_range, domain.y_min, domain.y_max);
	functionLib::init2(zs_range, domain.z_min, domain.z_max);
	// particle //

	CUDA_MPI_Frag_Particle <<< grid, block >>> (
		ppos,
		xs_range[0], ys_range[0], zs_range[0],
		xs_range[1], ys_range[1], zs_range[1],
		parray_end);

	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}


// MPI communication
void
MPI_Comm_Particle (
	ParticleMPIHost		pmpi_host,
	ParticleGrid		&pgrid_buff_s,
   	ParticlePosition	*ppos_buff_s,
	ParticleGrid		&pgrid_buff_r,
   	ParticlePosition	*ppos_buff_r)
{
	const int	num_mpi = NUM_MPI_PARTICLE;

	MPI_Status	stat;
	MPI_Request	requ_s[num_mpi],
			   	requ_r[num_mpi];

	int		num_send[num_mpi],
			num_recv[num_mpi];

	// particle grid
	for (int i=0; i<num_mpi; i++) {
		num_send[i] = pmpi_host.pmpi_buff_size_to[i];
		num_recv[i] = 0;

		MPI_Isend(&num_send[i], 1, MPI_INT, pmpi_host.pmpi_host_to  [i], 100000+i, MPI_COMM_WORLD, &requ_s[i]);
		MPI_Irecv(&num_recv[i], 1, MPI_INT, pmpi_host.pmpi_host_from[i], 100000+i, MPI_COMM_WORLD, &requ_r[i]);
	}

	for (int i=0; i<num_mpi; i++) {
		MPI_Wait(&requ_r[i], &stat);
		MPI_Wait(&requ_s[i], &stat);
	}
	MPI_Barrier(MPI_COMM_WORLD);


	int		num_send_offset[num_mpi+1],
			num_recv_offset[num_mpi+1];
	num_send_offset[0] = 0;
	num_recv_offset[0] = 0;
	for (int i=1; i<num_mpi; i++) {
		num_send_offset[i] = num_send_offset[i-1] + num_send[i-1];
		num_recv_offset[i] = num_recv_offset[i-1] + num_recv[i-1];
	}
	num_send_offset[num_mpi] = num_send_offset[num_mpi-1] + num_send[num_mpi-1];
	num_recv_offset[num_mpi] = num_recv_offset[num_mpi-1] + num_recv[num_mpi-1];

	pgrid_buff_r.parray_end = num_recv_offset[num_mpi];

	// particle position
	for (int i=0; i<num_mpi; i++) {
		MPI_Isend(&ppos_buff_s[num_send_offset[i]], num_send[i], pmpi_host.pposType, pmpi_host.pmpi_host_to  [i], 100000+i, MPI_COMM_WORLD, &requ_s[i]);
		MPI_Irecv(&ppos_buff_r[num_recv_offset[i]], num_recv[i], pmpi_host.pposType, pmpi_host.pmpi_host_from[i], 100000+i, MPI_COMM_WORLD, &requ_r[i]);
	}

	for (int i=0; i<num_mpi; i++) {
		MPI_Wait(&requ_r[i], &stat);
		MPI_Wait(&requ_s[i], &stat);
	}
}


// sort
void
Sort_ParticleBuffer_on_cpu (
	ParticlePosition	*ppos_h,
	int					parray_end)
{
    std::sort( ppos_h, ppos_h + parray_end, less_sort_cpu_particle_state_p() );
}


// Find
void
Find_ParticleBuffer_on_cpu (
	ParticleMPIHost		&pmpi_host,
   	ParticlePosition	*ppos_h,
   	int					num_particles)
{
	const int	num_mpi = NUM_MPI_PARTICLE;

	ParticlePosition	*fp1, *fp2;
	fp1 = ppos_h;
	
	int		check_array_start = 1;
//	int		buff_array_start  = 2;
	int		num_rest_particles = num_particles;

	// ppos_h : state_p 1 - 8
	for (int i=0; i<num_mpi; i++) {
		if (fp1->state_p == check_array_start) {
//			fp2 = std::find_if(fp1, fp1 + num_rest_particles, sort_cpu_particle_state_p           ( buff_array_start  ));
			fp2 = std::find_if(fp1, fp1 + num_rest_particles, sort_cpu_particle_state_p_not_equal ( check_array_start ));
		}
		else {
			fp2 = fp1;
		}

		pmpi_host.pmpi_buff_size_to[i]  = (int)(fp2 - fp1);
		num_rest_particles             -= (int)(fp2 - fp1);

		fp1 = fp2;

		check_array_start++;
//		buff_array_start++;
//		if (buff_array_start > 8) { buff_array_start = PARTICLE_NA; }
	}
	if (num_rest_particles != 0) {
		std::cout << "error : num_rest_particles \n";
	}

}


// compaction for MPI
void 
CompactParticle_on_gpu (
	ParticleGrid &pgrid,      ParticlePosition *ppos_d,
	ParticleGrid &pgrid_buff, ParticlePosition *ppos_buff_d)
{
	const int		parray_end = pgrid.parray_end;
	if (parray_end == 0) { return; }

    thrust::device_ptr< ParticlePosition > thrust_ppos_d(ppos_d);
    thrust::device_ptr< ParticlePosition > thrust_ppos_buff_d(ppos_buff_d);

	// adress : first
	thrust::device_ptr< ParticlePosition > first = thrust_ppos_buff_d;

	// adress : last & ppos_d to ppos_buff_d
	thrust::device_ptr< ParticlePosition > end   = thrust::copy_if( thrust_ppos_d, thrust_ppos_d + parray_end, thrust_ppos_buff_d,
				mpi_sort_gpu_particle_state_p() );

	const int	num_out_particle = (int)(end - first);
	pgrid_buff.parray_end = num_out_particle;

	if (num_out_particle < 0) {
		std::cout << "compaction error" << "num_out_particle = " << num_out_particle << std::endl;
		std::cout << "FILE=" << __FILE__ << "LINE=" << __LINE__ << std::endl;
		exit(-1);
	}

	// memory frg clear
	dim3	grid,
			block;

	const int	thread_length  = 256;
	const int	block_number   = parray_end/thread_length + 1;

	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length, 1, 1);

	// particle position 
	CUDA_Particle_MPI_Clear <<< grid, block >>> (
		ppos_d,
		parray_end);

	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}


void 
Update_MPI_Particle (
	ParticleGrid		&pgrid,      
	ParticlePosition	*ppos_d,
	ParticleGrid		&pgrid_buff,
   	ParticlePosition	*ppos_buff_d,
	int					index_num)
{
	const int	parray_mpi = pgrid_buff.parray_end;

	const int	thread_length  = 256;
	const int	block_number   = parray_mpi/thread_length + 1;

	dim3	grid,
			block;
	functionLib::set_dim3(&grid,  block_number,  1, 1);
	functionLib::set_dim3(&block, thread_length, 1, 1);

	// (YOKOUCHI 2020)
	if (user_flags::flg_particle == 1 || user_flags::flg_particle == 2 || user_flags::flg_particle == 3) {
		GPU_Particle_State_Cal_LSM <<< grid, block >>> (ppos_buff_d, pgrid_buff.num_particle_max, index_num);
	} else {
		GPU_Particle_State_Cal <<< grid, block >>> (ppos_buff_d, pgrid_buff.num_particle_max, index_num);
	}
	cudaThreadSynchronize(); // cudaThreadSynchronize *****


	int		parray_end = pgrid.parray_end;

	// &ppos_d[parray_end+1] ?? or &ppos_d[parray_end] ???
	CUDA_SAFE_CALL( cudaMemcpy( &ppos_d[parray_end  ], ppos_buff_d, (parray_mpi)*sizeof(ParticlePosition), cudaMemcpyDefault) );
//	CUDA_SAFE_CALL( cudaMemcpy( &ppos_d[parray_end+1], ppos_buff_d, (parray_mpi)*sizeof(ParticlePosition), cudaMemcpyDefault) );
	pgrid.parray_end += parray_mpi;


	if (pgrid.parray_end > pgrid.num_particle_max)	{
		std::cout << "mpi : number of particles is too large!!!\n";
		exit(-1);
	}
}


void
Check_Boundary_Particle (
	const Domain		&domain,
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos)
{
	dim3	grid,
			block;

	const int	parray_end = pgrid.parray_end;
	
	const int	thread_length  = 256;
	const int	block_number   = parray_end/thread_length + 1;

	functionLib::set_dim3(&grid,    block_number,  1, 1);
	functionLib::set_dim3(&block,   thread_length, 1, 1);

	// particle //
	FLOAT	xs_range[2], ys_range[2], zs_range[2];

	functionLib::init2(xs_range, domain.x_min, domain.x_max);
	functionLib::init2(ys_range, domain.y_min, domain.y_max);
	functionLib::init2(zs_range, domain.z_min, domain.z_max);
	// particle //


	boundary_flag_particle_cuda  <<< grid, block >>> (
		ppos,
		xs_range[0], ys_range[0], zs_range[0],
		xs_range[1], ys_range[1], zs_range[1],
		parray_end);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}


// MPI_Particle.cu
