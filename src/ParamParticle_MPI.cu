#include "ParamParticle_MPI.h"

#include "indexLib.h"
#include "defineParticleFlag.h"


// public *****
void Init_Particle_MPI(MPIinfo *mpi,
	Domain *cdo_h, ParticlePosition *ppos_h, ParticleMPIHost *pmpi_h, 
	Domain *cdo_d, ParticlePosition *ppos_d, ParticleMPIHost *pmpi_d)
{
	const int	ncpu_x = mpi->ncpu_x,
				ncpu_y = mpi->ncpu_y,
				ncpu_z = mpi->ncpu_z;

	const int	rank    = mpi->rank;
	const int	rank_x  = mpi->rank_x,
				rank_y  = mpi->rank_y,
				rank_z  = mpi->rank_z;
	const int	rank_xm = mpi->rank_xm,
				rank_ym = mpi->rank_ym,
				rank_zm = mpi->rank_zm;
	const int	rank_xp = mpi->rank_xp,
				rank_yp = mpi->rank_yp,
				rank_zp = mpi->rank_zp;


	// id
	pmpi_h->pid_rank = rank;

	const int	num_mpi = NUM_MPI_PARTICLE;

	for (int i=0; i<num_mpi; i++) {
		pmpi_h->pmpi_host_to  [i] = 0;
		pmpi_h->pmpi_host_from[i] = 0;

		pmpi_h->pmpi_buff_size_to[i] = 0;
	}

	// pmpi_host_to *****


	// MPI tag
	const int	ncpu3[3] = { ncpu_x, ncpu_y, ncpu_z };


	// MPI 1D
	// tag_particle_xp
	// tag_particle_xm
	pmpi_h->pmpi_host_to[0] = indexLib::get_index (rank_xp, rank_y, rank_z, ncpu3);
	pmpi_h->pmpi_host_to[1] = indexLib::get_index (rank_xm, rank_y, rank_z, ncpu3);

	// tag_particle_yp
	// tag_particle_ym
	pmpi_h->pmpi_host_to[2] = indexLib::get_index (rank_x, rank_yp, rank_z, ncpu3);
	pmpi_h->pmpi_host_to[3] = indexLib::get_index (rank_x, rank_ym, rank_z, ncpu3);

	// tag_particle_zp
	// tag_particle_zm
	pmpi_h->pmpi_host_to[4] = indexLib::get_index (rank_x, rank_y, rank_zp, ncpu3);
	pmpi_h->pmpi_host_to[5] = indexLib::get_index (rank_x, rank_y, rank_zm, ncpu3);


	// MPI 2D
	// tag_particle_xpyp
	// tag_particle_xmyp
	pmpi_h->pmpi_host_to[6]  = indexLib::get_index (rank_xp, rank_yp, rank_z, ncpu3);
	pmpi_h->pmpi_host_to[7]  = indexLib::get_index (rank_xm, rank_yp, rank_z, ncpu3);

	// tag_particle_ypzp
	// tag_particle_ymzp
	pmpi_h->pmpi_host_to[8]  = indexLib::get_index (rank_x, rank_yp, rank_zp, ncpu3);
	pmpi_h->pmpi_host_to[9]  = indexLib::get_index (rank_x, rank_ym, rank_zp, ncpu3);

	// tag_particle_xpzp
	// tag_particle_xmzp
	pmpi_h->pmpi_host_to[10] = indexLib::get_index (rank_xp, rank_y, rank_zp, ncpu3);
	pmpi_h->pmpi_host_to[11] = indexLib::get_index (rank_xm, rank_y, rank_zp, ncpu3);


	// tag_particle_xpym
	// tag_particle_xmym
	pmpi_h->pmpi_host_to[12] = indexLib::get_index (rank_xp, rank_ym, rank_z, ncpu3);
	pmpi_h->pmpi_host_to[13] = indexLib::get_index (rank_xm, rank_ym, rank_z, ncpu3);

	// tag_particle_ypzm
	// tag_particle_ymzm
	pmpi_h->pmpi_host_to[14] = indexLib::get_index (rank_x, rank_yp, rank_zm, ncpu3);
	pmpi_h->pmpi_host_to[15] = indexLib::get_index (rank_x, rank_ym, rank_zm, ncpu3);

	// tag_particle_xpzm
	// tag_particle_xmzm
	pmpi_h->pmpi_host_to[16] = indexLib::get_index (rank_xp, rank_y, rank_zm, ncpu3);
	pmpi_h->pmpi_host_to[17] = indexLib::get_index (rank_xm, rank_y, rank_zm, ncpu3);


	// MPI 3D
	// tag_particle_xpypzp
	pmpi_h->pmpi_host_to[18] = indexLib::get_index (rank_xp, rank_yp, rank_zp, ncpu3);

	// tag_particle_xmypzp
	// tag_particle_xpymzp
	// tag_particle_xpypzm
	pmpi_h->pmpi_host_to[19] = indexLib::get_index (rank_xm, rank_yp, rank_zp, ncpu3);
	pmpi_h->pmpi_host_to[20] = indexLib::get_index (rank_xp, rank_ym, rank_zp, ncpu3);
	pmpi_h->pmpi_host_to[21] = indexLib::get_index (rank_xp, rank_yp, rank_zm, ncpu3);

	// tag_particle_xmymzp
	// tag_particle_xpymzm
	// tag_particle_xmypzm
	pmpi_h->pmpi_host_to[22] = indexLib::get_index (rank_xm, rank_ym, rank_zp, ncpu3);
	pmpi_h->pmpi_host_to[23] = indexLib::get_index (rank_xp, rank_ym, rank_zm, ncpu3);
	pmpi_h->pmpi_host_to[24] = indexLib::get_index (rank_xm, rank_yp, rank_zm, ncpu3);

	// tag_particle_xmymzm
	pmpi_h->pmpi_host_to[25] = indexLib::get_index (rank_xm, rank_ym, rank_zm, ncpu3);


	// pmpi_host_from *****
	// MPI 1D
	pmpi_h->pmpi_host_from[0] = pmpi_h->pmpi_host_to[1];
	pmpi_h->pmpi_host_from[1] = pmpi_h->pmpi_host_to[0];

	pmpi_h->pmpi_host_from[2] = pmpi_h->pmpi_host_to[3];
	pmpi_h->pmpi_host_from[3] = pmpi_h->pmpi_host_to[2];

	pmpi_h->pmpi_host_from[4] = pmpi_h->pmpi_host_to[5];
	pmpi_h->pmpi_host_from[5] = pmpi_h->pmpi_host_to[4];

	// MPI 2D
	pmpi_h->pmpi_host_from[6] = pmpi_h->pmpi_host_to[13];
	pmpi_h->pmpi_host_from[7] = pmpi_h->pmpi_host_to[12];

	pmpi_h->pmpi_host_from[8] = pmpi_h->pmpi_host_to[15];
	pmpi_h->pmpi_host_from[9] = pmpi_h->pmpi_host_to[14];

	pmpi_h->pmpi_host_from[10] = pmpi_h->pmpi_host_to[17];
	pmpi_h->pmpi_host_from[11] = pmpi_h->pmpi_host_to[16];


	pmpi_h->pmpi_host_from[12] = pmpi_h->pmpi_host_to[7];
	pmpi_h->pmpi_host_from[13] = pmpi_h->pmpi_host_to[6];

	pmpi_h->pmpi_host_from[14] = pmpi_h->pmpi_host_to[9];
	pmpi_h->pmpi_host_from[15] = pmpi_h->pmpi_host_to[8];

	pmpi_h->pmpi_host_from[16] = pmpi_h->pmpi_host_to[11];
	pmpi_h->pmpi_host_from[17] = pmpi_h->pmpi_host_to[10];

	// MPI 3D
	pmpi_h->pmpi_host_from[18] = pmpi_h->pmpi_host_to[25];

	pmpi_h->pmpi_host_from[19] = pmpi_h->pmpi_host_to[23];
	pmpi_h->pmpi_host_from[20] = pmpi_h->pmpi_host_to[24];
	pmpi_h->pmpi_host_from[21] = pmpi_h->pmpi_host_to[22];

	pmpi_h->pmpi_host_from[22] = pmpi_h->pmpi_host_to[21];
	pmpi_h->pmpi_host_from[23] = pmpi_h->pmpi_host_to[19];
	pmpi_h->pmpi_host_from[24] = pmpi_h->pmpi_host_to[20];

	pmpi_h->pmpi_host_from[25] = pmpi_h->pmpi_host_to[18];


	// MPI_struct *****
	const int	num_ppos = 6;
	int		lengtharray[num_ppos];           /* Array of lengths */
	MPI_Aint	disparray[num_ppos];        /* Array of displacements */
	MPI_Datatype	typearray[num_ppos];    /* Array of MPI datatypes */

	MPI_Aint	startaddress, address;   /* Variables used to calculate displacements */

	// lengtharray
	for (int i=0; i<num_ppos; i++) {
		lengtharray[i] = 1;
	}

	// typearray
	typearray[0] = MFLOAT;
	typearray[1] = MFLOAT;
	typearray[2] = MFLOAT;
	typearray[3] = MFLOAT;
	typearray[4] = MPI_INT;
	typearray[5] = MPI_INT;

	/* First element, a, is at displacement 0 */
	disparray[0] = 0;

	/* Calculate displacement of b */
	MPI_Get_address(&ppos_h->x_p, &startaddress);
	MPI_Get_address(&ppos_h->y_p, &address);
	disparray[1] = address-startaddress;     /* Displacement of second element, b */

	MPI_Get_address(&ppos_h->z_p, &address);
	disparray[2] = address-startaddress;     /* Displacement of third element, n */

	MPI_Get_address(&ppos_h->vel_p, &address);
	disparray[3] = address-startaddress;     /* Displacement of third element, n */

	MPI_Get_address(&ppos_h->state_p, &address);
	disparray[4] = address-startaddress;     /* Displacement of third element, n */

	MPI_Get_address(&ppos_h->source_index_p, &address);
	disparray[5] = address-startaddress;     /* Displacement of third element, n */

	/* Build the data structure my_type */
	MPI_Type_create_struct(num_ppos, lengtharray, disparray, typearray, &pmpi_h->pposType);
	MPI_Type_commit(&pmpi_h->pposType);


	// memcpy
	pmpi_d->pid_rank = pmpi_h->pid_rank;
	for (int i=0; i<num_mpi; i++) {
		pmpi_d->pmpi_host_to  [i] = pmpi_h->pmpi_host_to  [i];
		pmpi_d->pmpi_host_from[i] = pmpi_h->pmpi_host_from[i];

		pmpi_d->pmpi_buff_size_to[i] = pmpi_h->pmpi_buff_size_to[i];
	}

	MPI_Type_create_struct(num_ppos, lengtharray, disparray, typearray, &pmpi_d->pposType);
	MPI_Type_commit(&pmpi_d->pposType);
}


// ParamParticle_MPI.cu //
