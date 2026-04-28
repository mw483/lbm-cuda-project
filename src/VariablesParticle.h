#ifndef VARIABLESPARTICLES_H_
#define VARIABLESPARTICLES_H_

#include <mpi.h>
#include "definePrecision.h"

#include "defineParticleFlag.h"


// Particle
struct ParticleCalFlag {
	// frag : restart
	int		prestart;
	int		pout_step;

	int		gen_step;
};

struct ParticleGrid {
	int		num_particle_max;
	int		parray_end;
};

struct ParticlePosition {
	FLOAT	x_p, y_p, z_p;
	FLOAT	vel_p;
	
	// (YOKOUCHI 2020)
	FLOAT 	u_sgs, v_sgs, w_sgs; // for LSM

	int		state_p;
	int		source_index_p;
};

struct ParticleMPIHost {
	int		pid_rank;

	int		pmpi_host_to[NUM_MPI_PARTICLE];
	int		pmpi_host_from[NUM_MPI_PARTICLE];

	int		pmpi_buff_size_to[NUM_MPI_PARTICLE];

	// MPI *****
	MPI_Datatype	pposType;
};


// All
struct Particle {
	ParticleCalFlag		pfrag;

	ParticleGrid		pgrid;
	ParticlePosition	*ppos;

	// buffer
	ParticleGrid		pgrid_buff_s, pgrid_buff_r;
	ParticlePosition	*ppos_buff_s, *ppos_buff_r;

	// mpi host
	ParticleMPIHost	pmpi_host;
};

#endif


