#ifndef PARAMPARTICLE_MPI_H_
#define PARAMPARTICLE_MPI_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mpi.h>
#include "Variables.h"
#include "VariablesParticle.h"

#include "definePrecision.h"
#include "defineParticleFlag.h"


void Init_Particle_MPI(MPIinfo *mpi,
		Domain *cdo_h, ParticlePosition *ppos_h, ParticleMPIHost *pmpi_h, 
		Domain *cdo_d, ParticlePosition *ppos_d, ParticleMPIHost *pmpi_d);

#endif

