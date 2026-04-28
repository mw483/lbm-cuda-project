#ifndef MPI_PARTICLE_H_
#define MPI_PARTICLE_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "Define.h"
#include "Variables.h"
#include "VariablesParticle.h"
#include "MPI_Particle_GPU.h"


// thrust *****
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/device_free.h>
#include <thrust/replace.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
// thrust *****


// Calculation
void
MPI_Particle (
	int				index_num,
	const Domain	&domain,
	Particle		*particle_h,
	Particle		*particle_d);


// function
// MPI frag (outside domain)
void
MPI_Frag_Particle (
	const Domain		&domain,
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos);


// MPI communication
void
MPI_Comm_Particle (
	ParticleMPIHost		pmpi_host,
	ParticleGrid		&pgrid_buff_s,
   	ParticlePosition	*ppos_buff_s,
	ParticleGrid		&pgrid_buff_r,
   	ParticlePosition	*ppos_buff_r);


// sort
void
Sort_ParticleBuffer_on_cpu (
	ParticlePosition	*ppos_h,
	int					parray_end);


// Find
void
Find_ParticleBuffer_on_cpu (
	ParticleMPIHost		&pmpi_host,
   	ParticlePosition	*ppos_h,
   	int					num_particles);


// compaction for MPI
void 
CompactParticle_on_gpu (
	ParticleGrid &pgrid,      ParticlePosition *ppos_d,
	ParticleGrid &pgrid_buff, ParticlePosition *ppos_buff_d);


void Update_MPI_Particle(
		ParticleGrid &pgrid,      ParticlePosition *ppos_d,
		ParticleGrid &pgrid_buff, ParticlePosition *ppos_buff_d,
		int index_num);

void
Check_Boundary_Particle (
	const Domain		&domain,
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos);


#endif
