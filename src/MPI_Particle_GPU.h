#ifndef MPI_PARTICLE_GPU_H_
#define MPI_PARTICLE_GPU_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Define.h"
#include "VariablesParticle.h"

// CUDA function *****
// MPI frag
__global__ void CUDA_MPI_Frag_Particle(
		ParticlePosition *ppos,
		FLOAT xs_min, FLOAT ys_min, FLOAT zs_min,
		FLOAT xs_max, FLOAT ys_max, FLOAT zs_max,
		int num);

__global__ void GPU_Particle_State_Cal(
		ParticlePosition *ppos, int num, int index_num);

// (YOKOUCHI 2020)
__global__ void GPU_Particle_State_Cal_LSM (
		ParticlePosition *ppos, int num, int index_num);

__global__ void CUDA_Particle_MPI_Clear(
		ParticlePosition *ppos, int num);

__global__ void GPU_MPI_Frag(
		ParticlePosition *ppos, int *frg, int num);

#endif

