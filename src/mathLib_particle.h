#ifndef MATH_LIB_OPERATOR_H_
#define MATH_LIB_OPERATOR_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Macro.h"
#include "Variables.h"
#include "VariablesParticle.h"

#include "defineParticleFlag.h"


// operator
struct sort_cpu_particle_state_p {
	explicit sort_cpu_particle_state_p(int i) : i_(i) {}
    bool operator()(const ParticlePosition &a)
    {
        return a.state_p == i_;
    }

	int i_;
};

struct sort_cpu_particle_state_p_not_equal {
	explicit sort_cpu_particle_state_p_not_equal(int i) : i_(i) {}
    bool operator()(const ParticlePosition &a)
    {
        return a.state_p != i_;
    }

	int i_;
};


struct mpi_sort_gpu_particle_state_p {
    __host__ __device__ bool operator()(const ParticlePosition &a)
    {
        return ((a.state_p != PARTICLE_CAL) && (a.state_p != PARTICLE_NA));
    }
};


struct less_sort_cpu_particle_state_p {
    bool operator()(const ParticlePosition &a, const ParticlePosition &b)
    {
        return a.state_p < b.state_p;
    }
};


struct less_sort_particle_state_p {
    __host__ __device__ bool operator()(const ParticlePosition &a, const ParticlePosition &b)
    {
        return a.state_p < b.state_p;
    }
};


// operator
struct NA_sort_particle_state_p {
    __host__ __device__ bool operator()(const ParticlePosition &a)
    {
        return a.state_p == PARTICLE_NA;
    }
};


struct NA_sort_cpu_particle_state_p {
    bool operator()(const ParticlePosition &a)
    {
        return a.state_p == PARTICLE_NA;
    }
};


#endif

