#ifndef RENUMBER_PARTICLE_H_
#define RENUMBER_PARTICLE_H_


#include <algorithm>
#include "defineParticleFlag.h"
#include "definePrecision.h"
#include "VariablesParticle.h"


void
Renumber_ParticlePosition (
	Particle	*particle_h,
	Particle	*particle_d);


void
CPU_Renumber_ParticlesPosition (
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos_h,
	ParticlePosition	*ppos_d);



// Renumber
void
Sort_ParticleArray_on_cpu (
	ParticlePosition	*ppos_h,
	int					parray_end);


int
Find_ParticleArray_on_cpu (
	ParticlePosition	*ppos_h,
	int					num_particles);


#endif
