#include "Renumber_Particle.h"

#include <cuda.h>
#include "mathLib_particle.h"


// thrust //
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
#include <thrust/find.h>
// thrust //


// sort
void
Renumber_ParticlePosition (
	Particle	*particle_h,
	Particle	*particle_d)
{
	// memcopy : device -> host //
	particle_h->pgrid.parray_end = particle_d->pgrid.parray_end;


	// host  //
	CPU_Renumber_ParticlesPosition (
		particle_h->pgrid,
		particle_h->ppos,
		particle_d->ppos); 


	// memcopy : host -> device //
	particle_d->pgrid.parray_end = particle_h->pgrid.parray_end;
}


void
CPU_Renumber_ParticlesPosition (
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos_h,
	ParticlePosition	*ppos_d
	)
{
	const int	parray_end_before = pgrid.parray_end;

	// memcopy : grid
	cudaMemcpy( ppos_h, ppos_d, parray_end_before * sizeof(ParticlePosition), cudaMemcpyDefault );

	// sort
	Sort_ParticleArray_on_cpu (ppos_h, parray_end_before);

	// array_end
	const int	parray_end_after = Find_ParticleArray_on_cpu (ppos_h, parray_end_before);


	// memcopy : grid
	cudaMemcpy( ppos_d, ppos_h, parray_end_before * sizeof(ParticlePosition), cudaMemcpyDefault );


	// parray_end (new)
	pgrid.parray_end = parray_end_after;
}


// Renumber
void
Sort_ParticleArray_on_cpu (
	ParticlePosition	*ppos_h,
	int					parray_end)
{
	std::sort(ppos_h, ppos_h + parray_end, less_sort_cpu_particle_state_p());
}


int
Find_ParticleArray_on_cpu (
	ParticlePosition	*ppos_h,
	int					num_particles)
{
	ParticlePosition	*fp = std::find_if(ppos_h, ppos_h + num_particles, NA_sort_cpu_particle_state_p());

	if ( (int)(fp - ppos_h) > num_particles )	{ 
		std::cout << "error : Find_ParticleArray_on_cpu \n";
		exit(3);
   	}
	if (ppos_h[(int)(fp - ppos_h)].state_p != PARTICLE_NA) {
		std::cout << "error find if : Find_ParticleArray_on_cpu \n";
		exit(3);
	}

	// return : valid array size
	return	(int)(fp - ppos_h);
}


// Renumber_Particle.cu //
