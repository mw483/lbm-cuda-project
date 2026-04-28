#ifndef PARAMPARTICLE_H_
#define PARAMPARTICLE_H_

#include "VariablesParticle.h"

#include "stDomain.h"
#include "stMPIinfo.h"


class	paramParticle {
private:
	int		rank_;
	int		num_particle_max_;

public:	
	paramParticle () {}

	paramParticle (
		char		*program_name,
		int			argc,
	   	char		*argv[], 
		int			rank,
		Particle	*particle_h, 
		Particle	*particle_d
		) :
		rank_(rank)
	{
		set (
			program_name,
			argc,
			argv,
			particle_h,
			particle_d
			);
	}

	~paramParticle () {}

public:
	void
	set (
		char		*program_name,
		int			argc,
	   	char		*argv[], 
		Particle	*particle_h, 
		Particle	*particle_d
		);


	// memcpy //
	void 
	cudaMemcpy_ParticleCalFlag (
		int n, 
		      ParticleCalFlag &pfragn,
	   	const ParticleCalFlag &pfrag
		);


	void
	cudaMemcpy_ParticleGrid (
		      ParticleGrid &pgridn,
		const ParticleGrid &pgrid
		);


	void
	cudaMemcpy_ParticlePosition (
		int n,
		      ParticlePosition *pposn,
		const ParticlePosition *ppos
		);


	// restart //
	void
	Read_ParticlePosition (
		Particle *particle_h,
		Particle *particle_d
		);


	void	Output_Restart_ParticlePosition (Particle *particle_h);

private:
	void
	allocate_ParticlePosition_All (
		ParticlePosition **ppos_h,
		ParticlePosition **ppos_d,
		int n);

	void
	allocate_ParticlePosition_Pinned_All (
		ParticlePosition **ppos_h,
		ParticlePosition **ppos_d,
		int n);


	void
	init_ParticlePosition (
		ParticlePosition *ppos_h,
		int n);

};


#endif
