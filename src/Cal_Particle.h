#ifndef CALPARTICLES_H_
#define CALPARTICLES_H_


#include "definePrecision.h"
#include "VariablesParticle.h"

#include "stCalFlag.h"
#include "stVariables.h"
#include "stBasisVariables.h"
#include "stFluidProperty.h"
#include "paramDomain.h"
#include "paramMPI.h"

// (YOKOUCHI 2020)
#include "paramStress.h"
#include <curand_kernel.h>


class	CalParticle {
private:
	int		rank_;

	int		gen_step_;
	int		renum_step_;
	int		thread_length_;

	// domain //
	Domain	domain_;

private:
	CalParticle () {}

public:	
	explicit
	CalParticle (
		      int			argc,
   		      char			*argv[], 
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain,
		      Particle		*particle
		)
	{
		set (
			argc, argv,
			pmpi,
			pdomain,
			particle
			);
	}

	~CalParticle () {}

public:	
	void
	set (
		      int			argc,
		      char			*argv[],
		const paramMPI		&pmpi,
		const paramDomain	&pdomain,
		      Particle		*particle
		);


	void 
	particleAdvection (
		int				argc,
   		char			*argv[], 
		int				t,
		Domain			&cdo, 
		Particle		*particle_h,
		Particle		*particle_d,
		BasisVariables	*cbq_d,
		FluidProperty	*cfp_d
		);

	void 
	particleAdvection_LSM (
		int				argc,
   		char			*argv[], 
		int				t,
		Domain			&cdo, 
		Particle		*particle_h,
		Particle		*particle_d,
		BasisVariables	*cbq_d,
		FluidProperty	*cfp_d,
		Stress		*str_d
		);

private:
	// 粒子の移流 //
	void
	gpu_advection (
		const ParticleGrid		&pgrid,
		      ParticlePosition	*ppos,
		const BasisVariables	*const cbq
		);

	// (YOKOUCHI 2020)	
	void
	gpu_advection_LSM (
		const ParticleGrid		&pgrid,
	      	      FluidProperty	*cfp,	
		      ParticlePosition	*ppos,
		const BasisVariables	*const cbq,
		const Stress		*const str,
		      int		t,
		      int		nn
		);
	

	// 計算外領域判定 //
	void
	boundary_flag (
		ParticleGrid		&pgrid,
		ParticlePosition	*ppos);


};

// (YOKOUCHI 2020)
void genRandNum (
//	FLOAT		*rnd_d,
	curandState	*state,
	const int	nn,
	int		rank_,
	int		t);

void particleRead_uniform ( 
	int		&pstart,
	int		&pnum,
	int		num_g[],
	FLOAT		point_g[],
	FLOAT		vec_g[],
	int		rank_);

void particleRead_LSM (
	int		&pstart,
	int 		&pnum,
	int		num_g[],
	FLOAT		point_g[],
	FLOAT		vec_g[],
	int		rank_);

#endif
