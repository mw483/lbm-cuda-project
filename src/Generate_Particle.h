#ifndef GENERATE_PARTICLES_H_
#define GENERATE_PARTICLES_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "Define.h"
#include "Variables.h"
#include "VariablesParticle.h"
#include "Generate_Particle_GPU.h"


void 
generate_particle (
	int		index_num,
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*cfp
	);


void 
generate_particle_box (
	      int			index_num,
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty
	);


void 
generate_particle_box_slice (
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty,
	const int			num_g[],
	const FLOAT			point_g[],
	const FLOAT			vec_g[]
	);

// (YOKOUCHI 2020)
void
generate_particle_LSM (
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty,
	const int		num_g[],
	const FLOAT		point_g[],
	const FLOAT		vec_g[],
	int			t,
	int			pstart
	);
	

void 
generate_particle_sphere (
	int		index_num,
	const Domain		&domain,
	      Particle		*particle,
	      FluidProperty	*fproperty
	);


void
particle_source_box (
	const Domain			&domain,
	      ParticleCalFlag	&pfrag,
		  ParticleGrid		&pgrid,
	      ParticlePosition	*ppos,
	const int	num[],
	const FLOAT	point[],
	const FLOAT	vec_dx[],
	int			&index_num
	);

// (YOKOUCHI 2020)
void
particle_source_LSM (
	const Domain			&domain,
	      ParticleCalFlag		&pfrag,
	      ParticleGrid		&pgrid,
	      ParticlePosition		*ppos,
	const int			num[],
	const FLOAT			point[],
	const FLOAT			vec_dx[],
	int				&index_num,
	int				t,
	int				pstart
	);
void 
particle_source_sphere (
	const Domain		&domain,
	ParticleCalFlag		&pfrag, 
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos,
	const int	num[],
	const FLOAT	point[],
	FLOAT		radius,
	const FLOAT	theta0[],
	const FLOAT	theta1[],
	int			&index_num);


void 
particle_NA_check (
	const Domain		&domain,
	ParticleCalFlag		&pfrag,
	ParticleGrid		&pgrid,
	ParticlePosition	*ppos,
	FluidProperty		*cfp
	);


#endif
