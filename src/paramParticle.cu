#include "paramParticle.h"

#include <iostream>
#include <fstream>
#include "allocateLib.h"
#include "option_parser.h"


// public *****
void paramParticle::
set (
	char		*program_name,
	int			argc,
   	char		*argv[], 
	Particle	*particle_h, 
	Particle	*particle_d
	)
{
	// option paser *****
	const char	program_args[] = "[options...]";

	// OptionParser creates the first line of help messages.
	OptionParser parser(program_name, program_args);

	// Parse argv.
	// This function must be executed before other functions are called.
	int		ret = parser.parse_args(argc, argv);

	// Parse error. Exit. Print usage automatically.
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }

	// Init CalFlag *****
	particle_h->pfrag.prestart  = parser.prestart();

	particle_h->pfrag.pout_step = parser.pout();
	particle_h->pfrag.gen_step  = parser.generate_step();
	
	particle_h->pgrid.num_particle_max = parser.particle();
	particle_h->pgrid.parray_end       = 0;

	// Particle
	const int	nn = particle_h->pgrid.num_particle_max;
	num_particle_max_ = nn;


	// particle position
	allocate_ParticlePosition_All(&particle_h->ppos, &particle_d->ppos,  nn);

	init_ParticlePosition(particle_h->ppos,  nn);

	// Particle
	cudaMemcpy_ParticleCalFlag (nn, particle_d->pfrag, particle_h->pfrag);
	cudaMemcpy_ParticleGrid    (particle_d->pgrid, particle_h->pgrid);
	cudaMemcpy_ParticlePosition(nn, particle_d->ppos,  particle_h->ppos);


	// buffer *****
	const int	buff_ratio = 10;
	const int	nn_buff = nn/buff_ratio;

	particle_h->pgrid_buff_s.num_particle_max = nn_buff;
	particle_h->pgrid_buff_r.num_particle_max = nn_buff;

	particle_h->pgrid_buff_s.parray_end       = 0;
	particle_h->pgrid_buff_r.parray_end       = 0;

	// particle position
	allocate_ParticlePosition_All(&particle_h->ppos_buff_s, &particle_d->ppos_buff_s,  nn_buff);
	allocate_ParticlePosition_All(&particle_h->ppos_buff_r, &particle_d->ppos_buff_r,  nn_buff);

	init_ParticlePosition(particle_h->ppos_buff_s,  nn_buff);
	init_ParticlePosition(particle_h->ppos_buff_r,  nn_buff);

	// Particle
	cudaMemcpy_ParticleGrid    (particle_d->pgrid_buff_s, particle_h->pgrid_buff_s);
	cudaMemcpy_ParticleGrid    (particle_d->pgrid_buff_r, particle_h->pgrid_buff_r);

	cudaMemcpy_ParticlePosition(nn_buff, particle_d->ppos_buff_s,  particle_h->ppos_buff_s);
	cudaMemcpy_ParticlePosition(nn_buff, particle_d->ppos_buff_r,  particle_h->ppos_buff_r);
	// buffer *****
}


// memcpy //
void paramParticle::
cudaMemcpy_ParticleCalFlag(
	int n, 
	      ParticleCalFlag &pfragn,
   	const ParticleCalFlag &pfrag
	)
{
	CUDA_SAFE_CALL( cudaMemcpy(&pfragn,   &pfrag,  sizeof(ParticleCalFlag), cudaMemcpyDefault) );
}


void paramParticle::
cudaMemcpy_ParticleGrid (
	      ParticleGrid &pgridn,
	const ParticleGrid &pgrid
	)
{
	pgridn.num_particle_max = pgrid.num_particle_max;
	pgridn.parray_end       = pgrid.parray_end;
}


void paramParticle::
cudaMemcpy_ParticlePosition (
	int n,
	      ParticlePosition *pposn,
	const ParticlePosition *ppos
	)
{
	CUDA_SAFE_CALL( cudaMemcpy(pposn,   ppos,   sizeof(ParticlePosition)*(n), cudaMemcpyDefault) );
}


// restart
void paramParticle::
Read_ParticlePosition (
	Particle *particle_h,
	Particle *particle_d
	)
{
	if (rank_ == 0) {
		std::cout << "---------------\n";
		std::cout << "restart particle position\n";
		std::cout << "---------------\n";
	}

	char	name[100];
	sprintf(name, "./init_profile_particle/init-rank%d.dat", rank_);

	std::ifstream fin;
	fin.open(name, std::ios::in | std::ios::binary);

	fin.read((char *)&particle_h->pgrid.parray_end,	sizeof(int));

	const int	n = particle_h->pgrid.parray_end;
	fin.read((char *) particle_h->ppos,				sizeof(ParticlePosition)*(n));

	fin.close();


	cudaMemcpy_ParticleGrid       (particle_d->pgrid, particle_h->pgrid);
	cudaMemcpy_ParticlePosition(n, particle_d->ppos,  particle_h->ppos);
}


void paramParticle::
Output_Restart_ParticlePosition (Particle *particle_h)
{
	char	name[100];
	sprintf(name, "./init_profile_particle/init-rank%d.dat", rank_);

	std::ofstream fout;
	fout.open(name);

	const int	n = particle_h->pgrid.parray_end;

	fout.write((char *)&particle_h->pgrid.parray_end,	sizeof(int));
	fout.write((char *) particle_h->ppos,				sizeof(ParticlePosition)*(n));

	fout.close();
}


// private
void paramParticle::
allocate_ParticlePosition_All (
	ParticlePosition **ppos_h,
	ParticlePosition **ppos_d,
	int	n
	)
{
	*ppos_h = (ParticlePosition *)malloc(sizeof(ParticlePosition) * n);
	CUDA_SAFE_CALL( cudaMalloc((void **)ppos_d, sizeof(ParticlePosition) * n) );
}


void paramParticle::
allocate_ParticlePosition_Pinned_All (
	ParticlePosition **ppos_h,
	ParticlePosition **ppos_d,
	int n)
{
	CUDA_SAFE_CALL( cudaMallocHost((void **)ppos_h, sizeof(ParticlePosition) * n) );
	CUDA_SAFE_CALL( cudaMalloc    ((void **)ppos_d, sizeof(ParticlePosition) * n) );
}


void paramParticle::
init_ParticlePosition(
	ParticlePosition *ppos_h,
	int n)
{
	for (int i=0; i<n; i++) {
		ppos_h[i].x_p = 0.0;
		ppos_h[i].y_p = 0.0;
		ppos_h[i].z_p = 0.0;

		ppos_h[i].vel_p = 0.0;

		// (YOKOUCHI 2020)
		ppos_h[i].u_sgs = 0.0;
		ppos_h[i].v_sgs = 0.0;
		ppos_h[i].w_sgs = 0.0;

		ppos_h[i].state_p   = PARTICLE_NA;
	}
}


// paramParticle.cu //
