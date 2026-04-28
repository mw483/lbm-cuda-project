#include "paramBasisVariables.h"

#include <fstream>
#include <random>
#include "allocateLib.h"
#include "functionLib.h"
#include "macroCUDA.h"
#include "defineReferenceVel.h"

#include "Define_user.h"
#include "defineCoefficient.h"


void	paramBasisVariables::
set (
	const paramMPI		&pmpi, 
	const paramCalFlag	&pcalflg,
	const paramDomain	&pdomain
	)
{
	// mpi //
	rank_ = pmpi.rank();


	// domain //
	nx_ = pdomain.nx();
	ny_ = pdomain.ny();
	nz_ = pdomain.nz();
	nn_ = pdomain.nn();


	// lbm //
	cfl_ref_ = pdomain.cfl_ref();
	vel_cfl_ref_ = pdomain.vel_cfl_ref();	// MOD2019a

	// flag //
	restart_ = pcalflg.restart();
}


void	paramBasisVariables::
allocate (
	BasisVariables	*basisv,
	defineMemory::FlagHostDevice	flag_memory
	)
{
	if		(flag_memory == defineMemory::Host_Memory) {
		allocate_host   (basisv);
	}
	else if	(flag_memory == defineMemory::Device_Memory) {
		allocate_device (basisv);
	}
	else {
		std::cout << "error : class paramBasisVariables()" << std::endl;
		exit(-1);
	}
}


void	paramBasisVariables::
init_host (BasisVariables	*basisv)
{
	if (restart_ == 0)	{ init_data (basisv); }
	else				{ read_data (basisv); }
}


void	paramBasisVariables::
memcpy_BasisVariables (
	      BasisVariables *basisvn,
	const BasisVariables *basisv
	)
{
	const int	nsize = nn_;

	CUDA_SAFE_CALL( cudaMemcpy(basisvn->r_n,   basisv->r_n,   sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(basisvn->u_n,   basisv->u_n,   sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(basisvn->v_n,   basisv->v_n,   sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(basisvn->w_n,   basisv->w_n,   sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CHECK_CUDA_ERROR("CUDA Error\n");
}


void	paramBasisVariables::
output_restart_BasisVariables (BasisVariables *basisv)
{
	const int	nsize = nn_;
	const int	rank  = rank_;

	char	name[100];
	sprintf(name, "./init_profile/init-basisvariables-rank%d.dat", rank);


	// fout //
	std::ofstream	fout;
	fout.open(name);
	if(!fout)	{ std::cout << "basisvariables : file is not opened\n"; exit(2); }

	fout.write((char *)basisv->r_n,  sizeof(FLOAT)*(nsize));
	fout.write((char *)basisv->u_n,  sizeof(FLOAT)*(nsize));
	fout.write((char *)basisv->v_n,  sizeof(FLOAT)*(nsize));
	fout.write((char *)basisv->w_n,  sizeof(FLOAT)*(nsize));

	fout.close();
	// fout //
}


// private //
// allocate stBasisVariables.h //
void paramBasisVariables::
allocate_host   (BasisVariables	*basisv)
{
	const int	nsize = nn_;
	allocateLib::new_host   (&basisv->r_n,  nsize);
	allocateLib::new_host   (&basisv->u_n,  nsize);
	allocateLib::new_host   (&basisv->v_n,  nsize);
	allocateLib::new_host   (&basisv->w_n,  nsize);
}


void paramBasisVariables::
allocate_device   (BasisVariables	*basisv)
{
	const int	nsize = nn_;
	allocateLib::new_device (&basisv->r_n,  nsize);
	allocateLib::new_device (&basisv->u_n,  nsize);
	allocateLib::new_device (&basisv->v_n,  nsize);
	allocateLib::new_device (&basisv->w_n,  nsize);
}


void	paramBasisVariables::
init_data (BasisVariables *basisv)
{
	if (rank_ == 0) { std::cout << "BasisVariables : init_data *********\n"; }

	const int	nx = nx_,
				ny = ny_,
				nz = nz_;
	std::cout << "check cref "<< cfl_ref_ <<"  "<< vel_cfl_ref_ << std::endl;
	const FLOAT init_u = vel_cfl_ref_==0 ? 0.0 : cfl_ref_;	// MOD2019a

	for (int k=0; k<nz; k++) {
		FLOAT height = ((FLOAT)k-3+0.5) * 20.0 / (FLOAT)(nz-6);
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;
				basisv->r_n[id] = 1.0;
				basisv->u_n[id] = init_u;	// MOD2019a
				//basisv->u_n[id] = 0.01 * 4.0 * height/20.0 * (1-height/20.0)  * cfl_ref_ / vel_cfl_ref_;	// MOD2019a
				basisv->v_n[id] = 0.0;
				basisv->w_n[id] = 0.0;
			}
		}
	}

	// generate disturbance MOD 2019 by Ishibashi
	if(user_flags::flg_disturbance == 1){
		std::random_device seed_gen;
		std::mt19937 engine(seed_gen());
		//std::normal_distribution<> ndist(0.0, user_init::dist*init_u);
		FLOAT tmp = 0.0;
		if(init_u==0.0){
			tmp = 1e-5;	
		}
		std::uniform_real_distribution<> ndist(-user_init::dist*(init_u+tmp), user_init::dist*(init_u+tmp));
		float dist_r = 1;
		for (int k=0; k<nz; k++) {
			//dist_r = (nz - k)/static_cast<float>(nz);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					const int	id = i + nx*j + nx*ny*k;
					basisv->u_n[id] += dist_r * ndist(engine);	// MOD2019a
					basisv->v_n[id] += dist_r * ndist(engine);
					basisv->w_n[id] += dist_r * ndist(engine);
				}
			}
		}
	}

}


void	paramBasisVariables::
read_data (BasisVariables *basisv)
{
	if (rank_ == 0) { std::cout << "BasisVariables : read_data *********\n"; }

	const int	nn   = nn_;
	const int	rank = rank_;

	char	name[100];
	sprintf(name, "./init_profile/init-basisvariables-rank%d.dat", rank);
//	sprintf(name, "./link_init_profile/init-basisvariables-rank%d.dat", rank);


	// fin //
	std::ifstream fin;
	fin.open(name, std::ios::in | std::ios::binary);
	if(!fin) { std::cout << "basisvariables : file is not opened\n"; exit(2); }

	fin.read((char *)basisv->r_n,  sizeof(FLOAT)*(nn));
	fin.read((char *)basisv->u_n,  sizeof(FLOAT)*(nn));
	fin.read((char *)basisv->v_n,  sizeof(FLOAT)*(nn));
	fin.read((char *)basisv->w_n,  sizeof(FLOAT)*(nn));

	fin.close();
	// fin //
}


// paramBasisVariables //
