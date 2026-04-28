#include "paramVariables.h"

#include <fstream>
#include "allocateLib.h"
#include "functionLib.h"
#include "macroCUDA.h"
#include "defineLBM.h"
#include "defineCoefficient.h"

#include "Define_user.h"

// public //
void paramVariables::
set (
	const paramMPI		&pmpi,
	const paramCalFlag	&pcalflg,
	const paramDomain	&pdomain
	)
{
	// mpi //
	rank_   = pmpi.rank  ();
	rank_x_ = pmpi.rank_x();
	rank_y_ = pmpi.rank_y();
	rank_z_ = pmpi.rank_z();


	// domain //
	domain_ = pdomain.domain();
	nx_ = pdomain.nx();
	ny_ = pdomain.ny();
	nz_ = pdomain.nz();
	dx_ = pdomain.dx();

	// flag //
	restart_ = pcalflg.restart();
	num_direction_vel_ = NUM_DIRECTION_VEL;
}


void paramVariables::
allocate (
	Variables	*variables,
	defineMemory::FlagHostDevice	flag_memory
	)
{
	if		(flag_memory == defineMemory::Host_Memory) {
		allocate_host   (variables);
	}
	else if	(flag_memory == defineMemory::Device_Memory) {
		allocate_device (variables);
	}
	else {
		std::cout << "error : class paramVariables()" << std::endl;
		exit(-1);
	}
}


void	paramVariables::
init_host (Variables	*variables_h)
{
	if	(restart_ == 0)	{ init_data (variables_h); }
	else				{ read_data (variables_h); }
}


void	paramVariables::
memcpy_Variables (
	      Variables *variablesn,
   	const Variables *variables
	)
{
	const int	nsize     = domain_.nn                   ;
	const int	nsize_lbm = domain_.nn*num_direction_vel_;

	CUDA_SAFE_CALL( cudaMemcpy(variablesn->f_n,   variables->f_n,   sizeof(FLOAT)*(nsize_lbm), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(variablesn->T_n,   variables->T_n,   sizeof(FLOAT)*(nsize    ), cudaMemcpyDefault) );
	CHECK_CUDA_ERROR("CUDA Error\n");
}


void	paramVariables::
output_restart_Variables (Variables *variables)
{
	const int	rank  = rank_;
	const int	nsize     = domain_.nn                   ;
	const int	nsize_lbm = domain_.nn*num_direction_vel_;

	char	name[100];
	sprintf(name, "./init_profile/init-variables-rank%d.dat", rank);


	// fout //
	std::ofstream	fout;
	fout.open(name);
	if(!fout)	{ std::cout << "file is not opened\n"; exit(2); }

	fout.write((char *)variables-> f_n,  sizeof(FLOAT)*(nsize_lbm));
	fout.write((char *)variables-> T_n,  sizeof(FLOAT)*(nsize    ));
	fout.close();
	// fout //
}


// private //
// allocate stVariables.h //
void	paramVariables::
allocate_host   (Variables	*variables)
{
	const int	nsize     = domain_.nn                   ;
	const int	nsize_lbm = domain_.nn*num_direction_vel_;

	allocateLib::new_host   (&variables ->f_n,  nsize_lbm);
	allocateLib::new_host   (&variables ->T_n,  nsize    );
}


void	paramVariables::
allocate_device (Variables	*variables)
{
	const int	nsize     = domain_.nn                   ;
	const int	nsize_lbm = domain_.nn*num_direction_vel_;

	allocateLib::new_device (&variables ->f_n,  nsize_lbm);
	allocateLib::new_device (&variables ->T_n,  nsize    );
}


// initialize //
void	paramVariables::
init_data (Variables *variables)
{
	if (rank_ == 0) { std::cout << "Variables : init_data *********\n"; }

//	const int	nsize     = domain_.nn                   ;
	const int	nsize_lbm = domain_.nn*num_direction_vel_;
	const int	nx = nx_;
	const int       ny = ny_;
	const int       nz = nz_;
	const FLOAT	dx = dx_;

	const FLOAT	dtdz0 = user_init::DTDZ_LOW;
	const FLOAT	dtdz1 = user_init::DTDZ_HIGH;
	const int	kzi = user_init::ZHIGH/dx;
//	const int	kz2 = user_init::kz2;
//	FLOAT t_ref;

	functionLib::fillArray(variables->f_n,	1.0,    nsize_lbm);
//	functionLib::fillArray(variables->T_n,	BASE_TEMPERATURE,  nsize    );
	functionLib::fillArrayT0(variables->T_n,  BASE_TEMPERATURE, dtdz0, dtdz1, kzi, nx, ny, nz, dx);
//	functionLib::fillArrayF0(variables->f_n,  BASE_TEMPERATURE, dtdz0, dtdz1, kzi, nx, ny, nz, dx);

int nxy = nx * ny;
for(int k=0;k<nz;k++){
	int id_z = 10 + nxy * nz;
	std::cout<<k<<" "<<variables->T_n[id_z]<<std::endl;
}


}


void	paramVariables::
read_data (Variables *variables)
{
	if (rank_ == 0) { std::cout << "Variables : read_data *********\n"; }

	const int	rank = rank_;
	const int	nsize     = domain_.nn                   ;
	const int	nsize_lbm = domain_.nn*num_direction_vel_;

	char	name[100];
	sprintf(name, "./init_profile/init-variables-rank%d.dat", rank);
//	sprintf(name, "./link_init_profile/init-variables-rank%d.dat", rank);


	// fin //
	std::ifstream fin;
	fin.open(name, std::ios::in | std::ios::binary);
	if(!fin) { std::cout << "Variables : file is not opened\n"; exit(2); }

	fin.read((char *)variables ->f_n,  sizeof(FLOAT)*(nsize_lbm));
	fin.read((char *)variables ->T_n,  sizeof(FLOAT)*(nsize    ));

	fin.close();
	// fin //
}


// paramVariables //
