#include "paramStress.h"

#include "allocateLib.h"
#include "functionLib.h"
#include "macroCUDA.h"

// (YOKOUCHI 2020)
#include "Define_user.h"


// public //
void	paramStress::
set (const paramDomain	&pdomain)
{
	nn_ = pdomain.nn();
}


void	paramStress::
init_host (Stress	*stress_h)
{
	init_data (stress_h);
}


void	paramStress::
allocate (
	Stress	*stress,
	defineMemory::FlagHostDevice	flag_memory
	)
{
	if		(flag_memory == defineMemory::Host_Memory) {
		allocate_host   (stress);
	}
	else if	(flag_memory == defineMemory::Device_Memory) {
		allocate_device (stress);
	}
	else {
		std::cout << "error : class paramStress()" << std::endl;
		exit(-1);
	}
}


void paramStress::
memcpy_Stress (Stress *stressn, const Stress *stress)
{
	const int	nsize = nn_;

	CUDA_SAFE_CALL( cudaMemcpy(stressn->vis_sgs,  stress->vis_sgs, sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->Fcs_sgs,  stress->Fcs_sgs, sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );

	CUDA_SAFE_CALL( cudaMemcpy(stressn->Div,      stress->Div,     sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->SS,       stress->SS,      sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->WW,       stress->WW,      sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );

	// force //
	CUDA_SAFE_CALL( cudaMemcpy(stressn->force_x,	stress->force_x,      sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->force_y,	stress->force_y,      sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->force_z,	stress->force_z,      sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	
	// (YOKOUCHI 2020)
	if (user_flags::flg_particle == 1) {
//	CUDA_SAFE_CALL( cudaMemcpy(stressn->vis_sgs_old,stress->vis_sgs_old,	sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );	
	CUDA_SAFE_CALL( cudaMemcpy(stressn->TKE_sgs,	stress->TKE_sgs,	sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->TKE_sgs_old,stress->TKE_sgs_old,	sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->u_m,	stress->u_m,		sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->v_m,	stress->v_m,		sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(stressn->w_m,	stress->w_m,		sizeof(FLOAT)*(nsize), cudaMemcpyDefault) );
}

	CHECK_CUDA_ERROR("CUDA Error\n");
}


// private //
// allocate stStress.h //
void paramStress::
allocate_host   (Stress	*stress)
{
	const int	nsize = nn_;

	allocateLib::new_host   (&stress->vis_sgs,	nsize);
	allocateLib::new_host   (&stress->Fcs_sgs,	nsize);

	allocateLib::new_host   (&stress->Div,		nsize);
	allocateLib::new_host   (&stress->SS,		nsize);
	allocateLib::new_host   (&stress->WW,		nsize);

	allocateLib::new_host   (&stress->force_x,	nsize);
	allocateLib::new_host   (&stress->force_y,	nsize);
	allocateLib::new_host   (&stress->force_z,	nsize);

	// (YOKOUCHI 2020)
	if (user_flags::flg_particle == 1) {
//	allocateLib::new_host	(&stress->vis_sgs_old,	nsize);
	allocateLib::new_host	(&stress->TKE_sgs,	nsize);	
	allocateLib::new_host	(&stress->TKE_sgs_old,	nsize);	
	allocateLib::new_host	(&stress->u_m,		nsize);	
	allocateLib::new_host	(&stress->v_m,		nsize);	
	allocateLib::new_host	(&stress->w_m,		nsize);	
	}

}


void paramStress::
allocate_device   (Stress	*stress)
{
	const int	nsize = nn_;

	allocateLib::new_device (&stress->vis_sgs,	nsize);
	allocateLib::new_device (&stress->Fcs_sgs,	nsize);

	allocateLib::new_device (&stress->Div,		nsize);
	allocateLib::new_device (&stress->SS, 		nsize);
	allocateLib::new_device (&stress->WW, 		nsize);

	allocateLib::new_device (&stress->force_x,	nsize);
	allocateLib::new_device (&stress->force_y,	nsize);
	allocateLib::new_device (&stress->force_z,	nsize);

	// (YOKOUCHI 2020)
	if (user_flags::flg_particle == 1) {
//	allocateLib::new_device (&stress->vis_sgs_old,	nsize);
	allocateLib::new_device (&stress->TKE_sgs,	nsize);
	allocateLib::new_device (&stress->TKE_sgs_old,	nsize);
	allocateLib::new_device (&stress->u_m,		nsize);
	allocateLib::new_device (&stress->v_m,		nsize);
	allocateLib::new_device (&stress->w_m,		nsize);
	}		

}


// initializea //
void	paramStress::
init_data (Stress *stress)
{
	const int	nsize = nn_;

	functionLib::fillArray(stress->vis_sgs,	0.0,    nsize);
	functionLib::fillArray(stress->Fcs_sgs,	0.0,    nsize);

	functionLib::fillArray(stress->Div,	0.0,    nsize);
	functionLib::fillArray(stress->SS,	0.0,    nsize);
	functionLib::fillArray(stress->WW,	0.0,    nsize);

	functionLib::fillArray(stress->force_x,	0.0,    nsize);
	functionLib::fillArray(stress->force_y,	0.0,    nsize);
	functionLib::fillArray(stress->force_z,	0.0,    nsize);

	// (YOKOUCHI 2020)
	if (user_flags::flg_particle == 1) {
//	functionLib::fillArray(stress->vis_sgs_old, 0.0,nsize);
	functionLib::fillArray(stress->TKE_sgs,     0.0,nsize);
	functionLib::fillArray(stress->TKE_sgs_old, 0.0,nsize);
	functionLib::fillArray(stress->u_m, 	    0.0,nsize);
	functionLib::fillArray(stress->v_m, 	    0.0,nsize);
	functionLib::fillArray(stress->w_m, 	    0.0,nsize);
	}

}



// paramStress //
