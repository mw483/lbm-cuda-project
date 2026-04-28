#include "paramFluidProperty.h"

#include <fstream>
#include "allocateLib.h"
#include "functionLib.h"
#include "macroCUDA.h"

#include "defineBoundaryFlag.h"
#include "defineReferenceVel.h"
#include "defineCoefficient.h"

#include "Define_user.h"

// public //
void paramFluidProperty::
set (
	const paramMPI		&pmpi, 
	const paramDomain	&pdomain
	)
{
	// mpiinfo //
	mpiinfo_ = pmpi.mpiinfo();


	// domain //
	domain_ = pdomain.domain();
}


void	paramFluidProperty::
allocate (
	FluidProperty	*fluid,
	defineMemory::FlagHostDevice	flag_memory
	)
{
	if		(flag_memory == defineMemory::Host_Memory) {
		allocate_host   (fluid);
	}
	else if	(flag_memory == defineMemory::Device_Memory) {
		allocate_device (fluid);
	}
	else {
		std::cout << "error : class paramFluidProperty()" << std::endl;
		exit(-1);
	}
}


void	paramFluidProperty::
init_host (FluidProperty	*fluid)
{
if (mpiinfo_.rank == 0) { std::cout << "pfluid  initViscosity" << std::endl; }
	initViscosity (fluid);

//	setBounaryViscosity      (fluid);
if (mpiinfo_.rank == 0) { std::cout << "pfluid  setBounaryViscosity_city" << std::endl; }
	setBounaryViscosity_city (fluid);

if (mpiinfo_.rank == 0) { std::cout << "pfluid  init_thermal_flux" << std::endl; }
	init_thermal_flux (fluid);

if (mpiinfo_.rank == 0) { std::cout << "pfluid  END init_thermal_flux" << std::endl; }
	MPI_Barrier(MPI_COMM_WORLD);
}


void	paramFluidProperty::
read_global_map (FluidProperty	*fluid)
{
	if (mpiinfo_.rank == 0) { std::cout << __PRETTY_FUNCTION__ << std::endl; }

//	const FLOAT	z_domain_min = -2.0;
//	const FLOAT	z_domain_min = domain_.zg_min;
//	const FLOAT	z_domain_min = -2.0;

//	const FLOAT	lbm_scale    =  2.0;
	const FLOAT	lbm_scale    = domain_.dx;	//MOD2018
	const FLOAT	z_domain_min = -lbm_scale;	//MOD2018

//std::cout<<"check  !!!!!!! "<<lbm_scale<<std::endl;
//std::cout<<"check  !!!!!!! "<<z_domain_min<<std::endl;
	read_map (
		fluid->id_obs,
		fluid->l_obs,
		z_domain_min,
		lbm_scale
		);
//	setPoiseuille (
//		fluid->id_obs,
//		fluid->l_obs
//		);
//	setSingleCube (
//		fluid->id_obs,
//		fluid->l_obs
//		);
	MPI_Barrier(MPI_COMM_WORLD);


	// status
	set_status_flag (
		fluid->status,
	   	fluid->l_obs
		);
	MPI_Barrier(MPI_COMM_WORLD);
}


void	paramFluidProperty::
memcpy_FluidProperty (
	      FluidProperty *      fluidn,
   	const FluidProperty *const fluid
	)
{
	if (mpiinfo_.rank == 0) { std::cout << __PRETTY_FUNCTION__ << std::endl; }
 
	const int	nn = domain_.nn;

	fluidn->vis0_lbm = fluid->vis0_lbm;

	// fluid or solid //
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->status,		fluid->status,		sizeof(char )*(nn),	cudaMemcpyDefault) );

	// viscosity //
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->vis_lbm,		fluid->vis_lbm,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );

	// obstacle //
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->id_obs,		fluid->id_obs,		sizeof(int  )*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->l_obs,		fluid->l_obs,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );

	CUDA_SAFE_CALL( cudaMemcpy(fluidn->u_obs,		fluid->u_obs,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->v_obs,		fluid->v_obs,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->w_obs,		fluid->w_obs,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );


	CUDA_SAFE_CALL( cudaMemcpy(fluidn->hflux_w,		fluid->hflux_w,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->hflux_e,		fluid->hflux_e,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->hflux_s,		fluid->hflux_s,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->hflux_n,		fluid->hflux_n,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CUDA_SAFE_CALL( cudaMemcpy(fluidn->hflux_r,		fluid->hflux_r,		sizeof(FLOAT)*(nn),	cudaMemcpyDefault) );
	CHECK_CUDA_ERROR("CUDA Error\n");
}


// private //
void paramFluidProperty::
allocate_host   (FluidProperty	*fluid_h) 
{
	const int	nsize = domain_.nn;

	// fluid or solid //
	allocateLib::new_host   (&fluid_h->status,		nsize);

	// viscosity //
	allocateLib::new_host   (&fluid_h->vis_lbm,		nsize);

	// obstacle //
	allocateLib::new_host   (&fluid_h->id_obs, 		nsize);
	allocateLib::new_host   (&fluid_h->l_obs, 		nsize);

	allocateLib::new_host   (&fluid_h->u_obs, 		nsize);
	allocateLib::new_host   (&fluid_h->v_obs, 		nsize);
	allocateLib::new_host   (&fluid_h->w_obs, 		nsize);

	allocateLib::new_host   (&fluid_h->hflux_w, 		nsize);
	allocateLib::new_host   (&fluid_h->hflux_e, 		nsize);
	allocateLib::new_host   (&fluid_h->hflux_s, 		nsize);
	allocateLib::new_host   (&fluid_h->hflux_n, 		nsize);
	allocateLib::new_host   (&fluid_h->hflux_r, 		nsize);


	// fill //
	// fluid or solid //
	functionLib::fillArray (fluid_h->status,	STATUS_FLUID,	nsize);

	// viscosity //
	functionLib::fillArray (fluid_h->vis_lbm,	0.0,		nsize);

	// obstacle //
	functionLib::fillArray (fluid_h->id_obs,	 -1,		nsize);
	functionLib::fillArray (fluid_h->l_obs,		-10.0,		nsize);

	functionLib::fillArray (fluid_h->u_obs,		  0.0,		nsize);
	functionLib::fillArray (fluid_h->v_obs,		  0.0,		nsize);
	functionLib::fillArray (fluid_h->w_obs,		  0.0,		nsize);

	functionLib::fillArray (fluid_h->hflux_w,		  0.0,		nsize);
	functionLib::fillArray (fluid_h->hflux_e,		  0.0,		nsize);
	functionLib::fillArray (fluid_h->hflux_s,		  0.0,		nsize);
	functionLib::fillArray (fluid_h->hflux_n,		  0.0,		nsize);
	functionLib::fillArray (fluid_h->hflux_r,		  0.0,		nsize);
}


void paramFluidProperty::
allocate_device (FluidProperty	*fluid_d)
{
	const int	nsize = domain_.nn;

	// fluid or solid //
	allocateLib::new_device (&fluid_d->status,		nsize);

	// viscosity //
	allocateLib::new_device (&fluid_d->vis_lbm,		nsize);

	// obstacle //
	allocateLib::new_device (&fluid_d->id_obs,		nsize);
	allocateLib::new_device (&fluid_d->l_obs,		nsize);

	allocateLib::new_device (&fluid_d->u_obs,		nsize);
	allocateLib::new_device (&fluid_d->v_obs,		nsize);
	allocateLib::new_device (&fluid_d->w_obs,		nsize);

	allocateLib::new_device (&fluid_d->hflux_w,		nsize);
	allocateLib::new_device (&fluid_d->hflux_e,		nsize);
	allocateLib::new_device (&fluid_d->hflux_s,		nsize);
	allocateLib::new_device (&fluid_d->hflux_n,		nsize);
	allocateLib::new_device (&fluid_d->hflux_r,		nsize);
}


void paramFluidProperty::
initViscosity (FluidProperty	*fluid)
{
	const int	nx = domain_.nx,
				ny = domain_.ny,
				nz = domain_.nz;

	const FLOAT	dx    = domain_.dx;
	const FLOAT	c_ref = domain_.c_ref;

	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				fluid->status[id] = STATUS_FLUID;

				// channel
//				fluid->vis_lbm[id] = domain_.n0/(cdo->Re_num)/(C_REF);
//				fluid->vis0_lbm    = domain_.n0/(cdo->Re_num)/(C_REF);

				fluid->vis_lbm[id] = coefficient::KVIS_AIR / (dx*c_ref);
				fluid->vis0_lbm    = coefficient::KVIS_AIR / (dx*c_ref);
			}
		}
	}
}


void paramFluidProperty::
setBounaryViscosity (FluidProperty	*fluid)
{
	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;

	const int	halo = domain_.halo;

	const int	nxg = domain_.nxg;
	const int	nyg = domain_.nyg;
	const int	nzg = domain_.nzg;

	const int	rank_x = mpiinfo_.rank_x;
	const int	rank_y = mpiinfo_.rank_y;
	const int	rank_z = mpiinfo_.rank_z;


//	const int	n_bc = 20;
	const int	n_bc_i = 20;
	const int	n_bc_j = 20;
//	const int	n_bc_k = 20;
	const int	n_bc_k = 10;

//	const FLOAT	coef_vis_bc = 15.0;	// vis = vis0 * vis_bc
//	const FLOAT	coef_vis_bc = 50.0;	// vis = vis0 * vis_bc
//	const FLOAT	coef_vis_i = 50.0;
	const FLOAT	coef_vis_i = 1.0;
//	const FLOAT	coef_vis_j = 50.0;
	const FLOAT	coef_vis_j = 1.0;
//	const FLOAT	coef_vis_k = 50.0;
	const FLOAT	coef_vis_k = 100.0;

	// boundary viscosity //
	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				const int	ig = (i - halo) + rank_x*(nx - 2*halo);
				const int	jg = (j - halo) + rank_y*(ny - 2*halo);
				const int	kg = (k - halo) + rank_z*(nz - 2*halo);

				const int	i_bc = n_bc_i - min( min( abs(ig), abs(ig - nxg) ), n_bc_i );
				const int	j_bc = n_bc_j - min( min( abs(jg), abs(jg - nyg) ), n_bc_j );
				const int	k_bc = n_bc_k - min( min( abs(kg), abs(kg - nzg) ), n_bc_k );

//				const int	id_bc = max( i_bc, max( j_bc, k_bc ) );
//				const FLOAT	weight_bc = (FLOAT)id_bc/(FLOAT)n_bc;
				const int	weight_bc_i = (FLOAT)i_bc/(FLOAT)n_bc_i;
				const int	weight_bc_j = (FLOAT)j_bc/(FLOAT)n_bc_j;
				const int	weight_bc_k = (FLOAT)k_bc/(FLOAT)n_bc_k;

				const FLOAT	vis0   = fluid->vis_lbm[id];
//				const FLOAT	vis_bc = vis0*coef_vis_bc;

				const FLOAT	vis_i = vis0*(1.0-weight_bc_i + coef_vis_i*weight_bc_i);
				const FLOAT	vis_j = vis0*(1.0-weight_bc_j + coef_vis_j*weight_bc_j);
				const FLOAT	vis_k = vis0*(1.0-weight_bc_k + coef_vis_k*weight_bc_k);
				fluid->vis_lbm[id] = max( vis_i, max(vis_j, vis_k) );
			}
		}
	}
}


void paramFluidProperty::
setBounaryViscosity_city (FluidProperty	*fluid)
{
	const int	nx = domain_.nx,
				ny = domain_.ny,
				nz = domain_.nz;

	const int	halo = domain_.halo;

//	const int	nbc    = 125;
	const int	nbc    = 10;
	const int	nbcz   = 100;

	const FLOAT	vis    = (fluid->vis0_lbm);
//	const FLOAT	vis_bc = vis*1.0;
	const FLOAT	vis_bc = vis*100.0;

	// boundary viscosity
	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				FLOAT	weight_bc = 0.0;

				const int	nxg = domain_.nxg;
				const int	nyg = domain_.nyg;
				const int	nzg = domain_.nzg;
				const int	ig = mpiinfo_.rank_x*(nx - 2*halo) + i;
				const int	jg = mpiinfo_.rank_y*(ny - 2*halo) + j;
				const int	kg = mpiinfo_.rank_z*(nz - 2*halo) + k;


				FLOAT	id_weight_x[2];
				FLOAT	id_weight_y[2];
				FLOAT	id_weight_z[2];

				id_weight_x[0] = fmax((nbc - ig),           0.0) / nbc;
				id_weight_y[0] = fmax((nbc - jg),           0.0) / nbc;
				id_weight_z[0] = fmax((nbcz - kg),           0.0) / nbc;	//MOD 2018

				id_weight_x[1] = fmax(ig - fabs(nxg - nbc), 0.0) / nbc;
				id_weight_y[1] = fmax(jg - fabs(nyg - nbc), 0.0) / nbc;
				id_weight_z[1] = fmax(kg - fabs(nzg - nbcz), 0.0) / nbc;	//MOD 2018

				weight_bc = fmax (id_weight_x[0], weight_bc);
				weight_bc = fmax (id_weight_x[1], weight_bc);

//				weight_bc = fmax (id_weight_y[0], weight_bc);
//				weight_bc = fmax (id_weight_y[1], weight_bc);

//				weight_bc = fmax (id_weight_z[0], weight_bc);
				weight_bc = fmax (id_weight_z[1], weight_bc);

				weight_bc = fmax ((FLOAT)0.0, weight_bc);
				weight_bc = fmin ((FLOAT)1.0, weight_bc);

				// viscosity //
				fluid->vis_lbm[id] =       weight_bc *vis_bc 
								  + (1.0 - weight_bc)*vis;
			}
		}
	}
}

void paramFluidProperty::setPoiseuille(
		int *id_obs,
		FLOAT *lv
	)
{
	// local //
	const int	halo = domain_.halo;

	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;

	const int	rank_x = mpiinfo_.rank_x;
	const int	rank_y = mpiinfo_.rank_y;
	const int	rank_z = mpiinfo_.rank_z;

	// map global > status

	for (int k=0; k<nz; k++) {	// k : height //
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;
				if( k<3 || k>nz-4 ){
					lv[id] =  1.0;
				}else{
					lv[id] = -1.0;
				}
				id_obs[id] = 0;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void paramFluidProperty::setSingleCube(
		int *id_obs,
		FLOAT *lv
		){
	const int	halo = domain_.halo;
	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;
	const int	rank_x = mpiinfo_.rank_x;
	const int	rank_y = mpiinfo_.rank_y;
	const int	rank_z = mpiinfo_.rank_z;
	const int	block_H = (nz - 2*halo) / 2 - 2;

	// vertical boundary
	for (int k=0; k<nz; k++) {	// k : height //
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;
				if( k<3 || k>nz-4 ){
					lv[id] =  1.0; // solid
				}else{
					lv[id] = -1.0; // fluid
				}
				id_obs[id] = 0;
			}
		}
	}
	// single block
	for (int k=0; k<block_H; k++){
		for (int j=3.0*block_H; j<4.0*block_H; j++){
			for (int i=6*block_H; i<7*block_H; i++){
				const int id = i + nx*j + nx*ny*(k+2);
				lv[id] = 1.0; // solid
			}
		}
	}
	//for (int k=0; k<block_H; k++){
	//	for (int j=2.5*block_H; j<3.5*block_H; j++){
	//		for (int i=3*block_H; i<4*block_H; i++){
	//			const int id = i + nx*j + nx*ny*(k+2);
	//			lv[id] = 1.0; // solid
	//		}
	//	}
	//}
	MPI_Barrier(MPI_COMM_WORLD);
}

	
void paramFluidProperty::
read_map (
	      int	*id_obs,
	      FLOAT	*lv,
	const FLOAT	z_domain_min,
	const FLOAT	lbm_scale
	)
{
	if (mpiinfo_.rank == 0) { std::cout << __PRETTY_FUNCTION__ << std::endl; }

	const int	nxg = domain_.nxg;
	const int	nyg = domain_.nyg;

	int		resolution[2];

	// global //
	MPI_Barrier(MPI_COMM_WORLD);
        const int       hl = domain_.halo;						//MOD 2018
		float	*height_map_g = new float[(nxg+2*hl)*(nyg+2*hl)];			//MOD 2018
	for (int i=0; i<(nxg+2*hl)*(nyg+2*hl); i++) { height_map_g[i] = 0; }		//MOD 2018

	// fin
	if (mpiinfo_.rank == 0) {
		char name[100];

		std::ifstream fin;
		sprintf(name, "./map/map_02_full_roughness.dat");

		std::cout << "fin open (map)\n";
		fin.open(name, std::ios::in);
		if(!fin) { std::cout << "file is not opened\n"; }

		fin >> resolution[0];
		fin >> resolution[1];

		std::cout << "resolution = " << resolution[0] << ", " << resolution[1] << std::endl;

		// map
//		int		*height_map = new int[resolution[0]*resolution[1]];
		FLOAT	*height_map = new FLOAT[resolution[0]*resolution[1]];

		for (int i=0; i<resolution[0]*resolution[1]; i++) {
			height_map[i] = 0;

			fin >> height_map[i];
//			std::cout << height_map[i] << "\n";
		}
		fin.close();
		std::cout << "fin close\n";

		// map -> map global
		int		map_offset[2] = { 20, 20 };

		// map : center
		if (resolution[0] < nxg)	{ map_offset[0] = (nxg - resolution[0])/2; }
		if (resolution[1] < nyg)	{ map_offset[1] = (nyg - resolution[1])/2; }

		// map > nx,ny
		if (resolution[0] >= nxg)	{ map_offset[0] = 0; }		// mod 2018
		if (resolution[1] >= nyg)	{ map_offset[1] = 0; }		// mod 2018

		int		i_s = map_offset[0],
				i_f = i_s + resolution[0];
		int		j_s = map_offset[1],
				j_f = j_s + resolution[1];

		if (i_f > nxg)	{ i_f = nxg; }
		if (j_f > nyg)	{ j_f = nyg; }


		// deco-boco //
//		const int	deco = 5;
//		for (int k=deco; k<nzg-deco; k++) {
//			for (int j=deco; j<nyg-deco; j++) {
//				int		id_map_g =  j     + nyg * k;
//
//				if ((j%6==0 && k%6==0) || (j%6==1 && k%6==0) || (j%6==0 && k%6==1) || (j%6==1 && k%6==1)) {
//					height_map_g[id_map_g] = 3;
//				}
//			}
//		}


		for (int j=j_s; j<j_f; j++) {
			for (int i=i_s; i<i_f; i++) {
				int		id_map   = (i-i_s) + resolution[0] * (j-j_s);
				int		id_map_g =  i      + nxg * j;

				height_map_g[id_map_g] = height_map[id_map];
			}
		}

		delete [] height_map;
	} 
	MPI_Barrier(MPI_COMM_WORLD);

	// resolution, height
	MPI_Bcast(&resolution[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&resolution[1], 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(height_map_g, (nxg*nyg), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	// local //
	const int	halo = domain_.halo;

	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;

	const int	rank_x = mpiinfo_.rank_x;
	const int	rank_y = mpiinfo_.rank_y;
	const int	rank_z = mpiinfo_.rank_z;


	// map global > status
//	const FLOAT	lbm_scale = 2.0; // 2m resolution //

	for (int k=0; k<nz; k++) {	// k : height //
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int	id = i + nx*j + nx*ny*k;

				int	ig = rank_x*(nx-2*halo) + (i-halo);
				int	jg = rank_y*(ny-2*halo) + (j-halo);
				int	kg = rank_z*(nz-2*halo) + (k-halo);

				if      (ig < 0    )	{ig = ig + nxg;}
//				else if (ig > nxg-1)	{ig = ig - nxg;}
				if      (jg < 0    )	{jg = jg + nyg;}
//				else if (ig > nyg-1)	{jg = jg - nyg;}


	
				const int	id_map = ig + nxg * jg;
				const FLOAT	height = height_map_g[id_map];

				const FLOAT	lbm_height = kg * lbm_scale + z_domain_min;
				if (lbm_height < height)	{
					lv[id] =  1.0;
				}
//				else if (k > nz-2){
//					lv[id] =  1.0;
//				}
//				else if (lbm_height >= 1900.0 )	{	//MOD2018
//					lv[id] =  1.0;			//MOD2018
//				}					//MOD2018
				else {
					lv[id] = -1.0;
				}
				id_obs[id] = 0;
	
			}
		}
	}
	// outer region //
//	for (int k=0; k<nz; k++) {	// k : height //
//		for (int j=0; j<ny; j++) {
//			for (int i=0; i<nx; i++) {
//				const int	id = i + nx*j + nx*ny*k;
//
//				const int	ig = rank_x*(nx-2*halo) + (i-halo);
//				const int	jg = rank_y*(ny-2*halo) + (j-halo);
//				const int	kg = rank_z*(nz-2*halo) + (k-halo);
//	
//				if (ig < 3) {
//					lv[id] = -1.0;
//				}
//			}
//		}
//	}

	MPI_Barrier(MPI_COMM_WORLD);
	delete [] height_map_g;
}


void paramFluidProperty::
set_status_flag (
	      char	*status,
	const FLOAT	*lv
	)
{
	const int	nn = domain_.nn;

	// wall +
	for (int i=0; i<nn; i++) {
		status[i] = (lv[i] >= 0.0) ? STATUS_WALL : STATUS_FLUID;
	}
}


void paramFluidProperty::
init_thermal_flux (FluidProperty* fluid_h)
{

	if(user_flags::flg_scalar==1)	{	//MOD2019

		const int	rank_x = mpiinfo_.rank_x;
		const int	rank_y = mpiinfo_.rank_y;
		const int	rank_z = mpiinfo_.rank_z;

		const int	nx_ = domain_.nx;		// MOD 2018
    		const int       ny_ = domain_.ny;		// MOD 2018
       		const int       nz_ = domain_.nz;		// MOD 2018

//    std::string  fname;

		char  fname_w[128];  sprintf(fname_w, "input/hflux_x%04dy%04dz%04d_w.dat", rank_x, rank_y, rank_z);
		char  fname_e[128];  sprintf(fname_e, "input/hflux_x%04dy%04dz%04d_e.dat", rank_x, rank_y, rank_z);
   		char  fname_s[128];  sprintf(fname_s, "input/hflux_x%04dy%04dz%04d_s.dat", rank_x, rank_y, rank_z);
   		char  fname_n[128];  sprintf(fname_n, "input/hflux_x%04dy%04dz%04d_n.dat", rank_x, rank_y, rank_z);
   		char  fname_r[128];  sprintf(fname_r, "input/hflux_x%04dy%04dz%04d_r.dat", rank_x, rank_y, rank_z);

//    printf("%s\n", fname_w);
//    printf("%s\n", fname_e);
//    printf("%s\n", fname_s);
//    printf("%s\n", fname_n);
//    printf("%s\n", fname_r);

  		read_thermal_flux(fname_w, fluid_h->hflux_w, nx_, ny_, nz_);	// MOD 2018
  		read_thermal_flux(fname_e, fluid_h->hflux_e, nx_, ny_, nz_);	// MOD 2018
   		read_thermal_flux(fname_s, fluid_h->hflux_s, nx_, ny_, nz_);	// MOD 2018
   		read_thermal_flux(fname_n, fluid_h->hflux_n, nx_, ny_, nz_);	// MOD 2018
   		read_thermal_flux(fname_r, fluid_h->hflux_r, nx_, ny_, nz_);	// MOD 2018
	}	// MOD2019

	if(user_flags::flg_scalar==2)	{	// MOD2019
//		const int       rank_x = mpiinfo_.rank_x;
//		const int	rank_y = mpiinfo_.rank_y;
//		const int	rank_z = mpiinfo_.rank_z;

		const int	nx_ = domain_.nx; 
		const int	ny_ = domain_.ny;
		const int	nz_ = domain_.nz;

//		const FLOAT hf = 0.1;
		const FLOAT hf_ = user_init::hf;
		constant_thermal_flux(fluid_h->hflux_r, hf_, nx_, ny_, nz_);
	}



}


void paramFluidProperty::
read_thermal_flux(std::string fname, FLOAT* hflux, const int nx_, const int ny_, const int nz_)
{
    std::ifstream fin_header;
    fin_header.open("input/header.dat", std::ios::in);

    int  nx, ny, nz;
    float xs, ys, zs;

    fin_header >> nx >> ny >> nz
               >>  xs >>  ys >>  zs;

	if (mpiinfo_.rank == 0) {
        std::cout << __PRETTY_FUNCTION__ << " : " << "read header : nx,ny,nz: xs,ys,zs = " << nx << "," << ny << "," << nz << ":" << xs << "," << ys << "," << zs << std::endl;
    }

    fin_header.close();

    float*  hflux_g;
    hflux_g = new float[nx*ny*nz];

    FILE	*fp = fopen(fname.c_str(), "r");
    fread(hflux_g, sizeof(float), nx*ny*nz, fp);
    fclose(fp);

// check array size	MOD 2018
	if(nx!=nx_) { std::cout<<"Error! mismatch, hflux and subdomain sieze in x"<<std::endl; }
        if(ny!=ny_) { std::cout<<"Error! mismatch, hflux and subdomain sieze in y"<<std::endl; }
        if(nz>nz_) { 
		std::cout<< "Mismatch, hflux and subdomain sieze in z"<<std::endl;
		nz = nz_;
		std::cout<<"nz(hflux) is reduced to nz(subdomain)"<<std::endl;
	}
//std::cout<<"nx, nx_ = "<<nx<<" "<<nx_<<std::endl;
//std::cout<<"ny, ny_ = "<<ny<<" "<<ny_<<std::endl;
//std::cout<<"nz, nz_ = "<<nz<<" "<<nz_<<std::endl;


    for (int k=0; k<nz; k++) {
    for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
        const int id  = i  + nx *j  + nx *ny *k;

	hflux[id] = hflux_g[id];

//	if(hflux[id]>0){std::cout<<i<<' '<<' '<<j<<' '<<k<<' '<<id<<' '<<hflux[id]<<std::endl;}

//	if(k!=1){hflux[id] = 0.0;}
    }
    }
    }

	MPI_Barrier(MPI_COMM_WORLD);		// MOD 2018


	delete [] hflux_g;
}


// MOD2019
void paramFluidProperty::
constant_thermal_flux(FLOAT* hflux, const FLOAT hf,  const int nx_, const int ny_, const int nz_)
{

	int  nx = nx_;
	int  ny = ny_;
	int  nz = nz_;
	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				const int id  = i  + nx *j  + nx *ny *k;
				hflux[id] = hf;
			}
		}
	}
        MPI_Barrier(MPI_COMM_WORLD);

}


//void paramFluidProperty::
//read_thermal_flux_from_global_data(std::string fname, FLOAT* hflux)
//{
//    std::ifstream fin_header;
//    fin_header.open("input/header.dat", std::ios::in);
//
//    int  nxh, nyh, nzh;
//    float xs, ys, zs;
//    const float dx0 = 2.0;
//
//    fin_header >> nxh >> nyh >> nzh
//               >>  xs >>  ys >>  zs;
//
////    float xe, ye, ze;
////    xe = xs + dx0*(nxh - 1);
////    ye = ys + dx0*(nyh - 1);
////    ze = zs + dx0*(nzh - 1);
//
//	if (mpiinfo_.rank == 0) {
//        std::cout << __PRETTY_FUNCTION__ << " : " << "read header : nxh,nyh,nzh: xs,ys,zs = " << nxh << "," << nyh << "," << nzh << ":" << xs << "," << ys << "," << zs << std::endl;
//    }
//
//    fin_header.close();
//
//    float*  hflux_g;
//    hflux_g = new float[nxh*nyh*nzh];
//
//    FILE	*fp = fopen(fname.c_str(), "w");
//    fread(hflux_g, sizeof(float), nxh*nyh*nzh, fp);
//    fclose(fp);
//    
//    const int nx = domain_.nx;
//    const int ny = domain_.ny;
//    const int nz = domain_.nz;
//
//    for (int k=0; k<nz; k++) {
//    for (int j=0; j<ny; j++) {
//    for (int i=0; i<nx; i++) {
//        const float  x = domain_.x_min + i*dx0;
//        const float  y = domain_.y_min + j*dx0;
//        const float  z = domain_.z_min + k*dx0;
//
//        const int   ir = int( (x - xs) / dx0 );
//        const int   jr = int( (y - ys) / dx0 );
//        const int   kr = int( (z - zs) / dx0 );
//
//        if (  ir >= 0 && ir < nxh
//           && jr >= 0 && jr < nyh
//           && kr >= 0 && kr < nzh ) {
//           const int id  = i  + nx *j  + nx *ny *k;
//           const int idr = ir + nxh*jr + nxh*nyh*kr;
//
//           hflux[id] = hflux_g[idr];
//
//        }
//    }
//    }
//    }
//
//    delete [] hflux_g;
//}
