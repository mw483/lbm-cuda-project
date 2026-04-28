#include "outputStatus.h"

#include <mpi.h>
#include <iostream>
#include "defineBoundaryFlag.h"
#include "defineReferenceVel.h"


// public //
void	outputStatus::
set (
	const MPIinfo	&mpi, 
	const CalFlag	&calflg,
	const Domain	&domain
	)
{
	rank_ = mpi.rank;
	ncpu_ = mpi.ncpu;

	nx_ = domain.nx;
	ny_ = domain.ny;
	nz_ = domain.nz;
	nn_ = domain.nn;

	halo_ = domain.halo;

	nxg_ = domain.nxg;
	nyg_ = domain.nyg;
	nzg_ = domain.nzg;

//	dt_ = domain.dt;
	dt_ = 1.0;

	// lbm //
	c_ref_ = domain.c_ref;
}


void	outputStatus::
cout_Status (
	const FLOAT *r,
	const FLOAT *u,
	const FLOAT *v,
	const FLOAT *w,
	const FLOAT *vis,
	FLOAT 		vis0,
	const FLOAT *vis_sgs,
	const FLOAT *Div,
	const char	*status
	)
{
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nz = nz_;
	const int	nxy = nx*ny;

	const int	halo = halo_;

	const int	nxg = nxg_;
	const int	nyg = nyg_;
	const int	nzg = nzg_;

	const FLOAT	dt = dt_;

	FLOAT	count   = 0.0,
			count_g = 0.0;

	if (rank_ == 0) {
		std::cout << "nx,  ny,  nz,  ncpu = " << nx  << ", " << ny  << ", " << nz  << ", " << ncpu_ << "\n";
		std::cout << "nxg, nyg, nzg       = " << nxg << ", " << nyg << ", " << nzg << "\n";
	}

	FLOAT	rho_all = 0.0,
			rho_max = -1.0e10,
			rho_min =  1.0e10;
	FLOAT	vel, vel_max = 0.0;

	FLOAT	div, div_max  = 0.0;
	FLOAT	energy = 0.0;

	FLOAT	vis_max = 0.0, vis_sgs_max = 0.0;
	FLOAT	vis_sgs_av = 0.0;

	// mpi
	FLOAT	rho_all_g =  0.0,
			rho_max_g = -1.0e10,
			rho_min_g =  1.0e10,
			vel_max_g =  0.0,
			div_max_g =  0.0,
			energy_g  =  0.0,
			vis_max_g =  0.0,
			vis_sgs_max_g = 0.0,
			vis_sgs_av_g  = 0.0;
		
	for (int k=halo; k<nz-halo; k++) {
		for (int j=halo; j<ny-halo; j++) {
			for (int i=halo; i<nx-halo; i++) {
				// index
				const int	id_c0_c0_c0 = i   + nx*j     + nxy*k    ;

				if (status[id_c0_c0_c0] == STATUS_WALL) { continue; }

				// rho
				rho_all += r[id_c0_c0_c0];
				rho_max = fmax(fabs(r[id_c0_c0_c0]), rho_max);
				rho_min = fmin(fabs(r[id_c0_c0_c0]), rho_min);

				// velocity
				vel = sqrt( pow(u[id_c0_c0_c0], 2) + pow(v[id_c0_c0_c0], 2) + pow(w[id_c0_c0_c0], 2) ); 
				vel_max = fmax(vel, vel_max);

				energy += 0.5*pow(vel, 2);
				
				// divergence
				div = Div[id_c0_c0_c0];
				div_max = fmax(div, div_max);

				// subgrid scale model
//				vis_max     = fmax(vis[id_c0_c0_c0],     vis_max);
				vis_max     = vis0;
				vis_sgs_max = fmax(vis_sgs[id_c0_c0_c0], vis_sgs_max);
				vis_sgs_av  += vis_sgs[id_c0_c0_c0];


				// float
				count++;
			}
		}
	}


	// mpi //
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&count,		&count_g,		1, MFLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Reduce(&rho_all,	&rho_all_g,		1, MFLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&rho_max,	&rho_max_g,		1, MFLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&rho_min,	&rho_min_g,		1, MFLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(&vel_max,	&vel_max_g,		1, MFLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&div_max,	&div_max_g,		1, MFLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

	MPI_Reduce(&energy,		&energy_g,		1, MFLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Reduce(&vis_max,		&vis_max_g,			1, MFLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&vis_sgs_max,	&vis_sgs_max_g,		1, MFLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&vis_sgs_av,		&vis_sgs_av_g,		1, MFLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);


	if (count != 0.0) {
		rho_all_g    *= 1.0/(FLOAT)count_g;
		energy_g     *= 1.0/(FLOAT)count_g;

		vis_sgs_av_g *= 1.0/(FLOAT)count_g;
	}
	// mpi *****

	if (rank_ == 0) {
		std::cout << "CFL = " << vel_max_g*dt    << " (max : " << vel_max_g * c_ref_ << ")\n";
		std::cout << "Rho    = " << rho_all_g    << " (max, min : " << rho_max_g << ", " << rho_min_g << ")\n";
		std::cout << "DIV    = " << div_max_g    << "\n";
		std::cout << "Energy = " << energy_g << "\n";
		std::cout << "VIS, VIS_SGS = " << vis_max_g << ", " << vis_sgs_max_g << "(" << vis_sgs_max_g/vis_max_g << ")\n";
		std::cout << "  average    = " << vis_max_g << ", " << vis_sgs_av_g  << "(" << vis_sgs_av_g /vis_max_g << ")\n";
		std::cout << "\n";
	}
}


// outputStatus //
