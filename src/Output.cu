#include "Output.h"


// Output
// File
void Output::Output_Cal_Condition(const struct timeval *begin, const struct timeval *end,
	MPI_Library *mpi,
	CalFlag *calfrg, Domain *cdo)
{
    const int	cal_time = (end->tv_sec  - begin->tv_sec) * 1000
						 + (end->tv_usec - begin->tv_usec) / 1000.0;

	char	 name[256];
	const int	step = cdo->step;
	const int	nx = cdo->nx,
		  		ny = cdo->ny - Offset_Y*2,
				nz = cdo->nz - Offset_Z*2,
				nn = cdo->nn;
	const int	ncpu_x = 1,
		  		ncpu_y = mpi->ncpu_y,
				ncpu_z = mpi->ncpu_z,
				ncpu   = ncpu_x * ncpu_y * ncpu_z;

	sprintf(name, "./log_calculation_nn_%d_%d_%d_cpu_%d_%d_%d.txt", nx, ny, nz, ncpu_x, ncpu_y, ncpu_z);

//	const int	cout_frg = calfrg->cout_frg,
//		  		fout_frg = calfrg->fout_frg,
//				restart_out_frg = calfrg->restart_out_frg;

//	const double	Re_num = cdo->Re_num;


	fout.open(name);
	if(!fout) {
		std::cout << "file is not opened\n";
	}

	fout << "time(ms)  / step = \t" << cal_time/(float)step << "\n";
	fout << "time(ms)         = \t" << cal_time << "\n";
	fout << "step             = \t" << step << "\n";
	fout << "nn  (grid)       = \t" << nn   << "\n";
	fout << "nx               = \t" << nx   << "\n";
	fout << "ny               = \t" << ny   << "\n";
	fout << "nz               = \t" << nz   << "\n";
	fout << "ncpu (mpi)       = \t" << ncpu   << "\n";
	fout << "ncpu_x           = \t" << ncpu_x << "\n";
	fout << "ncpu_y           = \t" << ncpu_y << "\n";
	fout << "ncpu_z           = \t" << ncpu_z << "\n";

	fout.close();
}


// channel flow
void Output::Output_Channel(MPI_Library *mpi, 
	const int step, Domain *cdo, const BasisVariables *cq, const Stress *str, const FluidProperty *cfp)
{
	const int	nx = cdo->nx,
		  		ny = cdo->ny,
				nz = cdo->nz;
	const int	nxy = nx*ny;

	// statistics
	const int	buff_x = 1;
	const int	nl   = cdo->n0;
	const int	nav = (cdo->ny-2*Offset_Y)*(cdo->nz-2*Offset_Z) * (mpi->ncpu_y)*(mpi->ncpu_z);

	// local
	FLOAT	*u_av,   *v_av,  *w_av;
	FLOAT	*uu_av,  *vv_av, *ww_av;
	FLOAT	*Fcs_av, *vis_sgs_av;

	u_av  = new FLOAT[nx];	v_av  = new FLOAT[nx];	w_av  = new FLOAT[nx];
	uu_av = new FLOAT[nx];	vv_av = new FLOAT[nx];	ww_av = new FLOAT[nx];
	Fcs_av = new FLOAT[nx];	vis_sgs_av = new FLOAT[nx];

	// global
	FLOAT	*u_av_g,   *v_av_g,  *w_av_g;
	FLOAT	*uu_av_g,  *vv_av_g, *ww_av_g;
	FLOAT	*Fcs_av_g, *vis_sgs_av_g;

	u_av_g  = new FLOAT[nx];	v_av_g  = new FLOAT[nx];	w_av_g  = new FLOAT[nx];
	uu_av_g = new FLOAT[nx];	vv_av_g = new FLOAT[nx];	ww_av_g = new FLOAT[nx];
	Fcs_av_g = new FLOAT[nx];	vis_sgs_av_g = new FLOAT[nx];


	// zero
	for (int i=0; i<nx; i++) {
		// local
		u_av [i] = 0.0;	v_av [i] = 0.0;	w_av [i] = 0.0;
		uu_av[i] = 0.0;	vv_av[i] = 0.0;	ww_av[i] = 0.0;

		Fcs_av[i] = 0.0;	vis_sgs_av[i] = 0.0;

		// global
		u_av_g [i] = 0.0;	v_av_g [i] = 0.0;	w_av_g [i] = 0.0;
		uu_av_g[i] = 0.0;	vv_av_g[i] = 0.0;	ww_av_g[i] = 0.0;

		Fcs_av_g[i] = 0.0;	vis_sgs_av_g[i] = 0.0;
	}

	// u, v, w
	for (int i=buff_x; i<nx-buff_x; i++) {
		for (int k=Offset_Z; k<nz-Offset_Z; k++) {
			for (int j=Offset_Y; j<ny-Offset_Y; j++) {
				const int	id = i + nx*j + nxy*k;

				FLOAT	cu = cq->u_n[id] * C_REF;
				FLOAT	cv = cq->v_n[id] * C_REF;
				FLOAT	cw = cq->w_n[id] * C_REF;

				u_av[i] += cu / nav;
				v_av[i] += cv / nav;
				w_av[i] += cw / nav;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(u_av, u_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(v_av, v_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(w_av, w_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	// uu, vv, ww
	for (int i=buff_x; i<nx-buff_x; i++) {
		for (int k=Offset_Z; k<nz-Offset_Z; k++) {
			for (int j=Offset_Y; j<ny-Offset_Y; j++) {
				const int	id = i + nx*j + nxy*k;

				FLOAT	cu = cq->u_n[id] * C_REF;
				FLOAT	cv = cq->v_n[id] * C_REF;
				FLOAT	cw = cq->w_n[id] * C_REF;

				uu_av[i] += pow(cu - u_av_g[i], 2) / nav;
				vv_av[i] += pow(cv - v_av_g[i], 2) / nav;
				ww_av[i] += pow(cw - w_av_g[i], 2) / nav;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(uu_av, uu_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(vv_av, vv_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(ww_av, ww_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// sub-grid scale model
	for (int i=buff_x; i<nx-buff_x; i++) {
		for (int k=Offset_Z; k<nz-Offset_Z; k++) {
			for (int j=Offset_Y; j<ny-Offset_Y; j++) {
				const int	id = i + nx*j + nxy*k;

				FLOAT	Fcs  = str->Fcs_sgs[id];
				FLOAT	svis = str->vis_sgs[id];

				Fcs_av[i]     += Fcs   / nav;
				vis_sgs_av[i] += svis  / nav;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(Fcs_av,     Fcs_av_g,     nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(vis_sgs_av, vis_sgs_av_g, nx, MFLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (mpi->rank == 0) {
		// fout
		const int	num_file = 1; // 1 or 2

		for (int ii=0; ii<num_file; ii++) {
			char	name[256];
			if (ii == 0) {
				sprintf(name, "./result_channel/output.dat");
			}
			else if (ii == 1) {
				sprintf(name, "./result_channel/output%dx%dx%d-%d.dat", nx, ny, nz, step);
			}
			else {
				sprintf(name, "./result_channel/output.dat");
			}

			std::ofstream fout;
			fout.open(name);

			for (int i=buff_x; i<nx-buff_x; i++) {
				fout << (FLOAT)(i-buff_x+0.5) / (nl) * cdo->Re_num * 2.0 << "\t"		// 1
					 << u_av_g [i] << "\t"<< v_av_g [i] << "\t" << w_av_g [i] << "\t"		// 2-4
					 << uu_av_g[i] << "\t"<< vv_av_g[i] << "\t" << ww_av_g[i] << "\t"		// 5-7
					 << Fcs_av_g[i] << "\t" << vis_sgs_av_g[i] << "\n";					// 8-9
			}
			fout.close();
		}
	}

	// local
	delete [] u_av;		delete [] v_av;		delete [] w_av;
	delete [] uu_av;	delete [] vv_av;	delete [] ww_av;
	delete [] Fcs_av;	delete [] vis_sgs_av;

	// global
	delete [] u_av_g;		delete [] v_av_g;		delete [] w_av_g;
	delete [] uu_av_g;	delete [] vv_av_g;	delete [] ww_av_g;
	delete [] Fcs_av_g;	delete [] vis_sgs_av_g;

	// statistics
}


// Output *****
