#ifndef OUTPUTPARAVIEW_H_
#define OUTPUTPARAVIEW_H_

#include <cstdio>
#include <cmath>
#include <string>

#include <fstream>
#include "paramDomain.h"
#include "paramMPI.h"
#include "stVariables.h"
#include "stBasisVariables.h"
#include "stFluidProperty.h"
#include "stStress.h"

#include "stSolidForce.h"


class
Paraview {
private:
	// mpi
	int		rank_;
	int		rank_x_,  rank_y_,  rank_z_;

	int		ncpu_;
	int		ncpu_x_,  ncpu_y_,  ncpu_z_;

	// domain
	int		nx_,  ny_,  nz_;
	int		nn_;
	int		n0_;

	int		nxg_,  nyg_,  nzg_;

	int		halo_;

	FLOAT	xg_min_, yg_min_, zg_min_;
	FLOAT	dx_;

	FLOAT	c_ref_;

	// ncpu
	int		ncpu_div_x_,
			ncpu_div_y_,
			ncpu_div_z_;

	int		bnx_, bny_, bnz_;
	int		lnx_, lny_, lnz_;
	int		lnn_;

public:
	Paraview () {}

	Paraview (
		char				*program_name,
		int					argc,
	   	char				*argv[], 
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain
		)
	{
		set (
			program_name,
			argc,
			argv,
			pmpi,
			pdomain
			);
	}

	~Paraview () {}

public:
	void
	set (
		char				*program_name,
		int					argc,
		char				*argv[], 
		const paramMPI		&pmpi, 
		const paramDomain	&pdomain
		);


	// make header file //
	void
	Make_vtr_binary_file (
		int		t,
		int		clip_x,
		int		clip_y,
		int		clip_z,
		int		downsize
		);


	void
	Make_multiblock_dataset (
		int		t,
		int		clip_x,
		int		clip_y,
		int		clip_z,
		int		downsize
		);


	void
	Output_Fluid_binary_All (
		int		t,
		const Variables			*const cq,
		const BasisVariables	*const cbq,
		const FluidProperty		*const cfp,
		const Stress			*const str,
		int		downsize
		);


	// vtr //
	void
	Output_Fluid_vtr_binary_All (
		int		t,
		int		downsize
		);

private:
	void
	check_ncpu_div ();


	void
	write_binary_file_int (
		int	*f,
		char	fname[],
		int fid,
		int downsize,
		int nx,  int ny,  int nz,
		int lnx, int lny, int lnz
		);


	void
	write_binary_file (
		FLOAT	*f,
		char	fname[],
		int fid,
		int downsize,
		int nx,  int ny,  int nz,
		int lnx, int lny, int lnz
		);


	void
	write_binary_file (
		FLOAT	*u,
		FLOAT	*v,
		FLOAT	*w,
		char	fname[],
		int fid,
		int downsize,
		int nx,  int ny,  int nz,
		int lnx, int lny, int lnz
		);


	void 
	Write_Scalar_Obstacle_int8_binary_file (
		FILE	*fp_write,
		FILE	*fp_read,
		int nx, int ny, int nz
		);


	// combine data //
	template <typename T>
	void
	combine_data (
		T		*f_out,
		int		lnx,
		int		lny,
		int		lnz,
		T		*f_in,
		int		nx,
		int		ny,
		int		nz
		);

};


// template
#include "outputParaview_inc.h"


#endif
