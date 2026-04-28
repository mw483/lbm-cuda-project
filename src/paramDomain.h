#ifndef PARAMDOMAIN_H_
#define PARAMDOMAIN_H_


// define //
#include "definePrecision.h"
#include "defineMemory.h"

// struct //
#include "stDomain.h"
#include "stMPIinfo.h"


class
paramDomain {
private:	
	// stDomain.h //
	Domain	domain_;

	// host or device memory //
	defineMemory::FlagHostDevice	flag_memory_;

public:	
	paramDomain () {}

	paramDomain	(
		char			*program_name,
		int				argc,
	   	char			*argv[], 
		const MPIinfo	&mpi,
		const defineMemory::FlagHostDevice	flag_memory
				)
	{
		set_paramDomain (
			program_name,
			argc,
			argv,
			mpi,
			flag_memory
			);
	}

	~paramDomain () {}


	// stDomain //
	// lattice boltzmann method //
	FLOAT	c_ref()		const { return domain_.c_ref; }
	FLOAT	cfl_ref()	const { return domain_.cfl_ref; }

	FLOAT	vel_cfl_ref()	const { return domain_.vel_cfl_ref; }

	// number of grid points (global) //
	int	nxg()	const { return domain_.nxg; }
	int	nyg()	const { return domain_.nyg; }
	int	nzg()	const { return domain_.nzg; }

	// number of grid points (local) //
	int	nx()	const { return domain_.nx; }
	int	ny()	const { return domain_.ny; }
	int	nz()	const { return domain_.nz; }
	int	nn()	const { return domain_.nn; }
	int	n0()	const { return domain_.n0; }

	// number of step //
	int	step()	const { return domain_.step; }

	// halo //
	int	halo()	const { return domain_.halo; }

	// time //
	FLOAT	time_end()	const { return domain_.time_end; }
	FLOAT	time()		const { return domain_.time; }
	FLOAT	dt()		const { return domain_.dt; }

	// dx, dy, dz //
	FLOAT	dx()	const { return domain_.dx; }

	// local //
	FLOAT	x_min()	const { return domain_.x_min; }
	FLOAT	y_min()	const { return domain_.y_min; }
	FLOAT	z_min()	const { return domain_.z_min; }

	FLOAT	x_max()	const { return domain_.x_max; }
	FLOAT	y_max()	const { return domain_.y_max; }
	FLOAT	z_max()	const { return domain_.z_max; }

	// global //
	FLOAT	xg_length()	const { return domain_.xg_length; }
	FLOAT	yg_length()	const { return domain_.yg_length; }
	FLOAT	zg_length()	const { return domain_.zg_length; }

	FLOAT	xg_min()	const { return domain_.xg_min; }
	FLOAT	yg_min()	const { return domain_.yg_min; }
	FLOAT	zg_min()	const { return domain_.zg_min; }

	FLOAT	xg_max()	const { return domain_.xg_max; }
	FLOAT	yg_max()	const { return domain_.yg_max; }
	FLOAT	zg_max()	const { return domain_.zg_max; }

	// region //
	FLOAT	*x()	const { return domain_.x; }
	FLOAT	*y()	const { return domain_.y; }
	FLOAT	*z()	const { return domain_.z; }


	const Domain	&domain() const { return domain_; }

public:	
	void
	set_paramDomain (
		char			*program_name,
		int				argc,
	   	char			*argv[], 
		const MPIinfo	&mpi,
		const defineMemory::FlagHostDevice	flag_memory
		);


	// memcpy //
	void
	memcpy_paramDomain (
		      paramDomain	&pdomain_n,
		const paramDomain	&pdomain
		);


	// time evolution //
	void	time_evolution ();

	// cout : nx, ny, nz, nxg, nyg, nzg //
	void	cout_Domain ();

	void	output_Condition ();
	void	read_Condition ();

private:
	void	check_Domain ();


	// axis host //
	void
	init_axis (
		FLOAT	*x,
		FLOAT	x_min,
		FLOAT	dx,
		int		nx,
		int		halo
		);

};


#endif
