#ifndef CLASSSOLIDFORCE_H_
#define CLASSSOLIDFORCE_H_


#include "definePrecision.h"
#include "defineMemory.h"

// struct //
#include "stMPIinfo.h"
#include "stDomain.h"
#include "stFluidProperty.h"

#include "stSolidForce.h"


class
classSolidForce {
private:
	// cuda //
	dim3	grid_,
			block_2d_;

	// mpiinfo //
	MPIinfo	mpiinfo_;


	// domain //
	int		nx_, ny_, nz_;
	int		nn_;
	int		halo_;
	FLOAT	dx_;

	// lbm //
	FLOAT	c_ref_;

	// solid //
	int		num_solid_;

	solidForce_STL	*sforce_stl_;

	solidForce		sforce_h_;
	solidForce		sforce_d_;

public:	
	classSolidForce () {}

	classSolidForce (
		const MPIinfo		&mpiinfo, 
		const Domain		&domain,
		const FluidProperty	*fluid,
		defineMemory::FlagHostDevice	flag_memory
		)
	{
		init_classSolidForce (
			mpiinfo,
			domain,
			fluid
			);
	}

	~classSolidForce () {}


	FLOAT	solidForce_x (int id_stl)	{ return	sforce_stl_[id_stl].force_x; }
	FLOAT	solidForce_y (int id_stl)	{ return	sforce_stl_[id_stl].force_y; }
	FLOAT	solidForce_z (int id_stl)	{ return	sforce_stl_[id_stl].force_z; }

	const FLOAT	*force_inner_x_h ()	{ return	sforce_h_.force_inner_x; }
	const FLOAT	*force_inner_y_h ()	{ return	sforce_h_.force_inner_y; }
	const FLOAT	*force_inner_z_h ()	{ return	sforce_h_.force_inner_z; }

	const FLOAT	*force_inner_x_d ()	{ return	sforce_d_.force_inner_x; }
	const FLOAT	*force_inner_y_d ()	{ return	sforce_d_.force_inner_y; }
	const FLOAT	*force_inner_z_d ()	{ return	sforce_d_.force_inner_z; }

	const solidForce	&solidforce_h() const { return sforce_h_; }
	const solidForce	&solidforce_d() const { return sforce_d_; }

	const solidForce_STL	&solidforce_stl(int i) const { return sforce_stl_[i]; }

public:
	void
	init_classSolidForce (
		const MPIinfo		&mpiinfo, 
		const Domain		&domain,
		const FluidProperty	*fluid
		);


	// memcpy //
	void	memcpy_solidForce_DeviceToHost ();
	void	memcpy_solidForce_HostToDevice ();


	void
	memcpy_solidForce (
		      solidForce &sforcen,
   		const solidForce &sforce
		);


	// num_solid //
	void
	set_num_solid (int	num_solid);


	// id_solidData //
	void
	set_id_solidData (
		const int			*id_solidData
		);


	// force //
	void
	cal_solidForce (
		const FLOAT	*r,
		const FLOAT	*u,
		const FLOAT	*v,
		const FLOAT	*w,
		const FLOAT	*lv
		);


	// force output //
	void
	output_solidForce (
		bool	first_cal,
		FLOAT	time
		);


private:
	// allocate //
	void
	allocate (
		);

	void	allocate_host   ();
	void	allocate_device ();


	// calculation //
	void
	reduction_solidForce ();


	void
	filter_solidForce ();


	void
	thrust_reduction ();


	void
	cal_solidForce_tensor (
		const FLOAT	*r,
		const FLOAT	*u,
		const FLOAT	*v,
		const FLOAT	*w,
		const FLOAT	*lv
		);
};


#endif
