#ifndef CAL_SOLID_H_
#define CAL_SOLID_H_


#include "definePrecision.h"
#include "defineMemory.h"
#include "stMPIinfo.h"
#include "stDomain.h"

// input output //
#include "stFluidProperty.h"

// other class //
#include "classSolidData.h"
#include "classReadSTL.h"
#include "classSolidForce.h"


class
cal_Solid {
private:
	// mpiinfo //
	MPIinfo	mpiinfo_;

	// domain //
	Domain	domain_;

	// solid //
	int		num_solid_;

	// classSTL //
	classSolidData	csoliddata_;

	classReadSTL	creadstl_;

	classSolidForce	csolidforce_;

public:	
	cal_Solid () {}

	cal_Solid (
		const MPIinfo		&mpiinfo, 
		const Domain		&domain,
		const FluidProperty	*fluid
		)
	{
		init_cal_Solid (
			mpiinfo,
			domain,
			fluid
			);
	}

	~cal_Solid () {}


	const solidForce	&solidforce_h() const { return csolidforce_.solidforce_h(); }
	const solidForce	&solidforce_d() const { return csolidforce_.solidforce_d(); }

public:
	// initialize //
	void
	init_cal_Solid (
		const MPIinfo	&mpiinfo, 
		const Domain	&domain,
		const FluidProperty	*fluid
		);


	// output levelset function //
	void	
	set_levelset_obstacle (
		FluidProperty	*fluid_h
		);


	// stl file : level set //
	void
	move_solid_data_gpu (
		FluidProperty	*fluid_d
		);


	// solid Force //
	void
	get_solidForce (
		const int	*id_solidData,
		const FLOAT	*r,
		const FLOAT	*u,
		const FLOAT	*v,
		const FLOAT	*w,
		const FLOAT	*lv
		);


	void
	cout_solidForce ();


	void
	output_solidForce (
		bool	first_cal,
		FLOAT	time
		);


	void
	memcpy_solidForce_DeviceToHost ();

private:
	// 物体status flag設定 //
	void
	set_status_flag (
		      char	*status,
		const FLOAT	*lv
		);

};


#endif
