#ifndef CLASSSOLIDINFO_H_
#define CLASSSOLIDINFO_H_


#include "definePrecision.h"
#include "stMPIinfo.h"
#include "stDomain.h"

// struct //
#include "stSolidData.h"


class
classSolidData {
private:
	// mpiinfo //
	MPIinfo	mpiinfo_;

	// domain //
	Domain	domain_;

	// solid //
	solidData	soliddata_;


public:	
	classSolidData () {}

	classSolidData (
		const MPIinfo	&mpiinfo, 
		const Domain	&domain
		)
	{
		init_classSolidData (
			mpiinfo,
			domain
			);
	}

	~classSolidData () {}

	const solidData	&soliddata() const { return soliddata_; }

public:
	// initialize //
	void
	init_classSolidData (
		const MPIinfo	&mpiinfo, 
		const Domain	&domain
		);


	void
	move_SolidData (
		FLOAT	force_x,
		FLOAT	force_y,
		FLOAT	force_z 
		);

private:
	// initialize //
	void
	set_solidData ();

};


#endif
