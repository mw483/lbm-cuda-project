#ifndef STRUCT_SETTING
#define STRUCT_SETTING

#include <iostream>

struct Setting {
		// Basic setting
		int 		NUM_RANK;
		int 		X_NUM_RANK;
		int			Y_NUM_RANK;
		int 		X_RANK;
		int 		Y_RANK;
		int 		X_DOMAIN;
		int 		Y_DOMAIN;
		int 		Z_DOMAIN;
		float 	dX;
		float 	dT;
		float  	c_ref;
	
		// Particle 
		int 		FILE_START;
		int 		FILE_END;
		int			POUT;
		int 		PGEN_STEP;
		int			NUM_GEN;

		// Flag
		int 		FLG_NUM;
		int			FLG_DENSITY;
		int			FLG_PROFILE;
		int			FLG_FOOT;
		int			FLG_FLUX;
		int			FLG_RESID;
	
		// For density
		int			N_XY;
		float*	Z_OUT;
		int 		N_XZ;
		float* 	Y_OUT;
		int			N_YZ;
		float*	X_OUT;
		float		H_AVE;

		// For footprint
		int 		N_SOURCE;
		int 		ID_DIGIT;
		int			N_SENSOR;
		float*	CTR_SENSOR;
		float		SIZE_SENSOR[3];

		// For flux
		int			N_FLUX;
		float*	Z_FLUX;
		
		// For Residence time
		float		Z_RESID;

		// Directory
		char*		DIR_DATA;
		char*		DIR_OUT;
		char*		FNAME_MAP;
		char*		FNAME_SOURCE;
};

#endif
