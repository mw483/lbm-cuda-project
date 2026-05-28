#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <string.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include "stSetting.h"
#include "setting.h"

/* ---------- Class Set_Value ---------- */
void Set_Value::option_read(char *argv[], Setting& setting) {
	int i = 0;
	// Read setting
	if (strcmp(argv[++i], "-NUM_RANK")		==0) setting.NUM_RANK 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (NUM_RANK)" 		<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-X_NUM_RANK")	==0) setting.X_NUM_RANK 	= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (X_NUM_RANK)" 	<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-Y_NUM_RANK")	==0) setting.Y_NUM_RANK 	= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (Y_NUM_RANK)" 	<< std::endl;
	  std::exit(0);
	}

	if (strcmp(argv[++i], "-X_RANK") 			==0) setting.X_RANK 			= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (X_RANK)" 			<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-Y_RANK")			==0) setting.Y_RANK 			= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (Y_RANK)" 			<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-X_DOMAIN")		==0) setting.X_DOMAIN 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (X_DOMAIN)" 		<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-Y_DOMAIN") 		==0) setting.Y_DOMAIN 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (Y_DOMAIN)" 		<< std::endl;
	 	std::exit(0);
	}

	if (strcmp(argv[++i], "-Z_DOMAIN") 		==0) setting.Z_DOMAIN 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (Z_DOMAIN)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-dX") 					==0) setting.dX 					= atof(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (dX)" 					<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-dT") 					==0) setting.dT 					= atof(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (dT)" 					<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FILE_START") 	==0) setting.FILE_START 	= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FILE_START)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FILE_END") 		==0) setting.FILE_END 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FILE_END)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-POUT") 				==0) setting.POUT 				= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (POUT)" 				<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-PGEN_STEP") 	==0) setting.PGEN_STEP 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (PGEN_STEP)" 	<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-NUM_GEN") 		==0) setting.NUM_GEN 			= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (NUM_GEN)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FLG_NUM") 		==0) setting.FLG_NUM 			= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FLG_NUM)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FLG_DENSITY") ==0) setting.FLG_DENSITY 	= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FLG_DENSITY)" << std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FLG_PROFILE") ==0) setting.FLG_PROFILE 	= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FLG_PROFILE)" << std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FLG_FOOT") 		==0) setting.FLG_FOOT 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FLG_FOOT)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FLG_FLUX") 		==0) setting.FLG_FLUX 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FLG_FLUX)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FLG_RESID") 	==0) setting.FLG_RESID 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (FLG_RESID)" 	<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-N_XY") 				==0) setting.N_XY 				= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (N_XY)" 				<< std::endl;
		std::exit(0);
	}

	setting.Z_OUT = new float[setting.N_XY];
	if (strcmp(argv[++i], "-Z_OUT") 			==0) {
		for (int ii=0; ii<setting.N_XY; ii++) setting.Z_OUT[ii]				= atof(argv[++i]);
	} else {
		std::cout << "run.sh is wrong. (Z_OUT)" 				<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-N_XZ") 				==0) setting.N_XZ 				= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (N_XZ)" 				<< std::endl;
		std::exit(0);
	}

	setting.Y_OUT = new float[setting.N_XZ];
	if (strcmp(argv[++i], "-Y_OUT") 			==0) {
		for (int ii=0; ii<setting.N_XZ; ii++) setting.Y_OUT[ii]				= atof(argv[++i]);
	} else {
		std::cout << "run.sh is wrong. (Y_OUT)" 				<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-N_YZ") 				==0) setting.N_YZ 				= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (N_YZ)" 				<< std::endl;
		std::exit(0);
	}

	setting.X_OUT = new float[setting.N_YZ];
	if (strcmp(argv[++i], "-X_OUT") 			==0) {
		for (int ii=0; ii<setting.N_YZ; ii++) setting.X_OUT[ii]				= atof(argv[++i]);
	} else {
		std::cout << "run.sh is wrong. (X_OUT)" 				<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-H_AVE") 			==0) setting.H_AVE 				= atof(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (H_AVE)" 			<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-N_SOURCE") 		==0) setting.N_SOURCE 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (N_SOURCE)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-ID_DIGIT") 		==0) setting.ID_DIGIT 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (ID_DIGIT)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-N_SENSOR") 		==0) setting.N_SENSOR 		= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (N_SENSOR)" 		<< std::endl;
		std::exit(0);
	}

	setting.CTR_SENSOR = new float[setting.N_SENSOR*3];
	if (strcmp(argv[++i], "-CTR_SENSOR") 	==0) {
		for (int ii=0; ii<setting.N_SENSOR*3; ii++) setting.CTR_SENSOR[ii] 	= atof(argv[++i]);
	} else {
		std::cout << "run.sh is wrong. (CTR_SENSOR)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-SIZE_SENSOR") ==0) {
		for (int ii=0; ii<3; ii++) 							setting.SIZE_SENSOR[ii] 	= atof(argv[++i]);
	} else {
		std::cout << "run.sh is wrong. (SIZE_SENSOR)" 		<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-N_FLUX") 			==0) setting.N_FLUX 			= atoi(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (N_FLUX)" 				<< std::endl;
		std::exit(0);
	}

	setting.Z_FLUX = new float[setting.N_FLUX];
	if (strcmp(argv[++i], "-Z_FLUX") 			==0) {
		for (int ii=0; ii<setting.N_FLUX; ii++) setting.Z_FLUX[ii]		= atof(argv[++i]);
	} else {
		std::cout << "run.sh is wrong. (Z_FLUX)" 				<< std::endl;
		std::exit(0);
	}
	
	if (strcmp(argv[++i], "-Z_RESID") 		==0) setting.Z_RESID			= atof(argv[++i]);
	else {
		std::cout << "run.sh is wrong. (Z_RESID)" 			<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-DIR_DATA") 		==0) setting.DIR_DATA 		= argv[++i];
	else {
		std::cout << "run.sh is wrong. (DIR_DATA)" 			<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-DIR_OUT") 		==0) setting.DIR_OUT 			= argv[++i];
	else {
		std::cout << "run.sh is wrong. (DIR_OUT)" 		<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FNAME_MAP") 	==0) setting.FNAME_MAP 		= argv[++i];
	else {
		std::cout << "run.sh is wrong. (FNAME_MAP)" 	<< std::endl;
		std::exit(0);
	}

	if (strcmp(argv[++i], "-FNAME_SOURCE") ==0) setting.FNAME_SOURCE= argv[++i];
	else {
		std::cout << "run.sh is wrong. (FNAME_SOURCE)" << std::endl;
		std::exit(0);
	}

	// cal
	setting.c_ref = setting.dX / setting.dT;
}

void Set_Value::output_info () {
	// Version
	std::cout << "-----------------------------------------" << std::endl;
	std::cout << "- This file is used for post-processing -" << std::endl;
	std::cout << "- of particle calculation.              -" << std::endl;
	std::cout << "-                                       -" << std::endl;
	std::cout << "- Ver.1.0 (2021/03/16)                  -" << std::endl;
	std::cout << "-  -> ParticleNum                       -" << std::endl;
	std::cout << "-  -> ParticleDensity                   -" << std::endl;
	std::cout << "-  -> ParticleProfile                   -" << std::endl;
	std::cout << "-  -> ParticleFootprint                 -" << std::endl;
	std::cout << "-  -> ParticleFlux                      -" << std::endl;
	std::cout << "-                                       -" << std::endl;
	std::cout << "- Ver.1.1 (2021/03/30)                  -" << std::endl;
	std::cout << "-  -> ParticleResidenceTime             -" << std::endl;
	std::cout << "-                                       -" << std::endl;
	std::cout << "-----------------------------------------" << std::endl;
	std::cout << std::endl;

	// Start time
	time_t	t = time(NULL);
	tm			*tm_st = localtime(&t);
	int year 	= tm_st->tm_year+1900;
	int month	= tm_st->tm_mon+1;
	int day		= tm_st->tm_mday;
	int hour	= tm_st->tm_hour;
	int min   = tm_st->tm_min;
	int sec		= tm_st->tm_sec;
	char		time_s[50];
	sprintf(time_s, "Post-processing starts from %d/%d/%d %d:%d:%d", year, month, day, hour, min, sec);
	std::cout << time_s << std::endl;
	std::cout << std::endl;
}

void Set_Value::output_setting (Setting& setting) {
	// Output setting to console
	std::cout << "--------------- Setting -----------------"	<< std::endl;
	std::cout << "--- Basic setting ---" 											<< std::endl;
	std::cout << "NUM_RANK     = " << setting.NUM_RANK 				<< std::endl;
	std::cout << "X_NUM_RANK   = " << setting.X_NUM_RANK 			<< std::endl;
	std::cout << "Y_NUM_RANK   = " << setting.Y_NUM_RANK 			<< std::endl;
	std::cout << "X_RANK       = " << setting.X_RANK 					<< std::endl;
	std::cout << "Y_RANK       = " << setting.Y_RANK 					<< std::endl;
	std::cout << "X_DOMAIN     = " << setting.X_DOMAIN	 			<< std::endl;
	std::cout << "Y_DOMAIN     = " << setting.Y_DOMAIN	 			<< std::endl;
	std::cout << "Z_DOMAIN     = " << setting.Z_DOMAIN 				<< std::endl;
	std::cout << "dX           = " << setting.dX				 			<< std::endl;
	std::cout << "dT           = " << setting.dT				 			<< std::endl;
	std::cout << "c_ref        = " << setting.c_ref			 			<< std::endl;
	std::cout << std::endl;
	std::cout << "--- For density ---"												<< std::endl;
	std::cout << "Z_OUT[" 				 << setting.N_XY << 				"]     = [";
	for (int i=0; i<setting.N_XY; i++) {
	std::cout << setting.Z_OUT[i];
	if (i < setting.N_XY-1) std::cout << " ";
	}
	std::cout << "]" 																					<< std::endl;
	std::cout << "Y_OUT[" 				 << setting.N_XZ << 				"]     = [";
	for (int i=0; i<setting.N_XZ; i++) {
	std::cout << setting.Y_OUT[i];
	if (i < setting.N_XZ-1) std::cout << " ";
	}
	std::cout << "]"																					<< std::endl;
	std::cout << "X_OUT[" 				 << setting.N_YZ << 				"]     = [";
	for (int i=0; i<setting.N_YZ; i++) {
	std::cout << setting.X_OUT[i];
	if (i < setting.N_YZ-1) std::cout << " ";
	}
	std::cout << "]"																					<< std::endl;
	std::cout << "H_AVE        = " << setting.H_AVE			 			<< std::endl;
	std::cout << std::endl;
	std::cout << "--- For footprint ---" 											<< std::endl;
	std::cout << "N_SOURCE     = " << setting.N_SOURCE        << std::endl;
	std::cout << "ID_DIGIT     = " << setting.ID_DIGIT				<< std::endl;
	std::cout << std::endl;
	std::cout << "---   For flux    ---"											<< std::endl;
	std::cout << "N_FLUX["				 << setting.N_FLUX <<				"]    = [";
	for (int i=0; i<setting.N_FLUX; i++ ) {
	std::cout << setting.Z_FLUX[i];
	if (i < setting.N_FLUX-1) std::cout << " ";
	}
	std::cout << "]"																					<< std::endl;
	std::cout << std::endl;
	std::cout << "--- For Residence ---"											<< std::endl;
	std::cout << "Z_RESID      = " << setting.Z_RESID         << std::endl;
	std::cout << std::endl;
	std::cout << "--- Data / Output ---" 											<< std::endl;
	std::cout << "DIR_DATA     = " << setting.DIR_DATA				<< std::endl;
	std::cout << "DIR_OUT      = " << setting.DIR_OUT					<< std::endl;
	std::cout << std::endl;
	std::cout << "---  Output Flag  ---"											<< std::endl;

	std::string	flg_num				= "off";
	std::string	flg_density		= "off";
	std::string	flg_profile		= "off";
	std::string	flg_footprint	= "off";
	std::string	flg_flux			= "off";
	std::string	flg_residual	= "off";

	if (setting.FLG_NUM 		== 1) flg_num 			= "on";
	if (setting.FLG_DENSITY == 1) flg_density 	= "on";
	if (setting.FLG_PROFILE == 1) flg_profile 	= "on";
	if (setting.FLG_FOOT		== 1) flg_footprint = "on";
	if (setting.FLG_FLUX 		== 1) flg_flux			= "on";
	if (setting.FLG_RESID 	== 1) flg_residual 	= "on";

	std::cout << "- Num        : " << flg_num									<< std::endl;
	std::cout << "- Density    : " << flg_density							<< std::endl;
	std::cout << "- Profile    : " << flg_profile							<< std::endl;
	std::cout << "- Footprint  : " << flg_footprint						<< std::endl;
	std::cout << "- Flux       : " << flg_flux								<< std::endl;
	std::cout << "- Residual   : " << flg_residual						<< std::endl;
	std::cout << "-----------------------------------------"  << std::endl;
	std::cout << std::endl;
}
