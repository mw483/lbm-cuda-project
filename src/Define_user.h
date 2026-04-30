#ifndef DEFINE_H_USER_
#define DEFINE_H_USER_
//#include "definePrecision.h"
namespace
user_init {
        #include "definePrecision.h"
        const   FLOAT   DTDZ_LOW        = 0.01;  // T0 = T_ground + DTDZ_LOW * z
        const   FLOAT   DTDZ_HIGH       = 0.01; //0.01; // T0 = T_ground + DTDZ_LOW * ZHIGH + DTDZ_HIGH * (z - ZHIGH)
        const   FLOAT   ZHIGH           = 20.0;
        const   int     pt_ref          = 300.0;        //305.0;
// for pressure driven mode (flg_dpdx=1 or flg_dpdy=1)
        const FLOAT dpdx = -7.95*1.0e-4;
        const FLOAT dpdy        = 0.0;                          // dp/dx [g/(m2s2)]
// for constant heat flux mode (flg_scalar=2)
        const FLOAT     hf= -0.1;                                 // scalar flux from floor (MUST BE flg_scalar=2)
// for coriolis force (flg_coriolis=1)
        const FLOAT     angular_velocity = 7.2921e-5;           // angular velocity of earth [rad/sec]
//      const FLOAT     latitude=-6.21462;                      // Jakarta
        const FLOAT     latitude=35.681236;                     // Shibuya
        const FLOAT dist = 0.1;         // standard deviation of disturbance intensity ( std(init_u) = dist * vel_ref_
        const FLOAT z0 = 0.1; // roughness length, if (z0 <= 0.0) then Spalding law
};

namespace
user_output {
        #include "definePrecision.h"
        const FLOAT     average_interval        = 200.0;                //(sec)
        const FLOAT     skip_time               = 0.0;          //(sec)
        const FLOAT     output_interval_ins     = {{OUT_INT_INST}};                //(sec)
        const FLOAT     time_output_ins_ini     = 0.0;          //(sec)
        const int       nz_out                  = 3;            // number of planes to be output (xy)
        const int       nj_out                  = 1;            // number of planes to be output (xz)
        const int       ni_out                  = 1;            // number of planes to be output (yz)
        const int       nv_out                  = 0;            // number of ranks to be out (volume output)
        const int kout[] = {10, 20, 30}; // Default: 47 layers {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64};
        const int jout[] = {64};
        const int iout[] = {256};
        const int       vout_rank[] = {};                       // volume output
//      const char      odir_user[] = "./Output/testesttest/";  // odir_user = <user output directoryvout_rank[] = {1};
};

namespace
user_flags	{
	#include "definePrecision.h"
	const int flg_buoyancy=0;	// buoyancy force 		1:on, otherwise:off
	const int flg_scalar=2;		// scalar			1:on, otherwise:off, 2:homongeneous from horizontal surfaces
					//  >> need "input" files (surface flux data)
	const int flg_dpdx=0;		// Pressure gradient in x 	1:on, otherwise:off
	const int flg_dpdy=0;		// Pressure gradient in x	1:on, otherwise:off
	const int flg_coriolis=0;	// coriolis force		1:on, otherwise:off
	const int flg_collision=0;	// collision			1:cumulant, otherwise:SRT
	const int flg_disturbance=0;	//disturbance		1:on, otherwise:off
	const int flg_wallFunction=1; // wall function 1:on, otherwise:off

	const int flg_particle=1;	// type of particle source 	1:LSM w/ SGS, 2:LSM w/o SGS, otherwise:uniform (YOKOUCHI 2020)
};

#endif
