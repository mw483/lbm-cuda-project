#ifndef STMPIINFO_H_
#define STMPIINFO_H_


#include <mpi.h>
#include "definePrecision.h"


struct 
MPIinfo {
	int		ncpu;
	int		ncpu_x, ncpu_y, ncpu_z;

	int		rank;
	int		rank_x, rank_y, rank_z;

	int		rank_xm, rank_ym, rank_zm;
	int		rank_xp, rank_yp, rank_zp;
	int		id_rank;
	int		id_rank_xm, id_rank_ym, id_rank_zm;
	int		id_rank_xp, id_rank_yp, id_rank_zp;

	int		tag_xm, tag_ym, tag_zm;
	int		tag_xp, tag_yp, tag_zp;
};


struct
Buffer {
	int		num_halo;
	int		num_variables;

	FLOAT	*buff_s_xm, *buff_s_ym, *buff_s_zm; // send
	FLOAT	*buff_r_xm, *buff_r_ym, *buff_r_zm; // recv
	FLOAT	*buff_s_xp, *buff_s_yp, *buff_s_zp; 
	FLOAT	*buff_r_xp, *buff_r_yp, *buff_r_zp; 
};


#endif
