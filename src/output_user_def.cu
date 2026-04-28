#include "output_user_def.h"

#include <iostream>
#include "option_parser.h"
#include "functionLib.h"
#include "fileLib.h"
#include "defineCoefficient.h"
#include "defineReferenceVel.h"
#include "defineBoundaryFlag.h"

#include "Define_user.h"		// mod 2018

#include <stdlib.h>
#include <math.h>

void
output_user_def (
	int			t,
	char			*program_name,
	int			argc,
	char			*argv[], 
	const paramMPI		&pmpi, 
	const paramDomain	&pdomain,
	const Variables     	*const variables,
	const BasisVariables	*const cbq,
	const FluidProperty	*const cfp
	)
{
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);

	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// mpi //
	int	rank    = pmpi.rank();
	int	rank_x  = pmpi.rank_x();
	int	rank_y  = pmpi.rank_y();
	int	rank_z  = pmpi.rank_z();

	int	ncpu    = pmpi.ncpu();
	int	ncpu_x  = pmpi.ncpu_x();
	int	ncpu_y  = pmpi.ncpu_y();
	int	ncpu_z  = pmpi.ncpu_z();


	// domain //
	int	nx  = pdomain.nx();
	int	ny  = pdomain.ny();
	int	nz  = pdomain.nz();
	int	nn  = pdomain.nn();

	int	n0  = pdomain.n0();

	int	nxg  = pdomain.nxg();
	int	nyg  = pdomain.nyg();
	int	nzg  = pdomain.nzg();

	int	halo  = pdomain.halo();


	FLOAT	xg_min  = pdomain.xg_min();
	FLOAT	yg_min  = pdomain.yg_min();
	FLOAT	zg_min  = pdomain.zg_min();

	FLOAT	dx  = pdomain.dx();

	FLOAT	c_ref   = pdomain.c_ref();		// sound speed //
	FLOAT	cfl_ref = pdomain.cfl_ref();

//	FLOAT	dt_real = 2.0 / c_ref;			// dt //
	FLOAT	dt_real = dx / c_ref;			// dt //  MOD2018


	// variables //
	FLOAT	*r = cbq->r_n;
	FLOAT	*u = cbq->u_n;
	FLOAT	*v = cbq->v_n;
	FLOAT	*w = cbq->w_n;

	FLOAT	*l_obs = cfp->l_obs;

	FLOAT   *T = variables->T_n;

	// user define function //

	int i, ii, j, jj, k, kk, n, id, idxy, idxz, i_vol;

	float	time = t * dt_real;
	float	average_interval	= user_output::average_interval;
	float	skip_time		= user_output::skip_time;
	float   output_interval_ins	= user_output::output_interval_ins;
	float	time_output_ins_ini	= user_output::time_output_ins_ini;
	float	wspeed;
	float	T0;		// T0 = T - 300.0
	int	nz_out			= user_output::nz_out;
	int	nj_out			= user_output::nj_out;
	int	ni_out			= user_output::ni_out;
	int	nv_out			= user_output::nv_out;
	int	nxy = nx * ny;
	int	nxz = nx * nz;
//	int	nyz = ny * nz;
	int	nxyz= nx * ny * nz;
	int	nxy_out = nx * ny * nz_out;

	static bool	flag;
	static bool	flag_ins;
	static int	i_count;
	static float	time_average;
	static float    time_output_ins;
	static int	*nx_air;
	static int	*nxy_air;
	static int	*n_obs;		// 2018

	// for output profile //
	static	double	*r_prof;
	static 	double	*u_prof;
	static 	double	*v_prof;
	static 	double	*w_prof;
        static  double  *t_prof;
	static 	double	*uu_prof;
	static 	double	*vv_prof;
	static 	double	*ww_prof;
	static 	double	*uv_prof;
	static 	double	*uw_prof;
	static 	double	*vw_prof;
        static  double  *ut_prof;
        static  double  *vt_prof;
        static  double  *wt_prof;
        static  double  *tt_prof;
	static  double  *uuu_prof;
	static  double  *vvv_prof;
	static  double  *www_prof;
        static  double  *ttt_prof;
	static  double  *uuw_prof;
	static  double  *vvw_prof;

	// for output xy //
	int	kout[nz_out];
	int	jout[nj_out];
	int	iout[ni_out];
	int	vout_rank[nv_out];
	for(k=0;k<nz_out;k++){
		kout[k] = user_output::kout[k];
	}
	for(j=0;j<nj_out;j++){
		jout[j] = user_output::jout[j];
	}
	for(i=0;i<ni_out;i++){
		iout[i] = user_output::iout[i];
	}
	for(i=0;i<nv_out;i++){
		vout_rank[i] = user_output::vout_rank[i];
	}

	static 	double	*u_xy;
	static 	double	*v_xy;
	static 	double	*w_xy;
        static  double  *t_xy;
	static 	double	*uu_xy;
	static 	double	*vv_xy;
	static 	double	*ww_xy;
        static  double  *tt_xy;
	static 	double	*uv_xy;
	static 	double	*uw_xy;
	static 	double	*vw_xy;
        static  double  *ut_xy;
        static  double  *vt_xy;
        static  double  *wt_xy;
	static 	float	*wspeed_max;
	static	int	*wspeed_time;
	// for average in y and time //
	static	double	*u_x;
	static	double	*v_x;
        static  double  *w_x;
        static  double  *t_x;
        static  double  *uu_x;
        static  double  *vv_x;
        static  double  *ww_x;
        static  double  *tt_x;
        static  double  *uv_x;
        static  double  *uw_x;
        static  double  *vw_x;
        static  double  *ut_x;
        static  double  *vt_x;
        static  double  *wt_x;
        static  double  *uuu_x;
        static  double  *vvv_x;
        static  double  *www_x;
        static  double  *ttt_x;
        static  double  *uuw_x;
        static  double  *vvw_x;

	if(rank == 0) printf("time=%f (i=%d)\n",time,t);

// special output for yz slices ==================================================================================
// end special output for yz slices =============================================================================



// instantaneous output //
	if (flag_ins == 0) {
		time_output_ins = time_output_ins_ini;
		flag_ins = 1;
	}
	if (time >= time_output_ins) {
	// output instantaneous u //
		char str_xy_i[100];		// for file name
		sprintf(str_xy_i,"./Output/xy_ins_u%08d_%04d.csv",t,rank);
		FILE *fp_xy_ui;
		fp_xy_ui = fopen(str_xy_i, "w");
		if (fp_xy_ui == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy_ui,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy_ui,"%d,%f,",t,time);
		fprintf(fp_xy_ui,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy_ui,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy_ui,"%6f,",   u[idxy]*c_ref );
				}
				fprintf(fp_xy_ui,"\n");
			}
		}
		fclose(fp_xy_ui);
	// output instantaneous v //
		sprintf(str_xy_i,"./Output/xy_ins_v%08d_%04d.csv",t,rank);
		FILE *fp_xy_vi;
		fp_xy_vi = fopen(str_xy_i, "w");
		if (fp_xy_vi == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy_vi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy_vi,"%d,%f,",t,time);
		fprintf(fp_xy_vi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy_vi,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy_vi,"%6f,",   v[idxy]*c_ref );
				}
				fprintf(fp_xy_vi,"\n");
			}
		}
		fclose(fp_xy_vi);
	// output instantaneous w //
		sprintf(str_xy_i,"./Output/xy_ins_w%08d_%04d.csv",t,rank);
		FILE *fp_xy_wi;
		fp_xy_wi = fopen(str_xy_i, "w");
		if (fp_xy_wi == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy_wi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy_wi,"%d,%f,",t,time);
		fprintf(fp_xy_wi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy_wi,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy_wi,"%6f,",   w[idxy]*c_ref );
				}
				fprintf(fp_xy_wi,"\n");
			}
		}
		fclose(fp_xy_wi);
        // output instantaneous t //
                sprintf(str_xy_i,"./Output/xy_ins_t%08d_%04d.csv",t,rank);
                FILE *fp_xy_ti;
                fp_xy_ti = fopen(str_xy_i, "w");
                if (fp_xy_ti == NULL) {
                        printf("cannot open\n");
                        exit(1);
                }
                fprintf(fp_xy_ti,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                fprintf(fp_xy_ti,"%d,%f,",t,time);
                fprintf(fp_xy_ti,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                for (k=0; k<nz_out; k++) {
                        fprintf(fp_xy_ti,"%4d\n",kout[k]);
                        for (j=halo; j<ny-halo; j++) {
                                for (i=halo; i<nx-halo; i++) {
                                        idxy = i + nx*j + nxy*kout[k];
                                        fprintf(fp_xy_ti,"%6f,",   T[idxy] );
                                }
                                fprintf(fp_xy_ti,"\n");
                        }
                }
                fclose(fp_xy_ti);



	// output instantaneous velocity distributions on xz slice at the center of subdomain
		for(jj=0; jj<nj_out; jj++) {
			if( (jout[jj] >=ny*rank_y) && (jout[jj] <=ny*(rank_y+1)) ) {
				char str_xz_i[100];		// for file name

	// output instantaneous u xz//
		sprintf(str_xz_i,"./Output/xz_ins_u%08d_%04d_%05d.csv",t,rank,jout[jj]);
				FILE *fp_xz_ui;
				fp_xz_ui = fopen(str_xz_i, "w");
				if (fp_xz_ui == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_xz_ui,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_xz_ui,"%d,%f,",t,time);
				fprintf(fp_xz_ui,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=0; k<nz; k++) {
					for (i=halo; i<nx-halo; i++) {
						idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
						fprintf(fp_xz_ui,"%6f,",   u[idxy]*c_ref );
					}
					fprintf(fp_xz_ui,"\n");
				}
				fclose(fp_xz_ui);
	// output instantaneous v xz//
		sprintf(str_xz_i,"./Output/xz_ins_v%08d_%04d_%05d.csv",t,rank,jout[jj]);
				FILE *fp_xz_vi;
				fp_xz_vi = fopen(str_xz_i, "w");
				if (fp_xz_vi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_xz_vi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_xz_vi,"%d,%f,",t,time);
				fprintf(fp_xz_vi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=0; k<nz; k++) {
					for (i=halo; i<nx-halo; i++) {
						idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
						fprintf(fp_xz_vi,"%6f,",   v[idxy]*c_ref );
					}
					fprintf(fp_xz_vi,"\n");
				}
				fclose(fp_xz_vi);
	// output instantaneous w xz//
		sprintf(str_xz_i,"./Output/xz_ins_w%08d_%04d_%05d.csv",t,rank,jout[jj]);
				FILE *fp_xz_wi;
				fp_xz_wi = fopen(str_xz_i, "w");
				if (fp_xz_wi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_xz_wi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_xz_wi,"%d,%f,",t,time);
				fprintf(fp_xz_wi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=0; k<nz; k++) {
					for (i=halo; i<nx-halo; i++) {
						idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
						fprintf(fp_xz_wi,"%6f,",   w[idxy]*c_ref );
					}
					fprintf(fp_xz_wi,"\n");
				}
				fclose(fp_xz_wi);
        // output instantaneous t xz//
                sprintf(str_xz_i,"./Output/xz_ins_t%08d_%04d_%05d.csv",t,rank,jout[jj]);
                                FILE *fp_xz_ti;
                                fp_xz_ti = fopen(str_xz_i, "w");
                                if (fp_xz_ti == NULL) {
                                        printf("cannot open\n");
                                        exit(1);
                                }
                                fprintf(fp_xz_ti,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                                fprintf(fp_xz_ti,"%d,%f,",t,time);
                                fprintf(fp_xz_ti,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                                for (k=0; k<nz; k++) {
                                        for (i=halo; i<nx-halo; i++) {
                                                idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
                                                fprintf(fp_xz_ti,"%6f,",   T[idxy] );
                                        }
                                        fprintf(fp_xz_ti,"\n");
                                }
                                fclose(fp_xz_ti);
			}
		}


	// output instantaneous velocity distributions on yz slice at the center of subdomain
		for(ii=0; ii<ni_out; ii++) {
			if( (iout[ii] >=nx*rank_x) && (iout[ii] <=nx*(rank_x+1)) ) {
				char str_yz_i[100];		// for file name
				i = iout[ii] - nx*rank_x;

	// output instantaneous u yz//
		sprintf(str_yz_i,"./Output/yz_ins_u%08d_%04d_%05d.csv",t,rank,iout[ii]);
				FILE *fp_yz_ui;
				fp_yz_ui = fopen(str_yz_i, "w");
				if (fp_yz_ui == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_yz_ui,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_yz_ui,"%d,%f,",t,time);
				fprintf(fp_yz_ui,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (j=halo; j<ny-halo; j++) {
						idxy = i + nx*j + nxy*k;
						fprintf(fp_yz_ui,"%6f,",   u[idxy]*c_ref );
					}
					fprintf(fp_yz_ui,"\n");
				}
				fclose(fp_yz_ui);
	// output instantaneous v yz//
		sprintf(str_yz_i,"./Output/yz_ins_v%08d_%04d_%05d.csv",t,rank,iout[ii]);
				FILE *fp_yz_vi;
				fp_yz_vi = fopen(str_yz_i, "w");
				if (fp_yz_vi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_yz_vi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_yz_vi,"%d,%f,",t,time);
				fprintf(fp_yz_vi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (j=halo; j<ny-halo; j++) {
						idxy = i + nx*j + nxy*k;
						fprintf(fp_yz_vi,"%6f,",   v[idxy]*c_ref );
					}
					fprintf(fp_yz_vi,"\n");
				}
				fclose(fp_yz_vi);
	// output instantaneous w yz//
		sprintf(str_yz_i,"./Output/yz_ins_w%08d_%04d_%05d.csv",t,rank,iout[ii]);
				FILE *fp_yz_wi;
				fp_yz_wi = fopen(str_yz_i, "w");
				if (fp_yz_wi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_yz_wi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_yz_wi,"%d,%f,",t,time);
				fprintf(fp_yz_wi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (j=halo; j<ny-halo; j++) {
						idxy = i + nx*j + nxy*k;
						fprintf(fp_yz_wi,"%6f,",   w[idxy]*c_ref );
					}
					fprintf(fp_yz_wi,"\n");
				}
				fclose(fp_yz_wi);
        // output instantaneous T yz//
                sprintf(str_yz_i,"./Output/yz_ins_t%08d_%04d_%05d.csv",t,rank,iout[ii]);
                                FILE *fp_yz_ti;
                                fp_yz_ti = fopen(str_yz_i, "w");
                                if (fp_yz_ti == NULL) {
                                        printf("cannot open\n");
                                        exit(1);
                                }
                                fprintf(fp_yz_ti,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                                fprintf(fp_yz_ti,"%d,%f,",t,time);
                                fprintf(fp_yz_ti,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                                for (k=2; k<nz-halo; k++) {
                                        for (j=halo; j<ny-halo; j++) {
                                                idxy = i + nx*j + nxy*k;
                                                fprintf(fp_yz_ti,"%6f,",   T[idxy] );
                                        }
                                        fprintf(fp_yz_ti,"\n");
                                }
                                fclose(fp_yz_ti);
			}
		}



	//=========================================================================================================


	// output instantaneous velocity in a volume
		for (i_vol=0; i_vol<nv_out; i_vol++) {
			if( rank == vout_rank[i_vol]) {
				char str_vo_i[100];		// for file name

	// output instantaneous u volume//
				sprintf(str_vo_i,"./Output/vo_ins_u%08d_%04d.csv",t,rank);
				FILE *fp_vo_ui;
				fp_vo_ui = fopen(str_vo_i, "w");
				if (fp_vo_ui == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_vo_ui,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_vo_ui,"%d,%f,",t,time);
				fprintf(fp_vo_ui,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (j=halo; j<ny-halo; j++) {
						for (i=halo; i<nx-halo; i++) {
							idxy = i + nx*j + nxy*k;
							fprintf(fp_vo_ui,"%6f,",   u[idxy]*c_ref );
						}
						fprintf(fp_vo_ui,"\n");
					}
				}
				fclose(fp_vo_ui);

	// output instantaneous v volume//
				sprintf(str_vo_i,"./Output/vo_ins_v%08d_%04d.csv",t,rank);
				FILE *fp_vo_vi;
				fp_vo_vi = fopen(str_vo_i, "w");
				if (fp_vo_vi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_vo_vi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_vo_vi,"%d,%f,",t,time);
				fprintf(fp_vo_vi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (j=halo; j<ny-halo; j++) {
						for (i=halo; i<nx-halo; i++) {
							idxy = i + nx*j + nxy*k;
							fprintf(fp_vo_vi,"%6f,",   v[idxy]*c_ref );
						}
						fprintf(fp_vo_vi,"\n");
					}
				}
				fclose(fp_vo_vi);

	// output instantaneous w volume//
				sprintf(str_vo_i,"./Output/vo_ins_w%08d_%04d.csv",t,rank);
				FILE *fp_vo_wi;
				fp_vo_wi = fopen(str_vo_i, "w");
				if (fp_vo_wi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
				fprintf(fp_vo_wi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
				fprintf(fp_vo_wi,"%d,%f,",t,time);
				fprintf(fp_vo_wi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (j=halo; j<ny-halo; j++) {
						for (i=halo; i<nx-halo; i++) {
							idxy = i + nx*j + nxy*k;
							fprintf(fp_vo_wi,"%6f,",   w[idxy]*c_ref );
						}
						fprintf(fp_vo_wi,"\n");
					}
				}
				fclose(fp_vo_wi);
        // output instantaneous t volume//
                                sprintf(str_vo_i,"./Output/vo_ins_t%08d_%04d.csv",t,rank);
                                FILE *fp_vo_ti;
                                fp_vo_ti = fopen(str_vo_i, "w");
                                if (fp_vo_ti == NULL) {
                                        printf("cannot open\n");
                                        exit(1);
                                }
                                fprintf(fp_vo_ti,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                                fprintf(fp_vo_ti,"%d,%f,",t,time);
                                fprintf(fp_vo_ti,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                                for (k=2; k<nz-halo; k++) {
                                        for (j=halo; j<ny-halo; j++) {
                                                for (i=halo; i<nx-halo; i++) {
                                                        idxy = i + nx*j + nxy*k;
                                                        fprintf(fp_vo_ti,"%6f,",   T[idxy] );
                                                }
                                                fprintf(fp_vo_ti,"\n");
                                        }
                                }
                                fclose(fp_vo_ti);
			}
		}


	//=========================================================================================================


		time_output_ins += output_interval_ins;
	}




	if (flag == 0) {
		time_average = skip_time + average_interval;
		// memory //
		nxy_air = new int[nz];
		n_obs	= new int[nxyz];	// 2018
		r_prof  = new double[nz];	// 2018
		u_prof  = new double[nz];
		v_prof  = new double[nz];
		w_prof  = new double[nz];
                t_prof  = new double[nz];
		uu_prof = new double[nz];
		vv_prof = new double[nz];
		ww_prof = new double[nz];
		tt_prof = new double[nz];
		uv_prof = new double[nz];
		uw_prof = new double[nz];
		vw_prof = new double[nz];
		ut_prof = new double[nz];
		vt_prof = new double[nz];
		wt_prof = new double[nz];
		uuu_prof = new double[nz];
		vvv_prof = new double[nz];
		www_prof = new double[nz];
		ttt_prof = new double[nz];
		uuw_prof = new double[nz];
		vvw_prof = new double[nz];
		u_xy  = new double[nxy_out];
		v_xy  = new double[nxy_out];
		w_xy  = new double[nxy_out];
		t_xy  = new double[nxy_out];
		uu_xy = new double[nxy_out];
		vv_xy = new double[nxy_out];
		ww_xy = new double[nxy_out];
		tt_xy = new double[nxy_out];
		uv_xy = new double[nxy_out];
		uw_xy = new double[nxy_out];
		vw_xy = new double[nxy_out];
		ut_xy = new double[nxy_out];
		vt_xy = new double[nxy_out];
		wt_xy = new double[nxy_out];
		wspeed_max = new float[nxy_out];
		wspeed_time = new int[nxy_out];
		nx_air  = new int[nxz];
		u_x   = new double[nxz];
		v_x   = new double[nxz];
		w_x   = new double[nxz];
		t_x   = new double[nxz];
		uu_x  = new double[nxz];
		vv_x  = new double[nxz];
		ww_x  = new double[nxz];
		tt_x  = new double[nxz];
		uv_x  = new double[nxz];
		uw_x  = new double[nxz];
		vw_x  = new double[nxz];
		ut_x  = new double[nxz];
		vt_x  = new double[nxz];
		wt_x  = new double[nxz];
		uuu_x = new double[nxz];
		vvv_x = new double[nxz];
		www_x = new double[nxz];
		ttt_x = new double[nxz];
		uuw_x = new double[nxz];
		vvw_x = new double[nxz];

		flag = 1;
		i_count = 0;
		// count grids in air //
		for (k=halo; k<nz-halo; k++) {
			nxy_air[k] = 0;
			for (i=halo; i<nx-halo; i++) {
				for (j=halo; j<ny-halo; j++) {
					id = i + nx*j + nxy*k;
					if ( (l_obs[id] == -1) ) {
						nxy_air[k] = nxy_air[k]+1;
						n_obs[id] = 1;
					}  else  {
						n_obs[id] = 0;
					}
				}
			}
			if(nxy_air[k]==0) nxy_air[k]=1;
		}
		// count grids in air //
		for (n=0; n<nxz; n++){
			nx_air[idxz] = 0;
		}
		for (k=halo; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				for (j=halo; j<ny-halo; j++) {
					id = i + nx*j + nxy*k;
					if ( (l_obs[id] == -1) ) {
						nx_air[idxz] = nx_air[idxz]+1;
					}
				}
				if(nx_air[idxz]==0) nx_air[idxz]=1;
			}
		}


	// Output header information //
		if(rank == 0) {
			char str_h[100];
			sprintf(str_h,"./Output/header.txt");
			FILE *fp_h;
			fp_h = fopen(str_h, "w");
			if (fp_h == NULL) {
				printf("cannot open\n");
				exit(1);
			}
			fprintf(fp_h,"ncpu     =%15d\n",ncpu   );
			fprintf(fp_h,"ncpu_x   =%15d\n",ncpu_x );
			fprintf(fp_h,"ncpu_y   =%15d\n",ncpu_y );
			fprintf(fp_h,"ncpu_z   =%15d\n",ncpu_z );
			fprintf(fp_h,"nx       =%15d\n",nx     );
			fprintf(fp_h,"ny       =%15d\n",ny     );
			fprintf(fp_h,"nz       =%15d\n",nz     );
			fprintf(fp_h,"nn       =%15d\n",nn     );
			fprintf(fp_h,"n0       =%15d\n",n0     );
			fprintf(fp_h,"nxg      =%15d\n",nxg    );
			fprintf(fp_h,"nyg      =%15d\n",nyg    );
			fprintf(fp_h,"nzg      =%15d\n",nzg    );
			fprintf(fp_h,"halo     =%15d\n",halo   );
			fprintf(fp_h,"xg_min   =%15f\n",xg_min );
			fprintf(fp_h,"yg_min   =%15f\n",yg_min );
			fprintf(fp_h,"zg_min   =%15f\n",zg_min );
			fprintf(fp_h,"dx       =%15f\n",dx     );
			fprintf(fp_h,"c_ref    =%15f\n",c_ref  );
			fprintf(fp_h,"cfl_ref  =%15f\n",cfl_ref);
			fprintf(fp_h,"dt_real  =%15f\n",dt_real);
			fclose(fp_h);
		}
	}

	// zero set //
	if (i_count == 0) {
		// zero set //
		for(k=0;k<nz;k++) {
			r_prof[k]  = 0.0;
			u_prof[k]  = 0.0;
			v_prof[k]  = 0.0;
			w_prof[k]  = 0.0;
			t_prof[k]  = 0.0;
			uu_prof[k] = 0.0;
			vv_prof[k] = 0.0;
			ww_prof[k] = 0.0;
			tt_prof[k] = 0.0;
			uv_prof[k] = 0.0;
			uw_prof[k] = 0.0;
			vw_prof[k] = 0.0;
			ut_prof[k] = 0.0;
			vt_prof[k] = 0.0;
			wt_prof[k] = 0.0;
			uuu_prof[k] = 0.0;
			vvv_prof[k] = 0.0;
			www_prof[k] = 0.0;
			ttt_prof[k] = 0.0;
			uuw_prof[k] = 0.0;
			vvw_prof[k] = 0.0;
		}

		for(n=0;n<nxy_out;n++) {
			u_xy[n]   = 0.0;
			v_xy[n]   = 0.0;
			w_xy[n]   = 0.0;
			t_xy[n]   = 0.0;
			uu_xy[n]  = 0.0;
			vv_xy[n]  = 0.0;
			ww_xy[n]  = 0.0;
			tt_xy[n]  = 0.0;
			uv_xy[n]  = 0.0;
			uw_xy[n]  = 0.0;
			vw_xy[n]  = 0.0;
			ut_xy[n]  = 0.0;
			vt_xy[n]  = 0.0;
			wt_xy[n]  = 0.0;
			wspeed_max[n] = 0.0;
			wspeed_time[n] = 0;
		}

		for(n=0;n<nxz;n++) {
			u_x[n]   = 0.0;
			v_x[n]   = 0.0;
			w_x[n]   = 0.0;
			t_x[n]   = 0.0;
			uu_x[n]  = 0.0;
			vv_x[n]  = 0.0;
			ww_x[n]  = 0.0;
			tt_x[n]  = 0.0;
			uv_x[n]  = 0.0;
			uw_x[n]  = 0.0;
			vw_x[n]  = 0.0;
			ut_x[n]  = 0.0;
			vt_x[n]  = 0.0;
			wt_x[n]  = 0.0;
			uuu_x[n] = 0.0;
			vvv_x[n] = 0.0;
			www_x[n] = 0.0;
			ttt_x[n] = 0.0;
			uuw_x[n] = 0.0;
			vvw_x[n] = 0.0;
		}

	}

	// integrate //

	if(time > skip_time) {
		i_count += 1;
		for (i=halo; i<nx-halo; i++) {
			for (j=halo; j<ny-halo; j++) {
				kk = 0;		// for output at specific heights //
				for (k=halo; k<nz-halo; k++) {
					id = i + nx*j + nxy*k;
// vertical profiles
					T0 = T[id] - 300.0;
					r_prof[k]  = r_prof[k]	+ r[id] * n_obs[id];
					u_prof[k]  = u_prof[k]  + u[id] * n_obs[id];
					v_prof[k]  = v_prof[k]  + v[id] * n_obs[id];
					w_prof[k]  = w_prof[k]  + w[id] * n_obs[id];
					t_prof[k]  = t_prof[k]  + T0 * n_obs[id];
					uu_prof[k] = uu_prof[k] + u[id] * u[id] * n_obs[id];
					vv_prof[k] = vv_prof[k] + v[id] * v[id] * n_obs[id];
					ww_prof[k] = ww_prof[k] + w[id] * w[id] * n_obs[id];
					tt_prof[k] = tt_prof[k] + T0 * T0 * n_obs[id];
					uv_prof[k] = uv_prof[k] + u[id] * v[id] * n_obs[id];
					uw_prof[k] = uw_prof[k] + u[id] * w[id] * n_obs[id];
					vw_prof[k] = vw_prof[k] + v[id] * w[id] * n_obs[id];
					ut_prof[k] = ut_prof[k] + u[id] * T0 * n_obs[id];
					vt_prof[k] = vt_prof[k] + v[id] * T0 * n_obs[id];
					wt_prof[k] = wt_prof[k] + w[id] * T0 * n_obs[id];
					uuu_prof[k] = uuu_prof[k] + u[id] * u[id] * u[id] * n_obs[id];
					vvv_prof[k] = vvv_prof[k] + v[id] * v[id] * v[id] * n_obs[id];
					www_prof[k] = www_prof[k] + w[id] * w[id] * w[id] * n_obs[id];
					ttt_prof[k] = ttt_prof[k] + T0 * T0 * T0 * n_obs[id];
					uuw_prof[k] = uuw_prof[k] + u[id] * u[id] * w[id] * n_obs[id];
					vvw_prof[k] = vvw_prof[k] + v[id] * v[id] * w[id] * n_obs[id];

// xz distribution (averaged in y direction)
					idxz = i + nx*k;
					u_x[idxz]    = u_x[idxz]    + u[id] * n_obs[id];
					v_x[idxz]    = v_x[idxz]    + v[id] * n_obs[id];
					w_x[idxz]    = w_x[idxz]    + w[id] * n_obs[id];
					t_x[idxz]    = t_x[idxz]    + T0 * n_obs[id];
					uu_x[idxz]   = uu_x[idxz]   + u[id] * u[id] * n_obs[id];
					vv_x[idxz]   = vv_x[idxz]   + v[id] * v[id] * n_obs[id];
					ww_x[idxz]   = ww_x[idxz]   + w[id] * w[id] * n_obs[id];
					tt_x[idxz]   = tt_x[idxz]   + T0    * T0 * n_obs[id];
					uv_x[idxz]   = uv_x[idxz]   + u[id] * v[id] * n_obs[id];
					uw_x[idxz]   = uw_x[idxz]   + u[id] * w[id] * n_obs[id];
					vw_x[idxz]   = vw_x[idxz]   + v[id] * w[id] * n_obs[id];
					ut_x[idxz]   = ut_x[idxz]   + u[id] * T0 * n_obs[id];
					vt_x[idxz]   = vt_x[idxz]   + v[id] * T0 * n_obs[id];
					wt_x[idxz]   = wt_x[idxz]   + w[id] * T0 * n_obs[id];
					uuu_x[idxz]  = uuu_x[idxz]  + u[id] * u[id] * u[id] * n_obs[id];
					vvv_x[idxz]  = vvv_x[idxz]  + v[id] * v[id] * v[id] * n_obs[id];
					www_x[idxz]  = www_x[idxz]  + w[id] * w[id] * w[id] * n_obs[id];
					ttt_x[idxz]  = ttt_x[idxz]  + T0    * T0    * T0 * n_obs[id];
					uuw_x[idxz]  = uuw_x[idxz]  + u[id] * u[id] * w[id] * n_obs[id];
					vvw_x[idxz]  = vvw_x[idxz]  + v[id] * v[id] * w[id] * n_obs[id];

// horizontal distributions
					if(k == kout[kk]) {
						idxy = i + nx*j + nxy*kk;
						u_xy[idxy]  = u_xy[idxy]  + u[id] * n_obs[id];
						v_xy[idxy]  = v_xy[idxy]  + v[id] * n_obs[id];
						w_xy[idxy]  = w_xy[idxy]  + w[id] * n_obs[id];
						t_xy[idxy]  = t_xy[idxy]  + T0 * n_obs[id];
						uu_xy[idxy] = uu_xy[idxy] + u[id] * u[id] * n_obs[id];
						vv_xy[idxy] = vv_xy[idxy] + v[id] * v[id] * n_obs[id];
						ww_xy[idxy] = ww_xy[idxy] + w[id] * w[id] * n_obs[id];
						tt_xy[idxy] = tt_xy[idxy] + T0 * T0 * n_obs[id];
						uv_xy[idxy] = uv_xy[idxy] + u[id] * v[id] * n_obs[id];
						uw_xy[idxy] = uw_xy[idxy] + u[id] * w[id] * n_obs[id];
						vw_xy[idxy] = vw_xy[idxy] + v[id] * w[id] * n_obs[id];
						ut_xy[idxy] = ut_xy[idxy] + u[id] * T0 * n_obs[id];
						vt_xy[idxy] = vt_xy[idxy] + v[id] * T0 * n_obs[id];
						wt_xy[idxy] = wt_xy[idxy] + w[id] * T0 * n_obs[id];
						wspeed = sqrt(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]) * n_obs[id];
						if(wspeed > wspeed_max[idxy]) {
							wspeed_max[idxy] = wspeed;
							wspeed_time[idxy] = t;
						}
						if(kk<nz_out) kk = kk + 1;
					}
				}
			}
		}
	}

	// output if time arrived time_average //

	if ( time >= time_average )  {
// vertical profiles
		char str_pr[100];
		sprintf(str_pr,"./Output/prof%08d_%04d.csv",t,rank);
		
		FILE *fp_pr;
		fp_pr = fopen(str_pr, "w");
		if (fp_pr == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_pr,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_pr,"%d,%f,",t,time);
		fprintf(fp_pr,"%f,%f,%f\n",xg_min,yg_min,zg_min);

		fprintf(fp_pr,"z,RHO,U,V,W,UU,VV,WW,UV,UW,VW,");
		fprintf(fp_pr,"UUU,VVV,WWW,UUW,VVW,");
		fprintf(fp_pr,"T,TT,UT,VT,WT,TTT\n");

		for (k=halo+1; k<nz-halo; k++) {
			r_prof[k]   = (r_prof[k] /i_count/nxy_air[k]);
			u_prof[k]   = (u_prof[k] /i_count/nxy_air[k])*c_ref;
			v_prof[k]   = (v_prof[k] /i_count/nxy_air[k])*c_ref;
			w_prof[k]   = (w_prof[k] /i_count/nxy_air[k])*c_ref;
			uu_prof[k]  = (uu_prof[k]/i_count/nxy_air[k])*c_ref*c_ref;
			vv_prof[k]  = (vv_prof[k]/i_count/nxy_air[k])*c_ref*c_ref;
			ww_prof[k]  = (ww_prof[k]/i_count/nxy_air[k])*c_ref*c_ref;
			uv_prof[k]  = (uv_prof[k]/i_count/nxy_air[k])*c_ref*c_ref;
			uw_prof[k]  = (uw_prof[k]/i_count/nxy_air[k])*c_ref*c_ref;
			vw_prof[k]  = (vw_prof[k]/i_count/nxy_air[k])*c_ref*c_ref;
			uuu_prof[k] = (uuu_prof[k]/i_count/nxy_air[k])*c_ref*c_ref*c_ref;
			vvv_prof[k] = (vvv_prof[k]/i_count/nxy_air[k])*c_ref*c_ref*c_ref;
			www_prof[k] = (www_prof[k]/i_count/nxy_air[k])*c_ref*c_ref*c_ref;
			uuw_prof[k] = (uuw_prof[k]/i_count/nxy_air[k])*c_ref*c_ref*c_ref;
			vvw_prof[k] = (vvw_prof[k]/i_count/nxy_air[k])*c_ref*c_ref*c_ref;
			t_prof[k]   = (t_prof[k] /i_count/nxy_air[k]);
			tt_prof[k]  = (tt_prof[k]/i_count/nxy_air[k]);
			ut_prof[k]  = (ut_prof[k]/i_count/nxy_air[k])*c_ref;
			vt_prof[k]  = (vt_prof[k]/i_count/nxy_air[k])*c_ref;
			wt_prof[k]  = (wt_prof[k]/i_count/nxy_air[k])*c_ref;
			ttt_prof[k] = (ttt_prof[k]/i_count/nxy_air[k]);
		}
		for (k=halo+1; k<nz-halo; k++) {
			fprintf(fp_pr,"%6f,",	dx*k		);
			fprintf(fp_pr,"%6f,",	r_prof[k]	);
			fprintf(fp_pr,"%6f,",	u_prof[k]	);
			fprintf(fp_pr,"%6f,",	v_prof[k]	);
			fprintf(fp_pr,"%6f,",	w_prof[k]	);
			fprintf(fp_pr,"%6f,",	uu_prof[k]	);
			fprintf(fp_pr,"%6f,",	vv_prof[k]	);
			fprintf(fp_pr,"%6f,",	ww_prof[k]	);
			fprintf(fp_pr,"%6f,",	uv_prof[k]	);
			fprintf(fp_pr,"%6f,",	uw_prof[k]	);
			fprintf(fp_pr,"%6f,",	vw_prof[k]	);
			fprintf(fp_pr,"%6f,",	uuu_prof[k]	);
			fprintf(fp_pr,"%6f,",	vvv_prof[k]	);
			fprintf(fp_pr,"%6f,",	www_prof[k]	);
			fprintf(fp_pr,"%6f,",	uuw_prof[k]	);
			fprintf(fp_pr,"%6f,",	vvw_prof[k]	);
			fprintf(fp_pr,"%6f,",	t_prof[k]	);
			fprintf(fp_pr,"%6f,",	tt_prof[k]	);
			fprintf(fp_pr,"%6f,",	ut_prof[k] 	);
			fprintf(fp_pr,"%6f,",	vt_prof[k]	);
			fprintf(fp_pr,"%6f,",	wt_prof[k]	);
			fprintf(fp_pr,"%6f\n",	ttt_prof[k]	);
		}
		fclose(fp_pr);

// =============================

// xz distribution (average in y and time) -- AVERAGE
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				u_x[idxz] = u_x[idxz] / i_count / nx_air[idxz] * c_ref;
				v_x[idxz] = v_x[idxz] / i_count / nx_air[idxz] * c_ref;
				w_x[idxz] = w_x[idxz] / i_count / nx_air[idxz] * c_ref;
				t_x[idxz] = t_x[idxz] / i_count / nx_air[idxz];
				uu_x[idxz] = uu_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref;
				vv_x[idxz] = vv_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref;
				ww_x[idxz] = ww_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref;
				tt_x[idxz] = tt_x[idxz] / i_count / nx_air[idxz];
				uv_x[idxz] = uv_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref;
				uw_x[idxz] = uw_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref;
				vw_x[idxz] = vw_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref;
				ut_x[idxz] = ut_x[idxz] / i_count / nx_air[idxz] * c_ref;
				vt_x[idxz] = vt_x[idxz] / i_count / nx_air[idxz] * c_ref;
				wt_x[idxz] = wt_x[idxz] / i_count / nx_air[idxz] * c_ref;
				uuu_x[idxz] = uuu_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref * c_ref;
				vvv_x[idxz] = vvv_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref * c_ref;
				www_x[idxz] = www_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref * c_ref;
				ttt_x[idxz] = ttt_x[idxz] / i_count / nx_air[idxz] * c_ref;
				uuw_x[idxz] = uuw_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref * c_ref;
				vvw_x[idxz] = vvw_x[idxz] / i_count / nx_air[idxz] * c_ref * c_ref * c_ref;

				uu_x[idxz] = uu_x[idxz] - u_x[idxz] * u_x[idxz];
				vv_x[idxz] = vv_x[idxz] - v_x[idxz] * v_x[idxz];
				ww_x[idxz] = ww_x[idxz] - w_x[idxz] * w_x[idxz];
				tt_x[idxz] = tt_x[idxz] - t_x[idxz] * t_x[idxz];
				uv_x[idxz] = uv_x[idxz] - u_x[idxz] * v_x[idxz];
				uw_x[idxz] = uw_x[idxz] - u_x[idxz] * w_x[idxz];
				vw_x[idxz] = vw_x[idxz] - v_x[idxz] * w_x[idxz];
				ut_x[idxz] = ut_x[idxz] - u_x[idxz] * t_x[idxz];
				vt_x[idxz] = vt_x[idxz] - v_x[idxz] * t_x[idxz];
				wt_x[idxz] = wt_x[idxz] - w_x[idxz] * t_x[idxz];

				uuu_x[idxz] = uuu_x[idxz] - u_x[idxz]*u_x[idxz]*u_x[idxz] - uu_x[idxz]*u_x[idxz] - uu_x[idxz]*u_x[idxz] - uu_x[idxz]*u_x[idxz];
				vvv_x[idxz] = vvv_x[idxz] - v_x[idxz]*v_x[idxz]*v_x[idxz] - vv_x[idxz]*v_x[idxz] - vv_x[idxz]*v_x[idxz] - vv_x[idxz]*v_x[idxz];
				www_x[idxz] = www_x[idxz] - w_x[idxz]*w_x[idxz]*w_x[idxz] - ww_x[idxz]*w_x[idxz] - ww_x[idxz]*w_x[idxz] - ww_x[idxz]*w_x[idxz];
				ttt_x[idxz] = ttt_x[idxz] - t_x[idxz]*t_x[idxz]*t_x[idxz] - tt_x[idxz]*t_x[idxz] - tt_x[idxz]*t_x[idxz] - tt_x[idxz]*t_x[idxz];
				uuw_x[idxz] = uuw_x[idxz] - u_x[idxz]*u_x[idxz]*w_x[idxz] - uw_x[idxz]*u_x[idxz] - uw_x[idxz]*u_x[idxz] - uu_x[idxz]*w_x[idxz];
				vvw_x[idxz] = vvw_x[idxz] - v_x[idxz]*v_x[idxz]*w_x[idxz] - vw_x[idxz]*v_x[idxz] - vw_x[idxz]*v_x[idxz] - vv_x[idxz]*w_x[idxz];
			}
		}


// xz distribution (average in y and time)
		char str_xz_yav[100];		// for file name

// xz distribution (average in y and time) -- um
		sprintf(str_xz_yav,"./Output/xz_yav_um%08d_%04d.csv",t,rank);
		FILE *fp_xz_yav;
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", u_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- vm
		sprintf(str_xz_yav,"./Output/xz_yav_vm%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", v_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- wm
		sprintf(str_xz_yav,"./Output/xz_yav_wm%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", w_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- Tm
		sprintf(str_xz_yav,"./Output/xz_yav_tm%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", t_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- uu
		sprintf(str_xz_yav,"./Output/xz_yav_uu%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", uu_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- vv
		sprintf(str_xz_yav,"./Output/xz_yav_vv%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", vv_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- ww
		sprintf(str_xz_yav,"./Output/xz_yav_ww%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", ww_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- tt
		sprintf(str_xz_yav,"./Output/xz_yav_tt%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", tt_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- uv
		sprintf(str_xz_yav,"./Output/xz_yav_uv%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", uv_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- uw
		sprintf(str_xz_yav,"./Output/xz_yav_uw%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", uw_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- vw
		sprintf(str_xz_yav,"./Output/xz_yav_vw%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", vw_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- ut
		sprintf(str_xz_yav,"./Output/xz_yav_ut%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", ut_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- vt
		sprintf(str_xz_yav,"./Output/xz_yav_vt%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", vt_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- wt
		sprintf(str_xz_yav,"./Output/xz_yav_wt%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", wt_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- uuu
		sprintf(str_xz_yav,"./Output/xz_yav_uuu%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", uuu_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- vvv
		sprintf(str_xz_yav,"./Output/xz_yav_vvv%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", vvv_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- www
		sprintf(str_xz_yav,"./Output/xz_yav_www%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", www_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- ttt
		sprintf(str_xz_yav,"./Output/xz_yav_ttt%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", ttt_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- uuw
		sprintf(str_xz_yav,"./Output/xz_yav_uuw%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", uuw_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);


// xz distribution (average in y and time) -- vvw
		sprintf(str_xz_yav,"./Output/xz_yav_vvw%08d_%04d.csv",t,rank);
		fp_xz_yav = fopen(str_xz_yav, "w");
		if (fp_xz_yav == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xz_yav,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xz_yav,"%d,%f,",t,time);
		fprintf(fp_xz_yav,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=2; k<nz-halo; k++) {
			for (i=halo; i<nx-halo; i++) {
				idxz = i + nx*k;
				fprintf(fp_xz_yav,"%6f,", vvw_x[idxz]	);
			}
			fprintf(fp_xz_yav,"\n");
		}
		fclose(fp_xz_yav);



// ==============================  AAA


// horizontal distributions
		char str_xy[100];		// for file name

// horizontal distributions -- u
		sprintf(str_xy,"./Output/xy_um%08d_%04d.csv",t,rank);
		FILE *fp_xy;
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (u_xy[idxy]/i_count)*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);
// horizontal distributions -- v
		sprintf(str_xy,"./Output/xy_vm%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (v_xy[idxy]/i_count)*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- w
		sprintf(str_xy,"./Output/xy_wm%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (w_xy[idxy]/i_count)*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- T
                sprintf(str_xy,"./Output/xy_tm%08d_%04d.csv",t,rank);
                fp_xy = fopen(str_xy, "w");
                if (fp_xy == NULL) {
                        printf("cannot open\n");
                        exit(1);
                }
                fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                fprintf(fp_xy,"%d,%f,",t,time);
                fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                for (k=0; k<nz_out; k++) {
                        fprintf(fp_xy,"%4d\n",kout[k]);
                        for (j=halo; j<ny-halo; j++) {
                                for (i=halo; i<nx-halo; i++) {
                                        idxy = i + nx*j + nxy*k;
                                        fprintf(fp_xy,"%6f,",   (t_xy[idxy]/i_count) );
                                }
                                fprintf(fp_xy,"\n");
                        }
                }
                fclose(fp_xy);

// horizontal distributions -- uu
		sprintf(str_xy,"./Output/xy_uu%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (uu_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- vv
		sprintf(str_xy,"./Output/xy_vv%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (vv_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- ww
		sprintf(str_xy,"./Output/xy_ww%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (ww_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- TT
                sprintf(str_xy,"./Output/xy_tt%08d_%04d.csv",t,rank);
                fp_xy = fopen(str_xy, "w");
                if (fp_xy == NULL) {
                        printf("cannot open\n");
                        exit(1);
                }
                fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                fprintf(fp_xy,"%d,%f,",t,time);
                fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                for (k=0; k<nz_out; k++) {
                        fprintf(fp_xy,"%4d\n",kout[k]);
                        for (j=halo; j<ny-halo; j++) {
                                for (i=halo; i<nx-halo; i++) {
                                        idxy = i + nx*j + nxy*k;
                                        fprintf(fp_xy,"%6f,",   (tt_xy[idxy]/i_count) );
                                }
                                fprintf(fp_xy,"\n");
                        }
                }
                fclose(fp_xy);

// horizontal distributions -- uv
		sprintf(str_xy,"./Output/xy_uv%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (uv_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- uw
		sprintf(str_xy,"./Output/xy_uw%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (uw_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- vw
		sprintf(str_xy,"./Output/xy_vw%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   (vw_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- uT
                sprintf(str_xy,"./Output/xy_ut%08d_%04d.csv",t,rank);
                fp_xy = fopen(str_xy, "w");
                if (fp_xy == NULL) {
                        printf("cannot open\n");
                        exit(1);
                }
                fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                fprintf(fp_xy,"%d,%f,",t,time);
                fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                for (k=0; k<nz_out; k++) {
                        fprintf(fp_xy,"%4d\n",kout[k]);
                        for (j=halo; j<ny-halo; j++) {
                                for (i=halo; i<nx-halo; i++) {
                                        idxy = i + nx*j + nxy*k;
                                        fprintf(fp_xy,"%6f,",   (ut_xy[idxy]/i_count)*c_ref );
                                }
                                fprintf(fp_xy,"\n");
                        }
                }
                fclose(fp_xy);

// horizontal distributions -- vT
                sprintf(str_xy,"./Output/xy_vt%08d_%04d.csv",t,rank);
                fp_xy = fopen(str_xy, "w");
                if (fp_xy == NULL) {
                        printf("cannot open\n");
                        exit(1);
                }       
                fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                fprintf(fp_xy,"%d,%f,",t,time);
                fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                for (k=0; k<nz_out; k++) {
                        fprintf(fp_xy,"%4d\n",kout[k]);
                        for (j=halo; j<ny-halo; j++) {
                                for (i=halo; i<nx-halo; i++) {
                                        idxy = i + nx*j + nxy*k;
                                        fprintf(fp_xy,"%6f,",   (vt_xy[idxy]/i_count)*c_ref );
                                }
                                fprintf(fp_xy,"\n");
                        }
                }
                fclose(fp_xy);

// horizontal distributions -- wT
                sprintf(str_xy,"./Output/xy_wt%08d_%04d.csv",t,rank);
                fp_xy = fopen(str_xy, "w");
                if (fp_xy == NULL) {
                        printf("cannot open\n");
                        exit(1);
                }
                fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
                fprintf(fp_xy,"%d,%f,",t,time);
                fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
                for (k=0; k<nz_out; k++) {
                        fprintf(fp_xy,"%4d\n",kout[k]);
                        for (j=halo; j<ny-halo; j++) {
                                for (i=halo; i<nx-halo; i++) {
                                        idxy = i + nx*j + nxy*k;
                                        fprintf(fp_xy,"%6f,",   (wt_xy[idxy]/i_count)*c_ref );
                                }
                                fprintf(fp_xy,"\n");
                        }
                }
                fclose(fp_xy);

// horizontal distributions -- wspeed_max
		sprintf(str_xy,"./Output/xy_gs%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%6f,",   wspeed_max[idxy]*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- wspeed_time
		sprintf(str_xy,"./Output/xy_gt%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<nz_out; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*k;
					fprintf(fp_xy,"%8d,",   wspeed_time[idxy] );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

		if(rank == 0) printf("> UserOutput time=%f count=%d\n",time,i_count);
		time_average = time_average + average_interval;
		i_count = 0;		// end of output mean statistics //
	}


	// user define function //
}


// (YOKOUCHI 2020)
void
output_vis_sgs (
	int			t,
	char			*program_name,
	int			argc,
	char			*argv[],
	const paramMPI		&pmpi,
	const paramDomain	&pdomain,
	const Stress		*const str,
	const BasisVariables	*const cbq,
	const FluidProperty	*const cfp
	)
{
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);

	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// mpi //
	int	rank    = pmpi.rank();
	int	rank_x  = pmpi.rank_x();
	int	rank_y  = pmpi.rank_y();
	int	rank_z  = pmpi.rank_z();

	int	ncpu    = pmpi.ncpu();
	int	ncpu_x  = pmpi.ncpu_x();
	int	ncpu_y  = pmpi.ncpu_y();
	int	ncpu_z  = pmpi.ncpu_z();


	// domain //
	int	nx  = pdomain.nx();
	int	ny  = pdomain.ny();
	int	nz  = pdomain.nz();
	int	nn  = pdomain.nn();

	int	n0  = pdomain.n0();

	int	nxg  = pdomain.nxg();
	int	nyg  = pdomain.nyg();
	int	nzg  = pdomain.nzg();

	int	halo  = pdomain.halo();


	FLOAT	xg_min  = pdomain.xg_min();
	FLOAT	yg_min  = pdomain.yg_min();
	FLOAT	zg_min  = pdomain.zg_min();

	FLOAT	dx  = pdomain.dx();

	FLOAT	c_ref   = pdomain.c_ref();		// sound speed //
	FLOAT	cfl_ref = pdomain.cfl_ref();

//	FLOAT	dt_real = 2.0 / c_ref;			// dt //
	FLOAT	dt_real = dx  / c_ref;

	// user function //
	
	int i, ii, j, jj, k, kk, n, id, idxy, idxz, i_vol;

	float	time = t * dt_real;
	float	average_interval	= user_output::average_interval;
	float	skip_time		= user_output::skip_time;
	float   output_interval_ins	= user_output::output_interval_ins;
	float	time_output_ins_ini	= user_output::time_output_ins_ini;
	float	wspeed;
	float	T0;		// T0 = T - 300.0
	int	nz_out			= user_output::nz_out;
	int	nj_out			= user_output::nj_out;
	int	ni_out			= user_output::ni_out;
	int	nv_out			= user_output::nv_out;
	int	nxy = nx * ny;
	int	nxz = nx * nz;
//	int	nyz = ny * nz;
	int	nxyz= nx * ny * nz;
	int	nxy_out = nx * ny * nz_out;

	static bool	flag;
	static bool	flag_ins;
	static int	i_count;
	static float	time_average;
	static float    time_output_ins;
	static int	*nx_air;
	static int	*nxy_air;
	static int	*n_obs;		// 2018
	
	// variables //
//	const FLOAT	*vis	= str->vis_sgs;
	const FLOAT	*tke 	= str->TKE_sgs;
	const FLOAT   	*um  	= str->u_m;
	const FLOAT   	*vm  	= str->v_m;
	const FLOAT   	*wm  	= str->w_m;
	const FLOAT   	*u	= cbq->u_n;
	const FLOAT	*v	= cbq->v_n;
	const FLOAT	*w	= cbq->w_n;
	
	FLOAT	*l_obs = cfp->l_obs;

	// for output profile //
//	double 	*vis_prof;
	double  *um_prof;
	double  *vm_prof;
	double  *wm_prof;
	double	*tke_prof;
	double	*fs_prof;

	// output instantaneous vis_sgs prof (plane average)//
	nxy_air  = new int[nz];
	n_obs	 = new int[nxyz];
//	vis_prof = new double[nz];	
	um_prof  = new double[nz];
	vm_prof  = new double[nz];
	wm_prof  = new double[nz];
	tke_prof = new double[nz];
	fs_prof  = new double[nz];
	
	flag 	= 1;
	i_count = 0;
	// count grids in air //
	for (k=halo; k<nz-halo; k++) {
		nxy_air[k] = 0;
		for (j=halo; j<ny-halo; j++) {
			for (i=halo; i<nx-halo; i++) {
				id = i + nx*j + nxy*k;
				if ( (l_obs[id] == -1) ) {
					nxy_air[k] = nxy_air[k]+1;
					n_obs[id] = 1;
				}  else  {
					n_obs[id] = 0;
				}
			}
		}
		if(nxy_air[k]==0) nxy_air[k]=1;
	}
	
	// zero set //
	if (i_count == 0) {
		for (k=0; k<nz; k++) {
//			vis_prof[k] = 0.0;
			um_prof[k]  = 0.0;
			vm_prof[k]  = 0.0;
			wm_prof[k]  = 0.0;
			tke_prof[k] = 0.0;
			fs_prof[k]  = 0.0;
		}
	

	// integrate //
	for (k=halo; k<nz-halo; k++) {
		for (j=halo; j<ny-halo; j++) {
			for (i=halo; i<nx-halo; i++) {
				id = i + nx*j + nxy*k;
//				vis_prof[k] = vis_prof[k] + vis[id] * n_obs[id];
				um_prof[k]  = um_prof[k]  + um[id]  * n_obs[id];
				vm_prof[k]  = vm_prof[k]  + vm[id]  * n_obs[id];
				wm_prof[k]  = wm_prof[k]  + wm[id]  * n_obs[id];
				tke_prof[k] = tke_prof[k] + tke[id] * n_obs[id];

				// calculation fs for LSM //
				double k_GS = 0.5 * ((u[id]-um[id])*(u[id]-um[id]) + (v[id]-vm[id])*(v[id]-vm[id]) + (w[id]-wm[id])*(w[id]-wm[id]));
				double fs   = tke[id] / (k_GS + tke[id]);
				fs_prof[k]  = fs_prof[k]  + fs * n_obs[id];
			}
		}
	}

	// output //
	char str_pr[100];
	sprintf(str_pr, "./Output/prof_fs%08d_%04d.csv", t, rank);
	
	FILE *fp_pr;
	fp_pr = fopen(str_pr, "w");
	if (fp_pr == NULL) {
		printf("cannot opne\n");
		exit(1);
	}
	for (k=halo+1; k<nz-halo; k++) {
//		vis_prof[k] = (vis_prof[k] / nxy_air[k])*c_ref*dx;
		um_prof[k]  = (um_prof[k]  / nxy_air[k])*c_ref;
		vm_prof[k]  = (vm_prof[k]  / nxy_air[k])*c_ref;
		wm_prof[k]  = (wm_prof[k]  / nxy_air[k])*c_ref;
		tke_prof[k] = (tke_prof[k] / nxy_air[k])*c_ref*c_ref;
		fs_prof[k]  = (fs_prof[k]  / nxy_air[k]);
	}
	for (k=halo+1; k<nz-halo; k++) {
		fprintf(fp_pr, "%f,%f,%f,%f,%f\n", um_prof[k], vm_prof[k], wm_prof[k], tke_prof[k], fs_prof[k]);
	}
	fclose(fp_pr);
	}

	delete [] nxy_air;
	delete [] n_obs;
//	delete [] vis_prof;	
	delete [] tke_prof;
	delete [] um_prof;
	delete [] vm_prof;
	delete [] wm_prof;
	delete [] fs_prof;
}	
