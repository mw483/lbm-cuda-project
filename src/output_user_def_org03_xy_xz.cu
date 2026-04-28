#include "output_user_def.h"

#include <iostream>
#include "option_parser.h"
#include "functionLib.h"
#include "fileLib.h"
#include "defineCoefficient.h"
#include "defineReferenceVel.h"
#include "defineBoundaryFlag.h"

#include <stdlib.h>
#include <math.h>

void
output_user_def (
	int						t,
	char					*program_name,
	int						argc,
	char					*argv[], 
	const paramMPI			&pmpi, 
	const paramDomain		&pdomain,
	const BasisVariables	*const cbq,
	const FluidProperty		*const cfp
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

	FLOAT	dt_real = 2.0 / c_ref;			// dt //


	// variables //
	// FLOAT	*r = cbq->r_n;
	FLOAT	*u = cbq->u_n;
	FLOAT	*v = cbq->v_n;
	FLOAT	*w = cbq->w_n;

	FLOAT	*l_obs = cfp->l_obs;

	// user define function //

	int i, j, jj, k, kk, n, id, idxy;

    float	time = t * dt_real;
	float	average_interval =    600.0;		// ***********************************************
	float	skip_time        =    600.0;		// ***********************************************
	float   output_interval_ins =  20000.0;		// ***********************************************
	float	time_output_ins_ini =  20.0;		// ***********************************************
	float	wspeed;
    int	nz_out = 12;							// ***********************************************
    int	nj_out = 2;								// ***********************************************
    int nxy = nx * ny;
	int	nxy_out = nx * ny * nz_out;

	static bool		flag;
	static bool		flag_ins;
	static int		i_count;
	static float	time_average;
	static float    time_output_ins;
	static int		*nxy_air;

	// for output profile //
	static 	float	*u_prof;
	static 	float	*v_prof;
	static 	float	*w_prof;
	static 	float	*uu_prof;
	static 	float	*vv_prof;
	static 	float	*ww_prof;
	static 	float	*uv_prof;
	static 	float	*uw_prof;
	static 	float	*vw_prof;
	// for output xy //
	int kout[] = {2,3,4,5,6,8,10,20,40,80,150,200};		// **********************************
    int jout[] = {100,200};								// **********************************
	static 	float	*u_xy;
	static 	float	*v_xy;
	static 	float	*w_xy;
	static 	float	*uu_xy;
	static 	float	*vv_xy;
	static 	float	*ww_xy;
	static 	float	*uv_xy;
	static 	float	*uw_xy;
	static 	float	*vw_xy;
	static 	float	*wspeed_max;

	if(rank == 0) printf("time=%f (i=%d)\n",time,t);

// instantaneous output //
	if (flag_ins == 0) {
		time_output_ins = time_output_ins_ini;
		flag_ins = 1;
	}
	if (time >= time_output_ins) {
	// output instantaneous u //
		char str_xy_i[100];		// for file name
		sprintf(str_xy_i,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_ins_u%08d_%04d.csv",t,rank);
		FILE *fp_xy_ui;
		fp_xy_ui = fopen(str_xy_i, "w");
		if (fp_xy_ui == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy_ui,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy_ui,"%d,%f,",t,time);
		fprintf(fp_xy_ui,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
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
		sprintf(str_xy_i,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_ins_v%08d_%04d.csv",t,rank);
		FILE *fp_xy_vi;
		fp_xy_vi = fopen(str_xy_i, "w");
		if (fp_xy_vi == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy_vi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy_vi,"%d,%f,",t,time);
		fprintf(fp_xy_vi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
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
		sprintf(str_xy_i,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_ins_w%08d_%04d.csv",t,rank);
		FILE *fp_xy_wi;
		fp_xy_wi = fopen(str_xy_i, "w");
		if (fp_xy_wi == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy_wi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy_wi,"%d,%f,",t,time);
		fprintf(fp_xy_wi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
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


	// output instantaneous velocity distributions on xz slice at the center of subdomain
		for(jj=0; jj<nj_out; jj++) {
			if( (jout[jj] >=ny*rank_y) && (jout[jj] <=ny*(rank_y+1)) ) {
				char str_xz_i[100];		// for file name

	// output instantaneous u xz//
		sprintf(str_xz_i,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xz_ins_u%08d_%04d_%05d.csv",t,rank,jout[jj]);
				FILE *fp_xz_ui;
				fp_xz_ui = fopen(str_xz_i, "w");
				if (fp_xz_ui == NULL) {
					printf("cannot open\n");
					exit(1);
				}
	//			fprintf(fp_xz_ui,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
	//			fprintf(fp_xz_ui,"%d,%f,",t,time);
	//			fprintf(fp_xz_ui,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (i=halo; i<nx-halo; i++) {
						idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
						fprintf(fp_xz_ui,"%6f,",   u[idxy]*c_ref );
					}
					fprintf(fp_xz_ui,"\n");
				}
				fclose(fp_xz_ui);
	// output instantaneous v xz//
		sprintf(str_xz_i,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xz_ins_v%08d_%04d_%05d.csv",t,rank,jout[jj]);
				FILE *fp_xz_vi;
				fp_xz_vi = fopen(str_xz_i, "w");
				if (fp_xz_vi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
	//			fprintf(fp_xz_vi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
	//			fprintf(fp_xz_vi,"%d,%f,",t,time);
	//			fprintf(fp_xz_vi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (i=halo; i<nx-halo; i++) {
						idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
						fprintf(fp_xz_vi,"%6f,",   v[idxy]*c_ref );
					}
					fprintf(fp_xz_vi,"\n");
				}
				fclose(fp_xz_vi);
	// output instantaneous w xz//
		sprintf(str_xz_i,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xz_ins_w%08d_%04d_%05d.csv",t,rank,jout[jj]);
				FILE *fp_xz_wi;
				fp_xz_wi = fopen(str_xz_i, "w");
				if (fp_xz_wi == NULL) {
					printf("cannot open\n");
					exit(1);
				}
	//			fprintf(fp_xz_wi,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
	//			fprintf(fp_xz_wi,"%d,%f,",t,time);
	//			fprintf(fp_xz_wi,"%f,%f,%f\n",xg_min,yg_min,zg_min);
				for (k=2; k<nz-halo; k++) {
					for (i=halo; i<nx-halo; i++) {
						idxy = i + nx*(jout[jj]-ny*rank_y) + nxy*k;
						fprintf(fp_xz_wi,"%6f,",   w[idxy]*c_ref );
					}
					fprintf(fp_xz_wi,"\n");
				}
				fclose(fp_xz_wi);
			}
		}
		time_output_ins += output_interval_ins;
	}

	if (flag == 0) {
		time_average = skip_time + average_interval;
		// memory //
		nxy_air = new int[nz];
		u_prof  = new float[nz];
		v_prof  = new float[nz];
		w_prof  = new float[nz];
		uu_prof = new float[nz];
		vv_prof = new float[nz];
		ww_prof = new float[nz];
		uv_prof = new float[nz];
		uw_prof = new float[nz];
		vw_prof = new float[nz];
		u_xy  = new float[nxy_out];
		v_xy  = new float[nxy_out];
		w_xy  = new float[nxy_out];
		uu_xy = new float[nxy_out];
		vv_xy = new float[nxy_out];
		ww_xy = new float[nxy_out];
		uv_xy = new float[nxy_out];
		uw_xy = new float[nxy_out];
		vw_xy = new float[nxy_out];
		wspeed_max = new float[nxy_out];
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
					}
				}
			}
			if(nxy_air[k]==0) nxy_air[k]=1;
		}
	// Output header information //
		if(rank == 0) {
			char str_h[100];
			sprintf(str_h,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/header.txt");
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
			u_prof[k]  = 0.0;
			v_prof[k]  = 0.0;
			w_prof[k]  = 0.0;
			uu_prof[k] = 0.0;
			vv_prof[k] = 0.0;
			ww_prof[k] = 0.0;
			uv_prof[k] = 0.0;
			uw_prof[k] = 0.0;
			vw_prof[k] = 0.0;
		}

		for(n=0;n<nxy_out;n++) {
			u_xy[n]   = 0.0;
			v_xy[n]   = 0.0;
			w_xy[n]   = 0.0;
			uu_xy[n]  = 0.0;
			vv_xy[n]  = 0.0;
			ww_xy[n]  = 0.0;
			uv_xy[n]  = 0.0;
			uw_xy[n]  = 0.0;
			vw_xy[n]  = 0.0;
			wspeed_max[n] = 0.0;
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
					u_prof[k]  = u_prof[k]  + u[id];
					v_prof[k]  = v_prof[k]  + v[id];
					w_prof[k]  = w_prof[k]  + w[id];
					uu_prof[k] = uu_prof[k] + u[id] * u[id];
					vv_prof[k] = vv_prof[k] + v[id] * v[id];
					ww_prof[k] = ww_prof[k] + w[id] * w[id];
					uv_prof[k] = uv_prof[k] + u[id] * v[id];
					uw_prof[k] = uw_prof[k] + u[id] * w[id];
					vw_prof[k] = vw_prof[k] + v[id] * w[id];
// horizontal distributions
					if(k == kout[kk]) {
						idxy = i + nx*j + nxy*kk;
						u_xy[idxy]  = u_xy[idxy]  + u[id];
						v_xy[idxy]  = v_xy[idxy]  + v[id];
						w_xy[idxy]  = w_xy[idxy]  + w[id];
						uu_xy[idxy] = uu_xy[idxy] + u[id] * u[id];
						vv_xy[idxy] = vv_xy[idxy] + v[id] * v[id];
						ww_xy[idxy] = ww_xy[idxy] + w[id] * w[id];
						uv_xy[idxy] = uv_xy[idxy] + u[id] * v[id];
						uw_xy[idxy] = uw_xy[idxy] + u[id] * w[id];
						vw_xy[idxy] = vw_xy[idxy] + v[id] * w[id];
						wspeed = sqrt(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
						if(wspeed > wspeed_max[idxy]) wspeed_max[idxy] = wspeed;
						if(kk<12) kk = kk + 1;
					}
				}
			}
		}
	}

	// output if time arrived time_average //

	if ( time >= time_average )  {
// vertical profiles
		char str_pr[100];
		sprintf(str_pr,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/prof%08d_%04d.csv",t,rank);
		
		FILE *fp_pr;
		fp_pr = fopen(str_pr, "w");
		if (fp_pr == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_pr,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_pr,"%d,%f,",t,time);
		fprintf(fp_pr,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=halo+1; k<nz-halo; k++) {
			fprintf(fp_pr,"%6f,",	dx*k                                       );
			fprintf(fp_pr,"%6f,",	(u_prof[k] /i_count/nxy_air[k])*c_ref      );
			fprintf(fp_pr,"%6f,",	(v_prof[k] /i_count/nxy_air[k])*c_ref      );
			fprintf(fp_pr,"%6f,",	(w_prof[k] /i_count/nxy_air[k])*c_ref      );
			fprintf(fp_pr,"%6f,",	(uu_prof[k]/i_count/nxy_air[k])*c_ref*c_ref);
			fprintf(fp_pr,"%6f,",	(vv_prof[k]/i_count/nxy_air[k])*c_ref*c_ref);
			fprintf(fp_pr,"%6f,",	(ww_prof[k]/i_count/nxy_air[k])*c_ref*c_ref);
			fprintf(fp_pr,"%6f,",	(uv_prof[k]/i_count/nxy_air[k])*c_ref*c_ref);
			fprintf(fp_pr,"%6f,",	(uw_prof[k]/i_count/nxy_air[k])*c_ref*c_ref);
			fprintf(fp_pr,"%6f\n", (vw_prof[k] /i_count/nxy_air[k])*c_ref*c_ref);
		}
		fclose(fp_pr);

// horizontal distributions
		char str_xy[100];		// for file name

// horizontal distributions -- u
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_um%08d_%04d.csv",t,rank);
		FILE *fp_xy;
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (u_xy[idxy]/i_count)*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- v
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_vm%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (v_xy[idxy]/i_count)*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- w
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_wm%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (w_xy[idxy]/i_count)*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- uu
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_uu%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (uu_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- vv
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_vv%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (vv_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- ww
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_ww%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (ww_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- uv
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_uv%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (uv_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- uw
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_uw%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (uw_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- vw
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_vw%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   (vw_xy[idxy]/i_count)*c_ref*c_ref );
				}
				fprintf(fp_xy,"\n");
			}
		}
		fclose(fp_xy);

// horizontal distributions -- wspeed_max
		sprintf(str_xy,"/work1/t2g-jh130020/13IH0298/LBM_city/Output/xy_ws%08d_%04d.csv",t,rank);
		fp_xy = fopen(str_xy, "w");
		if (fp_xy == NULL) {
			printf("cannot open\n");
			exit(1);
		}
		fprintf(fp_xy,"%d,%d,%d,%d,",rank,rank_x,rank_y,rank_z);
		fprintf(fp_xy,"%d,%f,",t,time);
		fprintf(fp_xy,"%f,%f,%f\n",xg_min,yg_min,zg_min);
		for (k=0; k<12; k++) {
			fprintf(fp_xy,"%4d\n",kout[k]);
			for (j=halo; j<ny-halo; j++) {
				for (i=halo; i<nx-halo; i++) {
					idxy = i + nx*j + nxy*kout[k];
					fprintf(fp_xy,"%6f,",   wspeed_max[idxy]*c_ref );
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
