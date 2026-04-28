#ifndef OUTPUT_USER_RESULTS_H_
#define OUTPUT_USER_RESULTS_H_

#include <cstdio>
#include <cmath>
#include <string>

#include <fstream>
#include "paramDomain.h"
#include "paramMPI.h"
#include "stVariables.h"
#include "stBasisVariables.h"
#include "stFluidProperty.h"
#include "stStress.h"

/*
void
output_user_def (
		int						t,
		char					*program_name,
		int						argc,
		char					*argv[], 
		const paramMPI			&pmpi, 
		const paramDomain		&pdomain,
		const Variables     	*const variables,
		const BasisVariables	*const cbq,
		const FluidProperty		*const cfp
		);
*/


class Output_user_results {
	private:
		paramMPI		const &pmpi;
		paramDomain		const &pdomain;
		Variables		const * const variables;
		BasisVariables	const * const cbq;
		FluidProperty	const * const cfp;
		Stress 			const * const str;


	public:

		// Constructor
		Output_user_results(
				const paramMPI &mpmpi,
				const paramDomain &mpdomain,
				const Variables *const mvariables,
				const BasisVariables *const mcbq,
				const FluidProperty *const mcfp,
				const Stress *const mstr
				);

		// Deconstructor
		~Output_user_results();

		void user_file_output(int t);

	private:
		// initialize function
		void init();		// first initialization
		void init_t(int t);	// initialization for each time step
		void malloc_vars();
		void malloc_ptrs();
		void fill_zero_vars();
		void add_vars();
		void mean_vars_prof();
		void mean_vars_xz_ytave();
		void mean_vars_xy_tave();
		void mean_vars_xz_tave();

		int is_halo(int id, bool ret_true);

		// file output : instantaneous value
		void fout_info();
		void fout_ins_xy_nz_out(const FLOAT* pdata, const char *ins_val_name, bool vel_flag);
		void fout_ins_xz_ny_out(const FLOAT* pdata, const char *ins_val_name, bool vel_flag); // not implemented
		void fout_ins_yz_nx_out(const FLOAT* pdata, const char *ins_val_name, bool vel_flag); // not implemented
		void fout_ins_volume(const FLOAT* pdata, const char *ins_val_name, bool vel_flag);

		// file output : mean values
		void fout_ave_prof();
		void fout_ave_xz_ytave(const double* pdata, const char *ins_val_name);
		void fout_ave_xz_tave(const double* pdata, const char *ins_val_name);
		void fout_ave_xy_tave(const double* pdata, const char *ins_val_name);
		//void fout_ave_xy_tave(const FLOAT* pdata, const char *ins_val_name);
		void fout_ave_xy_tave(const int* pdata, const char *ins_val_name);

	private:

		char *dirname;
		
		// mpi
		int rank;
		int rank_x;
		int rank_y;
		int rank_z;

		int ncpu;
		int ncpu_x;
		int ncpu_y;
		int ncpu_z;

		// domain
		int nx;
		int ny;
		int nz;
		int nn;
		int n0;
		int nxg;
		int nyg;
		int nzg;
		int halo;
		FLOAT xg_min;
		FLOAT yg_min;
		FLOAT zg_min;
		FLOAT dx;
		FLOAT c_ref;
		FLOAT cfl_ref;
		FLOAT dt_real;

		// variables

		FLOAT *r;
		FLOAT *u;
		FLOAT *v;
		FLOAT *w;
		FLOAT *l_obs;
		FLOAT *T;
		FLOAT *vis_sgs;
		FLOAT *force_x;
		FLOAT *force_y;
		FLOAT *force_z;
		FLOAT *SS;
		FLOAT *WW;
		FLOAT *DIV;

		// user define function
		int t;
		FLOAT time;
		//FLOAT skip_time;
		FLOAT initial_skip_time;
		//FLOAT output_interval_ins;
		FLOAT interval_ins_output;
		//FLOAT average_interval;
		FLOAT interval_ave_output;
		//FLOAT time_output_ins_ini;
		FLOAT next_ins_output_time;
		FLOAT next_ave_output_time;
		FLOAT wspeed;
		FLOAT T0; // T0 = T - 300.0

		int *kout;
		int *jout;
		int *iout;
		int *vout_rank;

		int nz_out;
		int nj_out;
		int ny_out;
		int ni_out;
		int nx_out;
		int nv_out;
		int nxy;
		int nxz;
		int nyz;
		int nxyz;
		int nxy_out;
		int nxz_out;
		int nyz_out;

		bool flag;
		bool flag_ins;

		bool flag_ins_fout;
		bool flag_ave_skip;
		bool flag_ave_fout;

		int i_count;
		FLOAT time_average;
		FLOAT time_output_ins;

		int *nxy_air;
		int *ny_air;
		int *is_air;

		// for average profile (size:nz)
		double *r_prof; 
		double *u_prof; 
		double *v_prof; 
		double *w_prof; 
		double *t_prof; 
		double *uu_prof; 
		double *vv_prof; 
		double *ww_prof; 
		double *uv_prof; 
		double *uw_prof; 
		double *vw_prof; 
		double *ut_prof; 
		double *vt_prof; 
		double *wt_prof; 
		double *tt_prof; 
		double *uuu_prof; 
		double *vvv_prof; 
		double *www_prof; 
		double *uuw_prof; 
		double *vvw_prof; 
		double *ttt_prof; 
		double *vsgs_prof;

		// for average in y and time (size:nxz)
		double	*u_x;
		double	*v_x;
		double  *w_x;
		double  *t_x;
		double  *uu_x;
		double  *vv_x;
		double  *ww_x;
		double  *uv_x;
		double  *uw_x;
		double  *vw_x;
		double  *ut_x;
		double  *vt_x;
		double  *wt_x;
		double  *tt_x;
		double  *uuu_x;
		double  *vvv_x;
		double  *www_x;
		double  *ttt_x;
		double  *uuw_x;
		double  *vvw_x;
		
		// for average in t (size:nxy_out)
		double	*u_xy;
		double	*v_xy;
		double	*w_xy;
		double  *t_xy;
		double	*uu_xy;
		double	*vv_xy;
		double	*ww_xy;
		double	*uv_xy;
		double	*uw_xy;
		double	*vw_xy;
		double  *ut_xy;
		double  *vt_xy;
		double  *wt_xy;
		double  *tt_xy;
		double	*wspeed_max;
		int		*wspeed_time;

		// for average in t (size:nxy_out)
		double  *u_xz;
		double  *v_xz;
		double  *w_xz;
		double  *t_xz;
		double  *uu_xz;
		double  *vv_xz;
		double  *ww_xz;
		double  *uv_xz;
		double  *uw_xz;
		double  *vw_xz;
		double  *ut_xz;
		double  *vt_xz;
		double  *wt_xz;
		double  *tt_xz;

		// pointer array for output values
		FLOAT * * ptr_ins_val;
		double * * ptr_ave_prof;
		double * * ptr_ave_xy;
		double * * ptr_ave_x;
};

#endif
