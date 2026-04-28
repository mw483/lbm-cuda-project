#include "Output_user_results.h"

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
#include <cstring>

// Constructor
Output_user_results::Output_user_results(
		const paramMPI &mpmpi,
		const paramDomain &mpdomain,
		const Variables *const mvariables,
		const BasisVariables *const mcbq,
		const FluidProperty *const mcfp,
		const Stress *const mstr
		) : 
	pmpi(mpmpi), pdomain(mpdomain), variables(mvariables), cbq(mcbq), cfp(mcfp), str(mstr)
{
	this->init();
	if(rank==0){
		this->fout_info();
	}
	this->malloc_vars();
	//this->malloc_ptrs();
}

Output_user_results::~Output_user_results(){}

//////////////////// public methods ////////////////////

void Output_user_results::user_file_output(int t){
	this->t = t;
	this->init_t(this->t);
	if(rank==0){
		printf("t = % 12.3f\n",time);
	}
	if(flag_ins_fout){
		// file output: instant values in z_out
		//fout_ins_xy_nz_out(this->r, "r", false);
		fout_ins_xy_nz_out(this->u, "u", true);
		fout_ins_xy_nz_out(this->v, "v", true);
		fout_ins_xy_nz_out(this->w, "w", true);
		//fout_ins_xy_nz_out(this->T, "t", false);
		//fout_ins_xy_nz_out(this->vis_sgs, "vsgs", false);

		// file output: instant values in y_out
		//fout_ins_xz_ny_out(this->r, "r", false);
		fout_ins_xz_ny_out(this->u, "u", true);
		fout_ins_xz_ny_out(this->v, "v", true);
		fout_ins_xz_ny_out(this->w, "w", true);
		//fout_ins_xz_ny_out(this->T, "t", false);
		//fout_ins_xz_ny_out(this->vis_sgs, "vsgs", false);

		// file output: instant values in x_out
		//fout_ins_yz_nx_out(this->r, "r", false);
		fout_ins_yz_nx_out(this->u, "u", true);
		fout_ins_yz_nx_out(this->v, "v", true);
		fout_ins_yz_nx_out(this->w, "w", true);
		//fout_ins_yz_nx_out(this->T, "t", false);
		//fout_ins_yz_nx_out(this->vis_sgs, "vsgs", false);

		// file output: instant values in volume
		//fout_ins_volume(this->r, "r", false);
		//fout_ins_volume(this->u, "u", true);
		//fout_ins_volume(this->v, "v", true);
		//fout_ins_volume(this->w, "w", true);
		//fout_ins_volume(this->T, "t", false);
		//fout_ins_volume(this->SS, "ss", false);
		//fout_ins_volume(this->WW, "ww", false);

		flag_ins_fout = false;
		if(rank==0){
			std::cout << "output instant values" <<  std::endl;
		}
	}
	if(flag_ave_skip){
		// add values
		//std::cout << "begin add vars..." << std::flush;
		this->add_vars();
		//std::cout << "\tdone" << std::endl;
		if(flag_ave_fout){
			// mean values and file output
			// prof
			this->mean_vars_prof();
			this->fout_ave_prof();

			// xz_ytave
			this->mean_vars_xz_ytave();
			this->fout_ave_xz_ytave(this->u_x, "u_x");
			this->fout_ave_xz_ytave(this->v_x, "v_x");
			this->fout_ave_xz_ytave(this->w_x, "w_x");
			//this->fout_ave_xz_ytave(this->t_x, "t_x");
			this->fout_ave_xz_ytave(this->uu_x, "uu_x");
			this->fout_ave_xz_ytave(this->vv_x, "vv_x");
			this->fout_ave_xz_ytave(this->ww_x, "ww_x");
			this->fout_ave_xz_ytave(this->uv_x, "uv_x");
			this->fout_ave_xz_ytave(this->uw_x, "uw_x");
			this->fout_ave_xz_ytave(this->vw_x, "vw_x");
			//this->fout_ave_xz_ytave(this->ut_x, "ut_x");
			//this->fout_ave_xz_ytave(this->vt_x, "vt_x");
			//this->fout_ave_xz_ytave(this->wt_x, "wt_x");
			//this->fout_ave_xz_ytave(this->tt_x, "tt_x");
			this->fout_ave_xz_ytave(this->uuu_x, "uuu_x");
			this->fout_ave_xz_ytave(this->vvv_x, "vvv_x");
			this->fout_ave_xz_ytave(this->www_x, "www_x");
			this->fout_ave_xz_ytave(this->uuw_x, "uuw_x");
			this->fout_ave_xz_ytave(this->vvw_x, "vvw_x");
			//this->fout_ave_xz_ytave(this->ttt_x, "ttt_x");

			// xy_tave with z_out
			this->mean_vars_xy_tave();
			this->fout_ave_xy_tave(this->u_xy, "u_xy");
			this->fout_ave_xy_tave(this->v_xy, "v_xy");
			this->fout_ave_xy_tave(this->w_xy, "w_xy");
			//this->fout_ave_xy_tave(this->t_xy, "t_xy");
			this->fout_ave_xy_tave(this->uu_xy, "uu_xy");
			this->fout_ave_xy_tave(this->vv_xy, "vv_xy");
			this->fout_ave_xy_tave(this->ww_xy, "ww_xy");
			this->fout_ave_xy_tave(this->uv_xy, "uv_xy");
			this->fout_ave_xy_tave(this->uw_xy, "uw_xy");
			this->fout_ave_xy_tave(this->vw_xy, "vw_xy");
			//this->fout_ave_xy_tave(this->ut_xy, "ut_xy");
			//this->fout_ave_xy_tave(this->vt_xy, "vt_xy");
			//this->fout_ave_xy_tave(this->wt_xy, "wt_xy");
			//this->fout_ave_xy_tave(this->tt_xy, "tt_xy");
			this->fout_ave_xy_tave(this->wspeed_max, "wspeed_max");
			this->fout_ave_xy_tave(this->wspeed_time, "wspeed_time");

			// xz_tave with y_out
			this->mean_vars_xz_tave();
			this->fout_ave_xz_tave(this->u_xz, "u_xz");
			this->fout_ave_xz_tave(this->v_xz, "v_xz");
			this->fout_ave_xz_tave(this->w_xz, "w_xz");
			//this->fout_ave_xy_tave(this->t_xz, "t_xz");
			this->fout_ave_xz_tave(this->uu_xz, "uu_xz");
			this->fout_ave_xz_tave(this->vv_xz, "vv_xz");
			this->fout_ave_xz_tave(this->ww_xz, "ww_xz");
			this->fout_ave_xz_tave(this->uv_xz, "uv_xz");
			this->fout_ave_xz_tave(this->uw_xz, "uw_xz");
			this->fout_ave_xz_tave(this->vw_xz, "vw_xz");
			//this->fout_ave_xy_tave(this->ut_xz, "ut_xz");
			//this->fout_ave_xy_tave(this->vt_xz, "vt_xz");
			//this->fout_ave_xy_tave(this->wt_xz, "wt_xz");
			//this->fout_ave_xy_tave(this->tt_xz, "tt_xz");

			// fill values with zero
			this->fill_zero_vars();

			// reset flag_ave_fout and i_count
			flag_ave_fout = false;
			i_count = 0;
			//fout_ins_volume(this->SS, "ss", false);
			//fout_ins_volume(this->WW, "ww", false);
			if(rank==0){
				std::cout << "output averaged values" <<  std::endl;
			}
		}
	}
}

//////////////////// private methods ////////////////////

void Output_user_results::init(){
	dirname = new char[100];
	sprintf(dirname, "./Output_bin/");

	rank = pmpi.rank();
	rank_x = pmpi.rank_x();
	rank_y = pmpi.rank_y();
	rank_z = pmpi.rank_z();

	ncpu = pmpi.ncpu();
	ncpu_x = pmpi.ncpu_x();
	ncpu_y = pmpi.ncpu_y();
	ncpu_z = pmpi.ncpu_z();

	nx = pdomain.nx();
	ny = pdomain.ny();
	nz = pdomain.nz();
	nn = pdomain.nn();

	n0 = pdomain.n0();

	nxg = pdomain.nxg();
	nyg = pdomain.nyg();
	nzg = pdomain.nzg();

	halo = pdomain.halo();

	xg_min = pdomain.xg_min();
	yg_min = pdomain.yg_min();
	zg_min = pdomain.zg_min();

	dx = pdomain.dx();

	c_ref = pdomain.c_ref();
	cfl_ref = pdomain.cfl_ref();

	dt_real = dx / c_ref;			// dt //  MOD2018

	// variables //
	r = cbq->r_n;
	u = cbq->u_n;
	v = cbq->v_n;
	w = cbq->w_n;
	l_obs = cfp->l_obs;
	T = variables->T_n;
	vis_sgs = str->vis_sgs;
	force_x = str->force_x;
	force_y = str->force_y;
	force_z = str->force_z;
	SS = str->SS;
	WW = str->WW;
	DIV = str->Div;

	// user define function //
	time = 0;
	this->t = 0;
	//time = t * dt_real;
	interval_ave_output = user_output::average_interval;	
	initial_skip_time	= user_output::skip_time;
	interval_ins_output = user_output::output_interval_ins;
	next_ins_output_time= user_output::time_output_ins_ini;
	next_ave_output_time= user_output::skip_time;
	nz_out			= user_output::nz_out;
	ny_out			= user_output::nj_out;
	nx_out			= user_output::ni_out;
	nv_out			= user_output::nv_out;
	nxy = nx * ny;
	nxz = nx * nz;
	nyz = ny * nz;
	nxyz= nx * ny * nz;
	nxy_out = nx * ny * nz_out;
	nxz_out = nx * nz * ny_out;
	nyz_out = ny * nz * nx_out;

	kout = new int[nz_out];
	jout = new int[ny_out];
	iout = new int[nx_out];
	vout_rank = new int[nv_out];
	for(int k=0;k<nz_out;k++){	kout[k] = user_output::kout[k];	}
	for(int j=0;j<ny_out;j++){	jout[j] = user_output::jout[j];	}
	for(int i=0;i<nx_out;i++){	iout[i] = user_output::iout[i];	}
	for(int i=0;i<nv_out;i++){	vout_rank[i] = user_output::vout_rank[i];	}

	nxy_air	=	new int[nz];	// number of air grid in z-direction (each xy-plane)
	ny_air	=	new int[nxz];	// number of air grid in y-direction (sideview)
	is_air	= 	new int[nxyz];	// all grid; fluid=1, solid=0;
	// set n_obs
	for(int id=0; id<nxyz; id++){
		// copy l_obs to is_air
		if(l_obs[id] < 0){
			is_air[id] = 1; // air
		}else{
			is_air[id] = 0; // solid
		}
		// exclude halo grid from is_air
		if(is_halo(id,true)){
			is_air[id] = 0;
		}
	}
	// set ny_air
	for(int id=0; id<nxz; id++){
		ny_air[id] = 0;
	}
	for(int id=0; id<nxyz; id++){
		int idxz = id%nx + nx*(id/(nx*ny));
		ny_air[idxz] += is_air[id]*is_halo(id,false); // count air cell in y_direction
	}
	for(int id=0; id<nxz; id++){
		if(ny_air[id]==0) ny_air[id]=1; // avoid zero-division
	}
	// set nxy_air
	for(int id=0; id<nz; id++){
		nxy_air[id] = 0;
	}
	for(int id=0; id<nxyz; id++){
		int idz = id/(nx*ny);
		nxy_air[idz] += is_air[id]*is_halo(id,false); // count air cell in z-direction
	}
	for(int id=0; id<nz; id++){
		if(nxy_air[id]==0) nxy_air[id]=1; // avoid zero-division
	}

	// output lv
	this->fout_ins_volume(this->l_obs, "l_obs", false);
}

void Output_user_results::init_t(int t){
	time = t * dt_real;
	// set flags
	if(time >= next_ins_output_time){
		flag_ins_fout = true;
		next_ins_output_time += interval_ins_output;
	}
	if(time >= initial_skip_time){
		flag_ave_skip = true;
	}
	if(time >= next_ave_output_time){
		flag_ave_fout = true;
		next_ave_output_time += interval_ave_output;
	}
}

void Output_user_results::malloc_ptrs(){
	// not implemented
	ptr_ins_val = new FLOAT*[5];
	ptr_ins_val[0] = r;
	ptr_ins_val[1] = u;
	ptr_ins_val[2] = v;
	ptr_ins_val[3] = w;
	ptr_ins_val[4] = T;

	ptr_ave_prof = new double*[22];
	ptr_ave_prof[0]  = r_prof;
	ptr_ave_prof[1]  = u_prof;
	ptr_ave_prof[2]  = v_prof;
	ptr_ave_prof[3]  = w_prof;
	ptr_ave_prof[4]  = t_prof;
	ptr_ave_prof[5]  = uu_prof;
	ptr_ave_prof[6]  = vv_prof;
	ptr_ave_prof[7]  = ww_prof;
	ptr_ave_prof[8]  = uv_prof;
	ptr_ave_prof[9]  = uw_prof;
	ptr_ave_prof[10] = vw_prof;
	ptr_ave_prof[11] = tt_prof;
	ptr_ave_prof[12] = ut_prof;
	ptr_ave_prof[13] = vt_prof;
	ptr_ave_prof[14] = wt_prof;
	ptr_ave_prof[15] = uuu_prof;
	ptr_ave_prof[16] = vvv_prof;
	ptr_ave_prof[17] = www_prof;
	ptr_ave_prof[18] = ttt_prof;
	ptr_ave_prof[19] = uuw_prof;
	ptr_ave_prof[20] = vvw_prof;
	ptr_ave_prof[21] = vsgs_prof;

	ptr_ave_xy = new double*[14];
	ptr_ave_xy[0]	= u_xy;
	ptr_ave_xy[1]	= v_xy;
	ptr_ave_xy[2]	= w_xy;
	ptr_ave_xy[3]	= t_xy;
	ptr_ave_xy[4]	= uu_xy;
	ptr_ave_xy[5]	= uv_xy;
	ptr_ave_xy[6]	= uw_xy;
	ptr_ave_xy[7]	= uv_xy;
	ptr_ave_xy[8]	= uw_xy;
	ptr_ave_xy[9]	= vw_xy;
	ptr_ave_xy[10]	= tt_xy;
	ptr_ave_xy[11]	= ut_xy;
	ptr_ave_xy[12]	= vt_xy;
	ptr_ave_xy[13]	= wt_xy;

	ptr_ave_x = new double*[20];
	ptr_ave_xy[0]	= u_x;
	ptr_ave_xy[1]	= v_x;
	ptr_ave_xy[2]	= w_x;
	ptr_ave_xy[3]	= t_x;
	ptr_ave_xy[4]	= uu_x;
	ptr_ave_xy[5]	= vv_x;
	ptr_ave_xy[6]	= ww_x;
	ptr_ave_xy[7]	= tt_x;
	ptr_ave_xy[8]	= uv_x;
	ptr_ave_xy[9]	= uw_x;
	ptr_ave_xy[10]	= vw_x;
	ptr_ave_xy[11]	= ut_x;
	ptr_ave_xy[12]	= vt_x;
	ptr_ave_xy[13]	= wt_x;
	ptr_ave_xy[14]	= uuu_x;
	ptr_ave_xy[15]	= vvv_x;
	ptr_ave_xy[16]	= www_x;
	ptr_ave_xy[17]	= ttt_x;
	ptr_ave_xy[18]	= uuw_x;
	ptr_ave_xy[19]	= vvw_x;
}

void Output_user_results::malloc_vars(){
	r_prof	=	new double[nz];
	u_prof	=	new double[nz];
	v_prof	=	new double[nz];
	w_prof	=	new double[nz];
	t_prof	=	new double[nz];
	uu_prof	=	new double[nz];
	vv_prof	=	new double[nz];
	ww_prof	=	new double[nz];
	tt_prof	=	new double[nz];
	uv_prof	=	new double[nz];
	uw_prof	=	new double[nz];
	vw_prof	=	new double[nz];
	ut_prof	=	new double[nz];
	vt_prof	=	new double[nz];
	wt_prof	=	new double[nz];
	uuu_prof=	new double[nz];
	vvv_prof=	new double[nz];
	www_prof=	new double[nz];
	ttt_prof=	new double[nz];
	uuw_prof=	new double[nz];
	vvw_prof=	new double[nz];
	vsgs_prof = new double[nz];

	u_xy	=	new double[nxy_out];
	v_xy	=	new double[nxy_out];
	w_xy	=	new double[nxy_out];
	t_xy	=	new double[nxy_out];
	uu_xy	=	new double[nxy_out];
	vv_xy	=	new double[nxy_out];
	ww_xy	=	new double[nxy_out];
	tt_xy	=	new double[nxy_out];
	uv_xy	=	new double[nxy_out];
	uw_xy	=	new double[nxy_out];
	vw_xy	=	new double[nxy_out];
	ut_xy	=	new double[nxy_out];
	vt_xy	=	new double[nxy_out];
	wt_xy	=	new double[nxy_out];
	wspeed_max	=	new double[nxy_out];
	wspeed_time	=	new	int[nxy_out];

	u_xz    =   new double[nxz_out];
	v_xz    =   new double[nxz_out];
	w_xz    =   new double[nxz_out];
	t_xz    =   new double[nxz_out];
	uu_xz   =   new double[nxz_out];
	vv_xz   =   new double[nxz_out];
	ww_xz   =   new double[nxz_out];
	uv_xz   =   new double[nxz_out];
	uw_xz   =   new double[nxz_out];
	vw_xz   =   new double[nxz_out];
	ut_xz   =   new double[nxz_out];
	vt_xz   =   new double[nxz_out];
	wt_xz   =   new double[nxz_out];
	tt_xz   =   new double[nxz_out];

	u_x		=	new double[nxz];
	v_x		=	new double[nxz];
	w_x		=	new double[nxz];
	t_x		=	new double[nxz];
	uu_x	=	new double[nxz];
	vv_x	=	new double[nxz];
	ww_x	=	new double[nxz];
	tt_x	=	new double[nxz];
	uv_x	=	new double[nxz];
	uw_x	=	new double[nxz];
	vw_x	=	new double[nxz];
	ut_x	=	new double[nxz];
	vt_x	=	new double[nxz];
	wt_x	=	new double[nxz];
	uuu_x	=	new double[nxz];
	vvv_x	=	new double[nxz];
	www_x	=	new double[nxz];
	ttt_x	=	new double[nxz];
	uuw_x	=	new double[nxz];
	vvw_x	=	new double[nxz];
}

void Output_user_results::fill_zero_vars(){
	for (int i=0;i<nz;i++){
		r_prof[i]	=0.0;
		u_prof[i]	=0.0;
		v_prof[i]	=0.0;
		w_prof[i]	=0.0;
		t_prof[i]	=0.0;
		uu_prof[i]	=0.0;
		vv_prof[i]	=0.0;
		ww_prof[i]	=0.0;
		tt_prof[i]	=0.0;
		uv_prof[i]	=0.0;
		uw_prof[i]	=0.0;
		vw_prof[i]	=0.0;
		ut_prof[i]	=0.0;
		vt_prof[i]	=0.0;
		wt_prof[i]	=0.0;
		uuu_prof[i]	=0.0;
		vvv_prof[i]	=0.0;
		www_prof[i]	=0.0;
		ttt_prof[i]	=0.0;
		uuw_prof[i]	=0.0;
		vvw_prof[i]	=0.0;
		vsgs_prof[i] = 0.0;
	}

	for (int i=0;i<nxz_out;i++){
		u_xz[i]	=	0.0;
		v_xz[i]	=	0.0;
		w_xz[i]	=	0.0;
		t_xz[i]	=	0.0;
		uu_xz[i]	=	0.0;
		vv_xz[i]	=	0.0;
		ww_xz[i]	=	0.0;
		tt_xz[i]	=	0.0;
		uv_xz[i]	=	0.0;
		uw_xz[i]	=	0.0;
		vw_xz[i]	=	0.0;
		ut_xz[i]	=	0.0;
		vt_xz[i]	=	0.0;
		wt_xz[i]	=	0.0;
	}

	for (int i=0;i<nxy_out;i++){
		u_xy[i]	=	0.0;
		v_xy[i]	=	0.0;
		w_xy[i]	=	0.0;
		t_xy[i]	=	0.0;
		uu_xy[i]	=	0.0;
		vv_xy[i]	=	0.0;
		ww_xy[i]	=	0.0;
		tt_xy[i]	=	0.0;
		uv_xy[i]	=	0.0;
		uw_xy[i]	=	0.0;
		vw_xy[i]	=	0.0;
		ut_xy[i]	=	0.0;
		vt_xy[i]	=	0.0;
		wt_xy[i]	=	0.0;
		wspeed_max[i]	= 0.0;
		wspeed_time[i]	= 0;
	}

	for (int i=0;i<nxz;i++){	
		u_x[i]		= 0.0;
		v_x[i]		= 0.0;
		w_x[i]		= 0.0;
		t_x[i]		= 0.0;
		uu_x[i]		= 0.0;
		vv_x[i]		= 0.0;
		ww_x[i]		= 0.0;
		tt_x[i]		= 0.0;
		uv_x[i]		= 0.0;
		uw_x[i]		= 0.0;
		vw_x[i]		= 0.0;
		ut_x[i]		= 0.0;
		vt_x[i]		= 0.0;
		wt_x[i]		= 0.0;
		uuu_x[i]	= 0.0;
		vvv_x[i]	= 0.0;
		www_x[i]	= 0.0;
		ttt_x[i]	= 0.0;
		uuw_x[i]	= 0.0;
		vvw_x[i]	= 0.0;
	}
}

inline void Output_user_results::add_vars(){
	i_count++;
	// prof
	for (int id=0; id<nxyz; id++){
		if(is_halo(id,true)){
			continue;
		}
		int idz = id/nxy;
		int idy = (id%nxy)/nx;
		//int idz_out = 0;
		int idxz = id%nx + nx*(id/nxy);
		int idxy_out = 0;
		int idxz_out = 0;
		double T0 = T[id] - 300.0;
		bool f_jout = false;
		for(int j=0; j<ny_out; j++){
			f_jout = false;
			if(idy == jout[j]+halo){
				f_jout = true;
				idxz_out = j*nxz + (id/nxy)*nx + id%nx;
				break;
			}
		}
		bool f_kout = false;
		for(int k=0; k<nz_out; k++){
			f_kout = false;
			if(idz == kout[k]+halo){
				f_kout = true;
				idxy_out = k*nxy + id%nxy;
				break;
			}
		}

		r_prof[idz] += (double)r[id] * is_air[id];
		u_prof[idz] += (double)u[id] * is_air[id];
		v_prof[idz] += (double)v[id] * is_air[id];
		w_prof[idz] += (double)w[id] * is_air[id];
		t_prof[idz] += T0 * is_air[id];
		uu_prof[idz] += (double)u[id] * u[id] * is_air[id];
		vv_prof[idz] += (double)v[id] * v[id] * is_air[id];
		ww_prof[idz] += (double)w[id] * w[id] * is_air[id];
		uv_prof[idz] += (double)u[id] * v[id] * is_air[id];
		uw_prof[idz] += (double)u[id] * w[id] * is_air[id];
		vw_prof[idz] += (double)v[id] * w[id] * is_air[id];
		ut_prof[idz] += (double)u[id] * T0 * is_air[id];
		vt_prof[idz] += (double)v[id] * T0 * is_air[id];
		wt_prof[idz] += (double)w[id] * T0 * is_air[id];
		tt_prof[idz] += T0 * T0 * is_air[id];
		uuu_prof[idz]+= (double)u[id] * u[id] * u[id] * is_air[id];
		vvv_prof[idz]+= (double)v[id] * v[id] * v[id] * is_air[id];
		www_prof[idz]+= (double)w[id] * w[id] * w[id] * is_air[id];
		uuw_prof[idz]+= (double)u[id] * u[id] * w[id] * is_air[id];
		vvw_prof[idz]+= (double)v[id] * v[id] * w[id] * is_air[id];
		ttt_prof[idz]+= T0 * T0 * T0 * is_air[id];
		vsgs_prof[idz]+=(double)vis_sgs[id];

		u_x[idxz] += (double)u[id] * is_air[id];
		v_x[idxz] += (double)v[id] * is_air[id];
		w_x[idxz] += (double)w[id] * is_air[id];
		t_x[idxz] += T0 * is_air[id];
		uu_x[idxz] += (double)u[id] * u[id] * is_air[id];
		vv_x[idxz] += (double)v[id] * v[id] * is_air[id];
		ww_x[idxz] += (double)w[id] * w[id] * is_air[id];
		uv_x[idxz] += (double)u[id] * v[id] * is_air[id];
		uw_x[idxz] += (double)u[id] * w[id] * is_air[id];
		vw_x[idxz] += (double)v[id] * w[id] * is_air[id];
		ut_x[idxz] += (double)u[id] * T0 * is_air[id];
		vt_x[idxz] += (double)v[id] * T0 * is_air[id];
		wt_x[idxz] += (double)w[id] * T0 * is_air[id];
		tt_x[idxz] += T0 * T0 * is_air[id];
		uuu_x[idxz] += (double)u[id] * u[id] * u[id] * is_air[id];
		vvv_x[idxz] += (double)v[id] * v[id] * v[id] * is_air[id];
		www_x[idxz] += (double)w[id] * w[id] * w[id] * is_air[id];
		uuw_x[idxz] += (double)u[id] * u[id] * w[id] * is_air[id];
		vvw_x[idxz] += (double)v[id] * v[id] * w[id] * is_air[id];
		ttt_x[idxz] += T0 * T0 * T0 * is_air[id];

		if(f_jout){
			u_xz[idxz_out] += (double)u[id] * is_air[id];
			v_xz[idxz_out] += (double)v[id] * is_air[id];
			w_xz[idxz_out] += (double)w[id] * is_air[id];
			t_xz[idxz_out] += T0 * is_air[id];
			uu_xz[idxz_out] += (double)u[id] * u[id] * is_air[id];
			vv_xz[idxz_out] += (double)v[id] * v[id] * is_air[id];
			ww_xz[idxz_out] += (double)w[id] * w[id] * is_air[id];
			uv_xz[idxz_out] += (double)u[id] * v[id] * is_air[id];
			uw_xz[idxz_out] += (double)u[id] * w[id] * is_air[id];
			vw_xz[idxz_out] += (double)v[id] * w[id] * is_air[id];
			ut_xz[idxz_out] += (double)u[id] * T0 * is_air[id];
			vt_xz[idxz_out] += (double)v[id] * T0 * is_air[id];
			wt_xz[idxz_out] += (double)w[id] * T0 * is_air[id];
			tt_xz[idxz_out] += T0 * T0 * is_air[id];
		}

		if(f_kout){
			u_xy[idxy_out] += (double)u[id] * is_air[id];
			v_xy[idxy_out] += (double)v[id] * is_air[id];
			w_xy[idxy_out] += (double)w[id] * is_air[id];
			t_xy[idxy_out] += T0 * is_air[id];
			uu_xy[idxy_out] += (double)u[id] * u[id] * is_air[id];
			vv_xy[idxy_out] += (double)v[id] * v[id] * is_air[id];
			ww_xy[idxy_out] += (double)w[id] * w[id] * is_air[id];
			uv_xy[idxy_out] += (double)u[id] * v[id] * is_air[id];
			uw_xy[idxy_out] += (double)u[id] * w[id] * is_air[id];
			vw_xy[idxy_out] += (double)v[id] * w[id] * is_air[id];
			ut_xy[idxy_out] += (double)u[id] * T0 * is_air[id];
			vt_xy[idxy_out] += (double)v[id] * T0 * is_air[id];
			wt_xy[idxy_out] += (double)w[id] * T0 * is_air[id];
			tt_xy[idxy_out] += T0 * T0 * is_air[id];
			wspeed = sqrt(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]) * is_air[id];
			if(wspeed > wspeed_max[idxy_out]){
				wspeed_max[idxy_out] = wspeed;
				wspeed_time[idxy_out] = t;
			}
		}
	}
}

inline void Output_user_results::mean_vars_prof(){
	for(int k=0; k<nz; k++){
		r_prof[k] = r_prof[k] / (i_count * nxy_air[k]);
		u_prof[k] = u_prof[k] / (i_count * nxy_air[k]) * c_ref;
		v_prof[k] = v_prof[k] / (i_count * nxy_air[k]) * c_ref;
		w_prof[k] = w_prof[k] / (i_count * nxy_air[k]) * c_ref;
		t_prof[k] = t_prof[k] / (i_count * nxy_air[k]);
		uu_prof[k] = uu_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref;
		vv_prof[k] = vv_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref;
		ww_prof[k] = ww_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref;
		uv_prof[k] = uv_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref;
		uw_prof[k] = uw_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref;
		vw_prof[k] = vw_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref;
		ut_prof[k] = ut_prof[k] / (i_count * nxy_air[k]) * c_ref;
		vt_prof[k] = vt_prof[k] / (i_count * nxy_air[k]) * c_ref;
		wt_prof[k] = wt_prof[k] / (i_count * nxy_air[k]) * c_ref;
		tt_prof[k] = tt_prof[k] / (i_count * nxy_air[k]);
		uuu_prof[k] = uuu_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref * c_ref;
		vvv_prof[k] = vvv_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref * c_ref;
		www_prof[k] = www_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref * c_ref;
		uuw_prof[k] = uuw_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref * c_ref;
		vvw_prof[k] = vvw_prof[k] / (i_count * nxy_air[k]) * c_ref * c_ref * c_ref;
		ttt_prof[k] = ttt_prof[k] / (i_count * nxy_air[k]);
		vsgs_prof[k] = vsgs_prof[k] / (i_count * nxy_air[k]);
	}
}

inline void Output_user_results::mean_vars_xz_ytave(){
	for(int id=0; id<nxz; id++){
		u_x[id] = u_x[id] / (i_count * ny_air[id]) * c_ref;
		v_x[id] = v_x[id] / (i_count * ny_air[id]) * c_ref;
		w_x[id] = w_x[id] / (i_count * ny_air[id]) * c_ref;
		t_x[id] = t_x[id] / (i_count * ny_air[id]);
		uu_x[id] = uu_x[id] / (i_count * ny_air[id]) * c_ref * c_ref;
		vv_x[id] = vv_x[id] / (i_count * ny_air[id]) * c_ref * c_ref;
		ww_x[id] = ww_x[id] / (i_count * ny_air[id]) * c_ref * c_ref;
		uv_x[id] = uv_x[id] / (i_count * ny_air[id]) * c_ref * c_ref;
		uw_x[id] = uw_x[id] / (i_count * ny_air[id]) * c_ref * c_ref;
		vw_x[id] = vw_x[id] / (i_count * ny_air[id]) * c_ref * c_ref;
		ut_x[id] = ut_x[id] / (i_count * ny_air[id]) * c_ref;
		vt_x[id] = vt_x[id] / (i_count * ny_air[id]) * c_ref;
		wt_x[id] = wt_x[id] / (i_count * ny_air[id]) * c_ref;
		tt_x[id] = tt_x[id] / (i_count * ny_air[id]);
		uuu_x[id] = uuu_x[id] / (i_count * ny_air[id]) * c_ref * c_ref * c_ref;
		vvv_x[id] = vvv_x[id] / (i_count * ny_air[id]) * c_ref * c_ref * c_ref;
		www_x[id] = www_x[id] / (i_count * ny_air[id]) * c_ref * c_ref * c_ref;
		uuw_x[id] = uuw_x[id] / (i_count * ny_air[id]) * c_ref * c_ref * c_ref;
		vvw_x[id] = vvw_x[id] / (i_count * ny_air[id]) * c_ref * c_ref * c_ref;
		ttt_x[id] = ttt_x[id] / (i_count * ny_air[id]);

		uu_x[id] = uu_x[id] - u_x[id] * u_x[id];
		vv_x[id] = vv_x[id] - v_x[id] * v_x[id];
		ww_x[id] = ww_x[id] - w_x[id] * w_x[id];
		uv_x[id] = uv_x[id] - u_x[id] * v_x[id];
		uw_x[id] = uw_x[id] - u_x[id] * w_x[id];
		vw_x[id] = vw_x[id] - v_x[id] * w_x[id];
		ut_x[id] = ut_x[id] - u_x[id] * t_x[id];
		vt_x[id] = vt_x[id] - v_x[id] * t_x[id];
		wt_x[id] = wt_x[id] - w_x[id] * t_x[id];
		tt_x[id] = tt_x[id] - t_x[id] * t_x[id];

		uuu_x[id] = uuu_x[id] - u_x[id]*u_x[id]*u_x[id] - 3*u_x[id]*uu_x[id];
		vvv_x[id] = vvv_x[id] - v_x[id]*v_x[id]*v_x[id] - 3*v_x[id]*vv_x[id];
		www_x[id] = www_x[id] - w_x[id]*w_x[id]*w_x[id] - 3*w_x[id]*ww_x[id];
		uuw_x[id] = uuw_x[id] - u_x[id]*u_x[id]*w_x[id] - 2*u_x[id]*uw_x[id] - w_x[id]*uu_x[id];
		vvw_x[id] = vvw_x[id] - v_x[id]*v_x[id]*w_x[id] - 2*v_x[id]*vw_x[id] - w_x[id]*vv_x[id];
		ttt_x[id] = ttt_x[id] - t_x[id]*t_x[id]*w_x[id] - 3*t_x[id]*tt_x[id];
	}
}

inline void Output_user_results::mean_vars_xz_tave(){
	for(int id=0; id<nxz_out; id++){
		u_xz[id] = u_xz[id] / i_count * c_ref;
		v_xz[id] = v_xz[id] / i_count * c_ref;
		w_xz[id] = w_xz[id] / i_count * c_ref;
		t_xz[id] = t_xz[id] / i_count;
		uu_xz[id] = uu_xz[id] / i_count * c_ref * c_ref;
		vv_xz[id] = vv_xz[id] / i_count * c_ref * c_ref;
		ww_xz[id] = ww_xz[id] / i_count * c_ref * c_ref;
		uv_xz[id] = uv_xz[id] / i_count * c_ref * c_ref;
		uw_xz[id] = uw_xz[id] / i_count * c_ref * c_ref;
		vw_xz[id] = vw_xz[id] / i_count * c_ref * c_ref;
		ut_xz[id] = ut_xz[id] / i_count * c_ref;
		vt_xz[id] = vt_xz[id] / i_count * c_ref;
		wt_xz[id] = wt_xz[id] / i_count * c_ref;
		tt_xz[id] = tt_xz[id] / i_count;
	}
}

inline void Output_user_results::mean_vars_xy_tave(){
	for(int id=0; id<nxy_out; id++){
		u_xy[id] = u_xy[id] / i_count * c_ref;
		v_xy[id] = v_xy[id] / i_count * c_ref;
		w_xy[id] = w_xy[id] / i_count * c_ref;
		t_xy[id] = t_xy[id] / i_count;
		uu_xy[id] = uu_xy[id] / i_count * c_ref * c_ref;
		vv_xy[id] = vv_xy[id] / i_count * c_ref * c_ref;
		ww_xy[id] = ww_xy[id] / i_count * c_ref * c_ref;
		uv_xy[id] = uv_xy[id] / i_count * c_ref * c_ref;
		uw_xy[id] = uw_xy[id] / i_count * c_ref * c_ref;
		vw_xy[id] = vw_xy[id] / i_count * c_ref * c_ref;
		ut_xy[id] = ut_xy[id] / i_count * c_ref;
		vt_xy[id] = vt_xy[id] / i_count * c_ref;
		wt_xy[id] = wt_xy[id] / i_count * c_ref;
		tt_xy[id] = tt_xy[id] / i_count;
		wspeed_max[id] = wspeed_max[id] * c_ref;
	}
}

inline int Output_user_results::is_halo(int id, bool ret_true){
	bool zhalo = ((id/(nx*ny))+halo)%nz 	< 2*halo;
	bool yhalo = ((id%(nx*ny))/nx+halo)%ny 	< 2*halo;
	bool xhalo = ((id%(nx*ny))+halo)%nx 	< 2*halo;
	if(ret_true){
		if(xhalo || yhalo || zhalo){
			return 1;
		}else{
			return 0;
		}
	}else{
		if(xhalo || yhalo || zhalo){
			return 0;
		}else{
			return 1;
		}
	}
}

inline void Output_user_results::fout_info(){
	std::string fnameStr = std::string(dirname) + "calc_info.txt";
	const char* fnameC = fnameStr.c_str();
	std::ofstream fp(fnameC);
	if(!fp.is_open()){
		fprintf(stderr, "cannot open:%s\n",fnameC);
		exit(1);
	}
	fp	<<	"ncpu    , "	<<	ncpu	<<	std::endl;
	fp	<<	"ncpu_x  , "	<<	ncpu_x	<<	std::endl;
	fp	<<	"ncpu_y  , "	<<	ncpu_y	<<	std::endl;
	fp	<<	"ncpu_z  , "	<<	ncpu_z	<<	std::endl;
	fp	<<	"nx      , "	<<	nx		<<	std::endl;
	fp	<<	"ny      , "	<<	ny		<<	std::endl;
	fp	<<	"nz      , "	<<	nz		<<	std::endl;
	fp	<<	"nn      , "	<<	nn		<<	std::endl;
	fp	<<	"n0      , "	<<	n0		<<	std::endl;
	fp	<<	"nxg     , "	<<	nxg		<<	std::endl;
	fp	<<	"nyg     , "	<<	nyg		<<	std::endl;
	fp	<<	"nzg     , "	<<	nzg		<<	std::endl;
	fp	<<	"halo    , "	<<	halo	<<	std::endl;
	fp	<<	"xg_min  , "	<<	xg_min	<<	std::endl;
	fp	<<	"yg_min  , "	<<	yg_min	<<	std::endl;
	fp	<<	"zg_min  , "	<<	zg_min	<<	std::endl;
	fp	<<	"dx      , "	<<	dx		<<	std::endl;
	fp	<< 	"dt		 , "	<<  dt_real	<< 	std::endl;
	fp	<<	"c_ref   , "	<<	c_ref	<<	std::endl;
	fp	<<	"cfl_ref , "	<<	cfl_ref	<<	std::endl;
	fp	<<	"dt_real , "	<<	dt_real	<<	std::endl;
	fp  <<  "t_ave   , "    <<  interval_ave_output << std::endl;
	fp  <<  "t_ins   , "    <<  interval_ins_output << std::endl;
	fp	<< 	"nz_out	 , "	<<	std::flush;
	for(int i=0; i<nz_out; i++){
		fp << kout[i];
		if(i<nz_out-1){ fp << "," << std::flush;}
		else{ fp << std::endl;}
	}
	fp	<< 	"ny_out	 , "	<<	std::flush;
	for(int i=0; i<ny_out; i++){
		fp << jout[i];
		if(i<ny_out-1){ fp << "," << std::flush;}
		else{ fp << std::endl;}
	}
	fp	<< 	"nx_out	 , "	<<	std::flush;
	for(int i=0; i<nx_out; i++){
		fp << iout[i];
		if(i<nx_out-1){ fp << "," << std::flush;}
		else{ fp << std::endl;}
	}
	fp.close();
}

inline void Output_user_results::fout_ins_xy_nz_out(
		const FLOAT* pdata,
		const char * ins_val_name,
		bool vel_flag
		){
	char * fname = new char[255];
	sprintf(fname,"%snzout_ins_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	FLOAT * data = new FLOAT[nxy_out];
	if(vel_flag){
		for(int k=0; k<nz_out; k++){
			for(int idl=0; idl<nxy; idl++){
				int idg = nxy*(kout[k]+halo) + idl;
				data[k*nxy + idl] = pdata[idg] * c_ref;
			}
		}
	}else{
		for(int k=0; k<nz_out; k++){
			for(int idl=0; idl<nxy; idl++){
				int idg = nxy*(kout[k]+halo) + idl;
				data[k*nxy + idl] = pdata[idg];
			}
		}
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(FLOAT)*nxy_out);
	// close file
	fp.close();
}

inline void Output_user_results::fout_ins_xz_ny_out(
		const FLOAT * pdata,
		const char * ins_val_name,
		bool vel_flag
		){
	char * fname = new char[255];
	sprintf(fname,"%snyout_ins_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	FLOAT * data = new FLOAT[nxz_out];
	if(vel_flag){
		for(int j=0; j<ny_out; j++){
			for(int idl=0; idl<nxz; idl++){
				int idg = nxy*int(idl/nx) + nx*(jout[j]+halo) + (idl%nx);
				data[j*nxz + idl] = pdata[idg] * c_ref;
			}
		}
	}else{
		for(int j=0; j<ny_out; j++){
			for(int idl=0; idl<nxz; idl++){
				int idg = nxy*int(idl/nx) + nx*(jout[j]+halo) + (idl%nx);
				data[j*nxz + idl] = pdata[idg];
			}
		}
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(FLOAT)*nxz_out);
	// close file
	fp.close();
}

inline void Output_user_results::fout_ins_yz_nx_out(
		const FLOAT * pdata,
		const char * ins_val_name,
		bool vel_flag
		){
	char * fname = new char[255];
	sprintf(fname,"%snxout_ins_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	FLOAT * data = new FLOAT[nyz_out];
	if(vel_flag){
		for(int i=0; i<nx_out; i++){
			for(int idl=0; idl<nyz; idl++){
				int idg = nxy*int(idl/ny) + nx*(idl%ny) + iout[i]+halo;
				data[i*nyz + idl] = pdata[idg] * c_ref;
			}
		}
	}else{
		for(int i=0; i<nx_out; i++){
			for(int idl=0; idl<nyz; idl++){
				int idg = nxy*int(idl/ny) + nx*(idl%ny) + iout[i]+halo;
				data[i*nyz + idl] = pdata[idg];
			}
		}
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(FLOAT)*nyz_out);
	// close file
	fp.close();
}

inline void Output_user_results::fout_ins_volume(
		const FLOAT * pdata,
		const char * ins_val_name,
		bool vel_flag
		){
	// open file
	char * fname = new char[255];
	sprintf(fname,"%svo_ins_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	FLOAT * data = new FLOAT[nxyz];
	if(vel_flag){
		for(int i=0; i<nxyz; i++){
			data[i] = pdata[i] * c_ref;
		}
	}else{
		for(int i=0; i<nxyz; i++){
			data[i] = pdata[i];
		}
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(FLOAT)*nxyz);
	// close file
	fp.close();
}

inline void Output_user_results::fout_ave_prof(
		){
	char * fname = new char[255];
	sprintf(fname, "%sprof_%08d_%04d.bin", dirname, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	fp.write(reinterpret_cast<char*>(r_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(u_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(v_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(w_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(t_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(uu_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(vv_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(ww_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(uv_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(uw_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(vw_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(ut_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(vt_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(wt_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(tt_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(uuu_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(vvv_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(www_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(uuw_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(vvw_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(ttt_prof),sizeof(double)*nz);
	fp.write(reinterpret_cast<char*>(vsgs_prof),sizeof(double)*nz);
	fp.close();
}

inline void Output_user_results::fout_ave_xz_ytave(
		const double* pdata,
		const char * ins_val_name
		){
	char * fname = new char[255];
	sprintf(fname,"%sxz_ave_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	double * data = new double[nxz];
	for(int i=0; i<nxz; i++){
		data[i] = pdata[i];
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(double)*nxz);
	// close file
	fp.close();
}

inline void Output_user_results::fout_ave_xz_tave(
		const double* pdata,
		const char* ins_val_name
		){
	char * fname = new char[255];
	sprintf(fname,"%sxz_tave_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	double * data = new double[nxz_out];
	for(int i=0; i<nxz_out; i++){
		data[i] = pdata[i];
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(double)*nxz_out);
	// close file
	fp.close();
}

inline void Output_user_results::fout_ave_xy_tave(
		const double* pdata,
		const char * ins_val_name
		){
	char * fname = new char[255];
	sprintf(fname,"%sxy_ave_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	//data
	double * data = new double[nxy_out];
	for(int i=0; i<nxy_out; i++){
		data[i] = pdata[i];
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(double)*nxy_out);
	// close file
	fp.close();
}

//inline void Output_user_results::fout_ave_xy_tave(
//		const FLOAT* pdata,
//		const char * ins_val_name
//		){
//	char * fname = new char[255];
//	sprintf(fname,"%sxy_ave_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
//	//printf("%s\t",fname);
//	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
//	// write data
//	// data
//	FLOAT * data = new FLOAT[nxy_out];
//	for(int i=0; i<nxy_out; i++){
//		data[i] = pdata[i];
//	}
//	fp.write(reinterpret_cast<char*>(data), sizeof(FLOAT)*nxy_out);
//	// close file
//	fp.close();
//}

inline void Output_user_results::fout_ave_xy_tave(
		const int* pdata,
		const char * ins_val_name
		){
	char * fname = new char[255];
	sprintf(fname,"%sxy_ave_%s_%08d_%04d.bin", dirname, ins_val_name, this->t, rank);
	//printf("%s\t",fname);
	std::ofstream fp(fname, std::ios::binary|std::ios::trunc);
	// write data
	// data
	int * data = new int[nxy_out];
	for(int i=0; i<nxy_out; i++){
		data[i] = pdata[i];
	}
	fp.write(reinterpret_cast<char*>(data), sizeof(int)*nxy_out);
	// close file
	fp.close();
}

