#include "calculation.h"

#include "defineCUDA.h"
#include "defineReferenceVel.h"
#include "paramMPI.h"
#include "functionLib.h"
#include "macroCUDA.h"
#include "defineLBM.h" // LBM scheme (D3Q19, D3Q27) (Bounce-back, 2nd-order)
#include "lbm_gpu.h"
#include "sgs_model_gpu.h"
#include "boundary.h"
#include "boundary_LBM.h"

#include "Define_user.h"	// MOD 2018


// public
void	Calculation::
set (const paramDomain	&pdomain)
{
	// mpi //
	rank_   = pmpi_.rank();
	rank_x_ = pmpi_.rank_x();
	rank_y_ = pmpi_.rank_y();
	rank_z_ = pmpi_.rank_z();

	ncpu_   = pmpi_.ncpu();
	ncpu_x_ = pmpi_.ncpu_x();
	ncpu_y_ = pmpi_.ncpu_y();
	ncpu_z_ = pmpi_.ncpu_z();


	// domain
	nx_ = pdomain.nx();
	ny_ = pdomain.ny();
	nz_ = pdomain.nz();
	nn_ = pdomain.nn();

	nxg_ = pdomain.nxg();
	nyg_ = pdomain.nyg();
	nzg_ = pdomain.nzg();

	halo_ = pdomain.halo();

    dx_ = pdomain.dx();
    dt_ = pdomain.dt();



	// lbm //
	c_ref_   = pdomain.c_ref();
	cfl_ref_ = pdomain.cfl_ref();

	// cuda
	functionLib::set_dim3(&grid_,
			(nx_-2*halo_)/BLOCKDIM_X,
			(ny_-2*halo_)/BLOCKDIM_Y,
			(nz_-2*halo_)/BLOCKDIM_Z);


	functionLib::set_dim3(&block_2d_,
			BLOCKDIM_X,
		   	BLOCKDIM_Y,
		   	1);

}


// initialize
void	Calculation::
initialize_calculation (
	Variables		*cq,
	Variables		*cqn,
	BasisVariables	*cbq,
	FluidProperty	*cfp
	)
{
	// boundary //
	mpi_boundary_Variables   (cq, cbq, cfp);


	// obstacle filter //
	velocity_obstacle_filter (cbq, cfp);

	// velocity_to_lbm_function //
	velocity_to_lbm_function (cq, cbq);


	// boundary //
	mpi_boundary_Variables      (cq, cbq, cfp);


	// copy //
	const int	nsize = nn_*NUM_DIRECTION_VEL;
	CUDA_SAFE_CALL( cudaMemcpy(cqn->f_n,   cq->f_n,   sizeof(FLOAT)*(nsize), cudaMemcpyDeviceToDevice) );
	MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier //
}


// Calculation
void	Calculation::
gpu_calculation (
	Variables		*cq,
   	Variables		*cqn,
	BasisVariables	*cbq,
	Stress			*str,
   	FluidProperty	*cfp
	)
{
	// Runge-Kutta //
	calculation_LBM (
	   	cq,
		cqn,
	   	cbq,
	   	str,
	   	cfp);

	// immersed boundary //


	// swap //
	functionLib::swap (&cqn->f_n, &cq->f_n);
	functionLib::swap (&cqn->T_n, &cq->T_n);

	MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier *****
}

// (YOKOUCHI 2020)
// Mean velocity
void	Calculation::
mean_velocity (
	const	BasisVariables	*cbq,
		Stress		*str,
		int		t
	)
{
	// input //
	const FLOAT	*u	= cbq->u_n;
	const FLOAT	*v 	= cbq->v_n;
	const FLOAT	*w 	= cbq->w_n;

	      FLOAT	*um	= str->u_m;
	      FLOAT	*vm	= str->v_m;
	      FLOAT	*wm	= str->w_m;
	
	mean_velocity_GPU <<< grid_, block_2d_ >>> (
		u,  v,  w,
		um, vm, wm,
		nx_, ny_, nz_,
		halo_,
		t);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}

// (YOKOUCHI 2020)
// TKE sgs
void	Calculation::
sgs_tke_LBM (
	const	BasisVariables	*cbq,
		Stress		*str,
		int		t
	)
{
	// input //
	const FLOAT	*u	 	= cbq->u_n;
	const FLOAT	*v	 	= cbq->v_n;
	const FLOAT	*w	 	= cbq->w_n;

	      FLOAT	*tke_sgs 	= str->TKE_sgs;
	      FLOAT	*tke_sgs_old 	= str->TKE_sgs_old;

	tke_LBM_GPU <<< grid_, block_2d_ >>> (
		u, v, w,
		tke_sgs,
		tke_sgs_old,
		nx_, ny_, nz_,
		halo_,
		t);
	
	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize();

}

void	Calculation::
calculation_LBM (
   	const Variables			*cq,
   	      Variables			*cqn,
	      BasisVariables	*cbq,
	      Stress			*str,
   	      FluidProperty		*cfp
	)
{
	// sgs model //
	sgs_model                (cbq, str, cfp);


	// LBM //
	stream_collision_thermal_convection (cq, cqn, cbq, str, cfp);


//	// force //
//	set_force                (str, cfp);
//	force_acceleration       (cq, cqn, str);


	// lbm_function_to_velocity //
	lbm_function_to_velocity (cqn, cbq);		//MOD2018 comment out // 2019 uncommented


	// obstacle //
	obstacle_variables (cqn, cbq, cfp);		// MOD2018 // 2019 uncommented


	// rho, uvw to lbm function //


	// boundary
	mpi_boundary_Variables      (cqn, cbq, cfp);
}


void	Calculation::
sgs_model (
	const BasisVariables	*const cbq,
	      Stress			*str,
   	const FluidProperty		*const cfp
	)
{
	// input //
	const FLOAT	*u = cbq->u_n;
	const FLOAT	*v = cbq->v_n;
	const FLOAT	*w = cbq->w_n;

//	const char	*status = cfp->status;
	const FLOAT	*l_obs  = cfp->l_obs;

	// output //
	FLOAT	*Fcs = str->Fcs_sgs;
	FLOAT	*Div = str->Div;
	FLOAT	*SS  = str->SS;
	FLOAT	*WW  = str->WW;

	cudaFuncSetCacheConfig( cuda_Fcs_CSM,	cudaFuncCachePreferL1 );


	// sgs_model_gpu.h //
	cuda_Fcs_CSM  <<< grid_, block_2d_ >>>  (
		u, v, w,
		l_obs,
		Fcs,
		Div,
		SS, WW,
		nx_, ny_, nz_,
		halo_
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize *****
}


void	Calculation::
stream_collision (
   	const Variables			*cq,
   	      Variables			*cqn,
	      BasisVariables	*cbq,
	      Stress			*str,
   	      FluidProperty		*cfp
	)
{
	// input //
	const FLOAT	*f       = cq->f_n;
	const FLOAT	*l_obs   = cfp->l_obs;
	const FLOAT	*vis_lbm = cfp->vis_lbm;
	const FLOAT	*Fcs     = str->Fcs_sgs;


	// output //
	FLOAT	*fn      = cqn->f_n;
	FLOAT	*vis_sgs = str->vis_sgs;


	cudaFuncSetCacheConfig( cuda_stream_collision_D3Q19,	cudaFuncCachePreferL1 );
	cudaFuncSetCacheConfig( cuda_stream_collision_D3Q27,	cudaFuncCachePreferL1 );

	// Stream Collision //
#ifdef D3Q19_MODEL_
	cuda_stream_collision_D3Q19   <<< grid_, block_2d_ >>>  (
		f,
		fn,
		l_obs,
		vis_lbm,
		vis_sgs,
		Fcs,
		nx_,     ny_,     nz_,
		halo_
		);
#endif
#ifdef D3Q27_MODEL_
	cuda_stream_collision_D3Q27   <<< grid_, block_2d_ >>>  (
		f,
		fn,
		l_obs,
		vis_lbm,
		vis_sgs,
		Fcs,
		nx_,     ny_,     nz_,
		halo_
		);
#endif

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	Calculation::
stream_collision_moving_boundary (
   	const Variables			*cq,
   	      Variables			*cqn,
	      BasisVariables	*cbq,
	      Stress			*str,
   	      FluidProperty		*cfp
	)
{
	// input //
	const FLOAT	*f       = cq->f_n;
	const FLOAT	*l_obs   = cfp->l_obs;
	const FLOAT	*u_obs   = cfp->u_obs;
	const FLOAT	*v_obs   = cfp->v_obs;
	const FLOAT	*w_obs   = cfp->w_obs;

	const FLOAT	*vis_lbm = cfp->vis_lbm;
	const FLOAT	*Fcs     = str->Fcs_sgs;


	// output //
	FLOAT	*fn      = cqn->f_n;
	FLOAT	*vis_sgs = str->vis_sgs;


	cudaFuncSetCacheConfig( cuda_stream_collision_D3Q19_moving_boundary,	cudaFuncCachePreferL1 );


	// Stream Collision //
	cuda_stream_collision_D3Q19_moving_boundary   <<< grid_, block_2d_ >>>  (
		f,
		fn,
		l_obs,
		u_obs,
		v_obs,
		w_obs,
		vis_lbm,
		vis_sgs,
		Fcs,
		nx_,     ny_,     nz_,
		halo_
		);


	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	Calculation::
stream_collision_thermal_convection (
   	const Variables			*cq,
   	      Variables			*cqn,
	      BasisVariables	*cbq,
	      Stress			*str,
   	      FluidProperty		*cfp
	)
{
	// input //
	const FLOAT	*f       = cq->f_n;
	const FLOAT	*T       = cq->T_n;
	const FLOAT	*l_obs   = cfp->l_obs;
	const FLOAT	*vis_lbm = cfp->vis_lbm;
	const FLOAT	*Fcs     = str->Fcs_sgs;

	FLOAT	*u = cbq->u_n;				//MOD2018
	FLOAT	*v = cbq->v_n;				//MOD2018
	FLOAT	*w = cbq->w_n;				//MOD2018
	FLOAT	*rho = cbq->r_n;			//MOD2018

    const FLOAT* bcTw = cfp->hflux_w; //  1,  0,  0 //
    const FLOAT* bcTe = cfp->hflux_e; // -1,  0,  0 //
    const FLOAT* bcTs = cfp->hflux_s; //  0,  1,  0 //
    const FLOAT* bcTn = cfp->hflux_n; //  0, -1,  0 //
    const FLOAT* bcTr = cfp->hflux_r; //  0,  0,  1 //


	// output //
	FLOAT	*fn      = cqn->f_n;
	FLOAT	*Tn      = cqn->T_n;
	FLOAT	*vis_sgs = str->vis_sgs;
	//(YOKOUCHI 2020)
//	FLOAT	*vis_sgs_old = str->vis_sgs_old;
	FLOAT	*tke_sgs     = str->TKE_sgs;
	FLOAT	*tke_sgs_old = str->TKE_sgs_old; 

	cudaFuncSetCacheConfig( cuda_stream_collision_D3Q19,	cudaFuncCachePreferL1 );
	cudaFuncSetCacheConfig( cuda_stream_collision_D3Q27,	cudaFuncCachePreferL1 );

	// Stream Collision //
#ifdef D3Q19_MODEL_
    std::cout << "causion : thermal convection of D3Q19 model is not implemented" << std::endl;
	cuda_stream_collision_D3Q19   <<< grid_, block_2d_ >>>  (
		f,
		fn,
		l_obs,
		vis_lbm,
		vis_sgs,
		Fcs,
		nx_,     ny_,     nz_,
		halo_
		);
#endif
#ifdef D3Q27_MODEL_

// for Boussinesq approximation   MOD2018
	const int nx = nx_;
	const int ny = ny_;
	const int nz = nz_;		// MOD2018
//	const int nxy = nx * ny;
	const int nxyz = nx * ny * nz;

/* Memory copy from Device to Host */
//	FLOAT *T_h;
//	T_h = (FLOAT *) malloc( sizeof(FLOAT)*nxyz);
//	cudaMemcpy(T_h, T, nxyz*sizeof(FLOAT), cudaMemcpyDeviceToHost); 
//std::cout<<T_h[0]<<" test"<<std::endl;

/* Create vertical T0 profile */
	FLOAT *T_refh;
	T_refh = (FLOAT *)malloc (nz*sizeof(FLOAT));
//	Calculation::Vertical_Profile_T (T_h, T_refh);
//	free(T_h);

/* Create vertical T0 profile - static T0 */
	Calculation::Vertical_Profile_T_const (T_refh);
//	Calculation::Vertical_Profile_T_const2 (T_h,T_refh);
//	free(T_h);

// for(int k=0;k<nz;k++){
// std::cout<<k<<" "<<T_refh[k]<<std::endl;
// }



/* Memory copy from Host to Device */
	FLOAT *T_ref;
	cudaMalloc ((void**) &T_ref, nz*sizeof(FLOAT));
	cudaMemcpy(T_ref,T_refh,nz*sizeof(FLOAT),cudaMemcpyHostToDevice);

	// for wall function
	//if(user_flags::flg_collision==1){
	//	cuda_stream_collision_T_D3Q27_cum   <<< grid_, block_2d_ >>>  (
	//			f, fn,
	//			T, Tn,
	//			l_obs,
	//			u, v, w, rho,
	//			vis_lbm,
	//			vis_sgs,
	//			T_ref,
	//			Fcs,
	//			bcTw, //  1,  0,  0 //
	//			bcTe, // -1,  0,  0 //
	//			bcTs, //  0,  1,  0 //
	//			bcTn, //  0, -1,  0 //
	//			bcTr, //  0,  0,  1 //
	//			dx_, dt_,
	//			c_ref_,
	//			nx_,     ny_,     nz_,
	//			halo_
	//			);
	//}else if(user_flags::flg_wallFunction==1){
	//	cuda_stream_collision_T_D3Q27_wall   <<< grid_, block_2d_ >>>  (
	//			f, fn,
	//			T, Tn,
	//			l_obs,
	//			l_obs_x, l_obs_y, l_obs_z,
	//			u, v, w, rho,
	//			vis_lbm,
	//			vis_sgs,
	//			T_ref,
	//			Fcs,
	//			bcTw, //  1,  0,  0 //
	//			bcTe, // -1,  0,  0 //
	//			bcTs, //  0,  1,  0 //
	//			bcTn, //  0, -1,  0 //
	//			bcTr, //  0,  0,  1 //
	//			dx_, dt_,
	//			c_ref_,
	//			nx_,     ny_,     nz_,
	//			halo_
	//			);
	//}else{
		cuda_stream_collision_T_D3Q27   <<< grid_, block_2d_ >>>  (
				f, fn,
				T, Tn,
				l_obs,
				u, v, w, rho,
				vis_lbm,
				vis_sgs,
	//			vis_sgs_old,	//(YOKOUCHI 2020)
	//			tke_sgs,	//(YOKOUCHI 2020)
	//			tke_sgs_old,	//(YOKOUCHI 2020)
				T_ref,
				Fcs,
				bcTw, //  1,  0,  0 //
				bcTe, // -1,  0,  0 //
				bcTs, //  0,  1,  0 //
				bcTn, //  0, -1,  0 //
				bcTr, //  0,  0,  1 //
				dx_, dt_,
				c_ref_,
				nx_,     ny_,     nz_,
				halo_
				);
	//}

	free(T_refh);
	cudaFree(T_ref);

#endif

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	Calculation::
set_force (
	      Stress			*str,
   	const FluidProperty		*cfp
	)
{
	// output //
	FLOAT	*force_x = str->force_x;
	FLOAT	*force_y = str->force_y;
	FLOAT	*force_z = str->force_z;


	// input //
	FLOAT	*l_obs = cfp->l_obs;

	cuda_gravity_force   <<< grid_, block_2d_ >>>  (
		force_x,
		force_y,
		force_z,
		l_obs,
		c_ref_,
		nx_,     ny_,     nz_,
		halo_
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	Calculation::
force_acceleration (
   	const Variables			*cq,
   	      Variables			*cqn,
	      Stress			*str
	)
{
#ifdef D3Q19_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q19_VELOCITY;
#endif
#ifdef D3Q27_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q27_VELOCITY;
#endif


	// input //
	const FLOAT	*f       = cq->f_n;
	const FLOAT	*force_x = str->force_x;
	const FLOAT	*force_y = str->force_y;
	const FLOAT	*force_z = str->force_z;

	// output //
	FLOAT	*fn      = cqn->f_n;


	cuda_force_acceleration   <<< grid_, block_2d_ >>>  (
		f,
		fn,
		force_x,
		force_y,
		force_z,
		nx_,     ny_,     nz_,
		halo_,
		lbm_velocity_model
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// obstacle //
void	Calculation::
obstacle_variables (
   	      Variables			*cq,
	      BasisVariables	*cbq,
   	const FluidProperty		*cfp
	)
{
#ifdef D3Q19_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q19_VELOCITY;
#endif
#ifdef D3Q27_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q27_VELOCITY;
#endif

	// u, v, w //
//	status_lv_velocity <<< grid_, block_2d_ >>>  (
//		cbq->r_n,
//		cbq->u_n, cbq->v_n, cbq->w_n,
//		cfp->l_obs,
//		nx_,     ny_,     nz_,
//		halo_
//		);

	status_lv_lbm_velocity <<< grid_, block_2d_ >>>  (
		cq->f_n,
		cq->T_n,
		cbq->r_n,
		cbq->u_n, cbq->v_n, cbq->w_n,
		cfp->l_obs,
		nx_,     ny_,     nz_,
		halo_,
		lbm_velocity_model
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// lbm_function_to_velocity //
void	Calculation::
lbm_function_to_velocity (
   	const Variables			*cq,
	      BasisVariables	*cbq
	)
{
#ifdef D3Q19_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q19_VELOCITY;
#endif
#ifdef D3Q27_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q27_VELOCITY;
#endif


	// input //
	const FLOAT	*f = cq->f_n;

	// output //
	FLOAT	*rho = cbq->r_n;
	FLOAT	*u   = cbq->u_n;
	FLOAT	*v   = cbq->v_n;
	FLOAT	*w   = cbq->w_n;


	cuda_lbm_function_to_velocity   <<< grid_, block_2d_ >>>  (
		f,
		rho,
		u, v, w,
		nx_,     ny_,     nz_,
		halo_,
		lbm_velocity_model
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// obstacle filter //
void	Calculation::
velocity_obstacle_filter (
	      BasisVariables	*cbq,
	const FluidProperty		*cfp
)
{
	// input //
	const FLOAT	*lv = cfp->l_obs;


	// output //
	FLOAT	*u   = cbq->u_n;
	FLOAT	*v   = cbq->v_n;
	FLOAT	*w   = cbq->w_n;
//	FLOAT	*T   = cq->T_n;		//MOD2018

	cuda_velocity_obstacle_filter   <<< grid_, block_2d_ >>>  (
		u, v, w,
		lv,
		nx_,     ny_,     nz_,
		halo_
		);


	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// velocity_to_lbm_function //
void	Calculation::
velocity_to_lbm_function (
   	      Variables			*cq,
	const BasisVariables	*cbq
	)
{
#ifdef D3Q19_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q19_VELOCITY;
#endif
#ifdef D3Q27_MODEL_
	LBM_VELOCITY_MODEL	lbm_velocity_model = D3Q27_VELOCITY;
#endif


	// input //
	const FLOAT	*rho = cbq->r_n;
	const FLOAT	*u   = cbq->u_n;
	const FLOAT	*v   = cbq->v_n;
	const FLOAT	*w   = cbq->w_n;

	// output //
	FLOAT	*f = cq->f_n;


	cuda_velocity_to_lbm_function   <<< grid_, block_2d_ >>>  (
		f,
		rho,
		u, v, w,
		nx_,     ny_,     nz_,
		halo_,
		lbm_velocity_model
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// mpi & boundary //
void	Calculation::
mpi_boundary_Variables (
	Variables		*cq,
   	BasisVariables	*cbq,
   	FluidProperty	*cfp
	)
{
	mpi_Variables      (cq, cbq, cfp);
	boundary_Variables (cq, cbq, cfp);
}


// mpi communication //
void	Calculation::
mpi_Variables (
	Variables		*cq,
   	BasisVariables	*cbq,
   	FluidProperty	*cfp
	)
{
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nz = nz_;


	// LBM //
	const int	nxyz = nx*ny*nz;
	const int	num_direction = NUM_DIRECTION_VEL;


	// MPI //
	const int	phy_num = (num_direction + 3 + 1 + 1); // f + u,v,w + lv + T
	FLOAT	**fp = new FLOAT*[phy_num];

	for (int i=0; i<num_direction; i++) {
		fp[i] = &cq->f_n[i*nxyz];
	}
	fp[num_direction  ] = cbq->u_n;
	fp[num_direction+1] = cbq->v_n;
	fp[num_direction+2] = cbq->w_n;

	// stFluidProperty.h //
	fp[num_direction+3] = cfp->l_obs;

    // T //
	fp[num_direction+4] = cq->T_n;


	MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier //
//	pmpi_.mpi_cuda_xyz(fp, phy_num);
	pmpi_.mpi_cuda_x(fp, phy_num);
	pmpi_.mpi_cuda_y(fp, phy_num);
	MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier //


	delete [] fp;//	fp = NULL;
	// MPI //
}


void	Calculation::
boundary_Variables (
	Variables		*cq,
   	BasisVariables	*cbq,
   	FluidProperty	*cfp
	)
{
	boundary_x_Variables (cq, cbq, cfp);
//	boundary_y_Variables (cq, cbq, cfp);
	boundary_z_Variables (cq, cbq, cfp);
}


void	Calculation::
boundary_x_Variables (
	Variables		*cq,
   	BasisVariables	*cbq,
   	FluidProperty	*cfp
	)
{
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nz = nz_;

	const int	rank_x = rank_x_;
	const int	ncpu_x = ncpu_x_;

	// x boundary
	dim3	grid_yz,
			block_yz;

	const int	ny_cast = functionLib::cast_value_large(ny, BLOCKDIM_Y);
	const int	nz_cast = functionLib::cast_value_large(nz, BLOCKDIM_Z);
	functionLib::set_dim3(&grid_yz,  ny_cast/BLOCKDIM_Y, nz_cast/BLOCKDIM_Z, 1);
	functionLib::set_dim3(&block_yz, BLOCKDIM_Y, BLOCKDIM_Z, 1);

	// boundary x //
//	boundary_LBM_x_D3Q19_inflow_outflow <<< grid_yz,  block_yz >>>  (	// diverge //
	boundary_LBM_x_inflow_outflow <<< grid_yz,  block_yz >>>  (
//	boundary_LBM_x_inflow_outflow_driver <<< grid_yz,  block_yz >>>  (
		rank_x,
	   	ncpu_x,
		 cq->f_n,
		 cq->T_n,
		cbq->r_n,
		cbq->u_n,
		cbq->v_n,
		cbq->w_n,
		RHO_REF,
		cfl_ref_,
//		dx_,			// MOD2019
		nx, ny, nz
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	Calculation::
boundary_y_Variables (
	Variables		*cq,
   	BasisVariables	*cbq,
   	FluidProperty	*cfp
	)
{
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nz = nz_;

	const int	rank_y = rank_y_;
	const int	ncpu_y = ncpu_y_;

	// y boundary //
	dim3	grid_xz,
			block_xz;

	const int	nx_cast = functionLib::cast_value_large(nx, BLOCKDIM_X);
	const int	nz_cast = functionLib::cast_value_large(nz, BLOCKDIM_Z);
	functionLib::set_dim3(&grid_xz,  nx_cast/BLOCKDIM_X, nz_cast/BLOCKDIM_Z, 1);
	functionLib::set_dim3(&block_xz, BLOCKDIM_X, BLOCKDIM_Z, 1);


	boundary_LBM_y_Neumann <<< grid_xz,  block_xz >>>  (
		rank_y,
	   	ncpu_y,
		 cq->f_n,
		cbq->r_n,
		cbq->u_n,
		cbq->v_n,
		cbq->w_n,
		nx, ny, nz);


	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


void	Calculation::
boundary_z_Variables (
	Variables		*cq,
   	BasisVariables	*cbq,
   	FluidProperty	*cfp
	)
{
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nz = nz_;

	const int	rank_z = rank_z_;
	const int	ncpu_z = ncpu_z_;

	// z boundary //
	dim3	grid_xy,
			block_xy;

	const int	nx_cast = functionLib::cast_value_large(nx, BLOCKDIM_X);
	const int	ny_cast = functionLib::cast_value_large(ny, BLOCKDIM_Y);
	functionLib::set_dim3(&grid_xy,  nx_cast/BLOCKDIM_X, ny_cast/BLOCKDIM_Y, 1);
	functionLib::set_dim3(&block_xy, BLOCKDIM_X, BLOCKDIM_Y, 1);

        // temperature
	const FLOAT	pt_ref = user_init::pt_ref;

////	boundary_LBM_z_Neumann <<< grid_xy,  block_xy >>>  (
////	boundary_LBM_z_Slip <<< grid_xy,  block_xy >>>  (

	boundary_LBM_z_Upper <<< grid_xy,  block_xy >>>  (
		rank_z,
	   	ncpu_z,
		 cq->f_n,
		 cq->T_n,
		cbq->r_n,
		cbq->u_n,
		cbq->v_n,
		cbq->w_n,
		RHO_REF,
		cfl_ref_,
		pt_ref,
		nx, ny, nz
		);


//#ifdef D3Q19_MODEL_
//	boundary_LBM_z_D3Q19_Upper <<< grid_xy,  block_xy >>>  (
//		rank_z,
//	   	ncpu_z,
//		 cq->f_n,
//		cbq->r_n,
//		cbq->u_n,
//		cbq->v_n,
//		cbq->w_n,
//		nx, ny, nz
//		);
//#endif
//#ifdef D3Q27_MODEL_
//	boundary_LBM_z_D3Q27_Upper <<< grid_xy,  block_xy >>>  (
//		rank_z,
//	   	ncpu_z,
//		 cq->f_n,
//		cbq->r_n,
//		cbq->u_n,
//		cbq->v_n,
//		cbq->w_n,
//		nx, ny, nz
//		);
//#endif


	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
}


// For Boussinesq approximation   MOD2018
void Calculation::Vertical_Profile_T (const FLOAT *T, FLOAT *T_ref )
{
	int id;
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nz = nz_;
	const int	nxy = nx * ny;

	FLOAT*  T_ref0;
	T_ref0 = new FLOAT[nz];

	for (int k=0;k<nz;k++) {
		T_ref0[k] = 0.0;
		T_ref[k]  = 0.0;
	}
	for (int k=0; k<nz; k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				id = i + nx*j + nxy*k;
				T_ref0[k] = T_ref0[k] + T[id];
			}
		}
	}

	T_ref[0]    = T_ref0[0] + T_ref0[1] + T_ref0[2];
	T_ref[1]    = T_ref[0] + T_ref0[3];
	T_ref[nz-1] = T_ref0[nz-1] + T_ref0[nz-2] + T_ref0[nz-3];
	T_ref[nz-2] = T_ref[nz-1] + T_ref0[nz-4];
	T_ref[0]    = T_ref[0] / (3*nxy);
	T_ref[1]    = T_ref[1] / (4*nxy);
	T_ref[nz-1] = T_ref[nz-1] / (3*nxy);
	T_ref[nz-2] = T_ref[nz-2] / (4*nxy);
	for (int k=2;k<nz-2;k++) {
		for (int kk=-2;kk<3;kk++) {
			T_ref[k] = T_ref[k] + T_ref0[k+kk];
		}
		T_ref[k] = T_ref[k] / (5 * nxy);
	}

	delete [] T_ref0;

//	for (int k=0;k<nz;k++) {
//		T_ref[k] = T_ref[k] / nxy;
//	}
}
void Calculation::Vertical_Profile_T_const (FLOAT *T_ref )
{
	const int	nz = nz_;
	const FLOAT	T_base	= 300.0;
	const FLOAT	dtemp	= 0.01;
	const FLOAT	dz	= 20.0;
	const FLOAT	zi	= 500.0;
	for (int k=0;k<nz;k++) {
//		T_ref[k] = ((k*dz<zi)?	T_base  : T_base + (k*dz-zi)*dtemp);
		T_ref[k] = T_base;
	}
}

void Calculation::Vertical_Profile_T_const2 (const FLOAT *T, FLOAT *T_ref)
{
	int		id;
	const FLOAT	T_base  = 300.0;
	const FLOAT	dtemp   = 0.01;
	const FLOAT	dz	= 20.0;
	const FLOAT	zi	= 500.0;
	const int	nz = nz_;
	const int	nx = nx_;
	const int	ny = ny_;
	const int	nxy = nx*ny;
	const int	kref = 20;

	FLOAT	T_ave = 0.0;
	for (int k=0;k<kref;k++) {
		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				id = i + nx*j + nxy*k;
				T_ave = T_ave + T[id];
			}
		}
	}
	T_ave = T_ave / (nxy*kref);

	for (int k=0;k<nz;k++) {
		T_ref[k] = ((k*dz<zi)?  T_base : T_base + (k*dz-zi)*dtemp)*1.3;
		T_ref[k] = ((T_ref[k]<T_ave)?  T_ave : T_ref[k])*1.3;
	}

}

