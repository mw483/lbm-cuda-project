#include "classReadSTL.h"

#include "macroCUDA.h"
#include "defineCoefficient.h"
#include "defineCUDA.h"

// Lib //
#include "functionLib.h"
#include "indexLib.h"
#include "allocateLib.h"

// stl_solid
#include "cstdio"
#include "STL_gpu.h"
//#include "solid_ls.h"


// public //
// initialize //
void classReadSTL::
init_classReadSTL (
		  int		rank,
	const Domain	&domain,
	const solidData &soliddata
	)
{
	set (
		rank,
		domain
		);


	// STL data from i/o disk //
	read_disk_stl_solid (
		stl_solid_h_,
		stl_solid_d_
		);
}


// STLデータの読み込み //
void classReadSTL::
read_STL_levelset_data (
		      int		*id_obs,
		      FLOAT		*lv,
		const solidData &soliddata
	)
{
	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;

	// axis //
	const FLOAT	dx = domain_.dx;

	FLOAT	*x = domain_.x;
	FLOAT	*y = domain_.y;
	FLOAT	*z = domain_.z;

	const int	n_parts = 1; //

	// LevelSet関数の補間 //
	// scale //
	const FLOAT	scale = 1.0e-3; // 単位系 (m = 1, mm=0.001)


	// soliddata offset //
	const FLOAT	x_solid = soliddata.x_s / scale;
	const FLOAT	y_solid = soliddata.y_s / scale;
	const FLOAT	z_solid = soliddata.z_s / scale;


	translate_solid (
		x_solid,
		y_solid,
		z_solid,
		&(stl_solid_h_[0].stl_solid_info)
		);


	// levelset //
	interpolate_solid_data (
		id_obs,
		lv, 
		x,  y,  z,
		nx, ny, nz,
		dx,
		scale,
		n_parts,
		stl_solid_h_[0].stl_solid_info,
		stl_solid_h_[0].stl_solid_parts
		);
	// axis //
}


void classReadSTL::
read_STL_levelset_data_gpu (
		      int		*id_obs,
		      FLOAT		*lv,
		const solidData &soliddata
	)
{
	const int	halo = domain_.halo;

	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;

	const FLOAT	dx = domain_.dx;

	// axis //
	FLOAT	*x_h = domain_.x;
	FLOAT	*y_h = domain_.y;
	FLOAT	*z_h = domain_.z;

	static int	flg;

	FLOAT	*x_d;
	FLOAT	*y_d;
	FLOAT	*z_d;

	allocateLib::new_device (&x_d,  nx);
	allocateLib::new_device (&y_d,  ny);
	allocateLib::new_device (&z_d,  nz);

	CUDA_SAFE_CALL( cudaMemcpy(x_d,   x_h,   sizeof(FLOAT)*(nx), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(y_d,   y_h,   sizeof(FLOAT)*(ny), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(z_d,   z_h,   sizeof(FLOAT)*(nz), cudaMemcpyHostToDevice) );


	if (flg == 0)	flg = 1;

	const int	n_parts = 1; //

	// LevelSet関数の補間 //
	// scale //
	const FLOAT	scale = 1.0e-3; // 単位系 (m = 1, mm=0.001)


	// soliddata offset //
//	static int	step;
//	const FLOAT	x_solid = (soliddata.x_s + 0.00001*step) / scale;
//	const FLOAT	y_solid = (soliddata.y_s + 0.00001*step) / scale;
//	const FLOAT	z_solid = (soliddata.z_s + 0.00001*step) / scale;
//	step++;

	const FLOAT	x_solid = soliddata.x_s / scale;
	const FLOAT	y_solid = soliddata.y_s / scale;
	const FLOAT	z_solid = soliddata.z_s / scale;

	translate_solid (
		x_solid,
		y_solid,
		z_solid,
		&(stl_solid_d_[0].stl_solid_info)
		);


	// levelset //
	interpolate_solid_data_gpu <<< grid_, block_2d_ >>>  (
		id_obs,
		lv, 
		x_d,  y_d,  z_d,
		nx, ny, nz,
		dx,
		scale,
		n_parts,
		stl_solid_d_[0].stl_solid_info,
		stl_solid_d_[0].stl_solid_parts->coi,
		stl_solid_d_[0].stl_solid_parts->fs,
		halo
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //
	// axis //

	CUDA_SAFE_CALL( cudaFree(x_d) );
	CUDA_SAFE_CALL( cudaFree(y_d) );
	CUDA_SAFE_CALL( cudaFree(z_d) );
}


void classReadSTL::
read_STL_velocity_data_gpu (
		const FLOAT		*l_obs,
			  FLOAT		*u_obs,
			  FLOAT		*v_obs,
			  FLOAT		*w_obs,
		const solidData &soliddata
	)
{
	const int	halo = domain_.halo;

	const int	nx = domain_.nx;
	const int	ny = domain_.ny;
	const int	nz = domain_.nz;

	const FLOAT	dx = domain_.dx;

//	const FLOAT	c_ref = domain_.c_ref;
	const FLOAT	rps   =  0.0; // LBM time //
//	const FLOAT	rps   =  5.0; // LBM time //

	// LevelSet関数の補間 //
	// scale //
//	const FLOAT	x_s = soliddata.x_s;
//	const FLOAT	y_s = soliddata.y_s;
//	const FLOAT	z_s = soliddata.z_s;
//
//	const FLOAT	u_s = soliddata.u_s;
//	const FLOAT	v_s = soliddata.v_s;
//	const FLOAT	w_s = soliddata.w_s;


	// levelset //
//	interpolate_solid_velocity_gpu <<< grid_, block_2d_ >>>  (
//		u_obs,
//		v_obs,
//		w_obs,
//		l_obs,
//		x_s, y_s, z_s,
//		u_s, v_s, w_s,
//		nx, ny, nz,
//		dx,
//		halo
//		);

	FLOAT	*x_n;
	FLOAT	*y_n;
	FLOAT	*z_n;

	allocateLib::new_device (&x_n,  nx);
	allocateLib::new_device (&y_n,  ny);
	allocateLib::new_device (&z_n,  nz);

	cudaMemcpy(x_n, domain_.x, sizeof(FLOAT)*(nx), cudaMemcpyDefault);
	cudaMemcpy(y_n, domain_.y, sizeof(FLOAT)*(ny), cudaMemcpyDefault);
	cudaMemcpy(z_n, domain_.z, sizeof(FLOAT)*(nz), cudaMemcpyDefault);


	interpolate_solid_velocity_spin_gpu <<< grid_, block_2d_ >>>  (
		u_obs,
		v_obs,
		w_obs,
		l_obs,
		x_n,
		y_n,
		z_n,
		rps,
		nx, ny, nz,
		dx,
		halo
		);

	CHECK_CUDA_ERROR("CUDA Error\n");
	cudaThreadSynchronize(); // cudaThreadSynchronize //

	cudaFree (x_n);
	cudaFree (y_n);
	cudaFree (z_n);
}




// private //
void classReadSTL::
set (
	      int		rank,
	const Domain	&domain
	)
{
	// mpiinfo //
	rank_ = rank;


	// domain //
	domain_ = domain;


	// cuda
	functionLib::set_dim3(&grid_,
		(domain_.nx-2*domain_.halo)/BLOCKDIM_X, 
		(domain_.ny-2*domain_.halo)/BLOCKDIM_Y, 
		(domain_.nz-2*domain_.halo)/BLOCKDIM_Z
		);


	functionLib::set_dim3(&block_2d_,
		BLOCKDIM_X,
		BLOCKDIM_Y,
		1
		);
}


// STL //
void classReadSTL::
read_disk_stl_solid (
	stl_solid	*&stl_solid_h,
	stl_solid	*&stl_solid_d
	)
{
	// CPU //
	// STL data load
//	const int	n_parts = num_stl_solid_;
	const int	n_parts = 1;


	// メモリ確保
	stl_solid_h = (stl_solid *) malloc( sizeof(stl_solid)*n_parts );
	stl_solid_d = (stl_solid *) malloc( sizeof(stl_solid)*n_parts );
	for (int i=0; i<n_parts; i++) {
		stl_solid_h[i].stl_solid_parts = (stl_solid_parts *) malloc( sizeof(stl_solid_parts) );
		stl_solid_d[i].stl_solid_parts = (stl_solid_parts *) malloc( sizeof(stl_solid_parts) );
	}


	// file name
	char	**parts_filename = (char **) malloc(sizeof(char *)*n_parts);
	for (int i=0; i<n_parts; i++) {
		parts_filename[i] = (char *) malloc(sizeof(char)*256);
	}


	// read data //
	sprintf(parts_filename[0], "./stl_golf/sphere_v2.dat");
//	sprintf(parts_filename[0], "./stl_golf/golf_ball_v2.dat");
	// file name


	// STLデータの読み込み & 距離関数のメモリ確保 //
	for(int i=0; i<n_parts; i++) {
		if (rank_ == 0) { printf("stl data : solid_filename=\"%s\"\n", parts_filename[i]); }

		load_solid_v2 (
				&(stl_solid_h[i].stl_solid_info),
				 (stl_solid_h[i].stl_solid_parts), 
				parts_filename[i]);


		allocate_device_memory_solid (
				&(stl_solid_h[i].stl_solid_info), 
				 (stl_solid_h[i].stl_solid_parts), 
				&(stl_solid_d[i].stl_solid_info), 
				 (stl_solid_d[i].stl_solid_parts)
				);
	}
	// CPU //
}


// load host //
void classReadSTL::
load_solid_v2
// =======================================================================
//
// purpos     :  load level-set function for solid data (version 2.0)
//
// date       :  2013, Jan 28
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
	stl_solid_info	*sinfo,		/* level-set data for solid parts	*/
	stl_solid_parts	*parts,		/* level-set data for solid parts	*/
	char	*filename		/* level-set data filename		*/
)
// -----------------------------------------------------------------------
{
	int		nlen;

	// fread 一時変数
	char	*comment;

	int		nx , ny , nz ;
	int		nxs, nys, nzs;

	double	dx, dy, dz;
	double	Lx, Ly, Lz;
	double	xoff, yoff, zoff;

	FILE	*fp = fopen(filename,"rb");

	// fread //
	rewind(fp);
	fread(&nlen, sizeof(int), 1, fp);
	comment = (char *) malloc(sizeof(char)*nlen);
	fread(comment, sizeof(char), nlen, fp);

	fread(&nx  , sizeof(int),    1, fp);
	fread(&ny  , sizeof(int),    1, fp);
	fread(&nz  , sizeof(int),    1, fp);
	fread(&nxs , sizeof(int),    1, fp);
	fread(&nys , sizeof(int),    1, fp);
	fread(&nzs , sizeof(int),    1, fp);

	fread(&dx  , sizeof(double), 1, fp);
	fread(&dy  , sizeof(double), 1, fp);
	fread(&dz  , sizeof(double), 1, fp);
	fread(&Lx  , sizeof(double), 1, fp);
	fread(&Ly  , sizeof(double), 1, fp);
	fread(&Lz  , sizeof(double), 1, fp);
	fread(&xoff, sizeof(double), 1, fp);
	fread(&yoff, sizeof(double), 1, fp);
	fread(&zoff, sizeof(double), 1, fp);
	// fread //

	// 構造体変数へのキャスト
	sinfo->nx   = nx  ;		sinfo->nxs  = nxs ;
	sinfo->ny   = ny  ;		sinfo->nys  = nys ;
	sinfo->nz   = nz  ;		sinfo->nzs  = nzs ;

	sinfo->dx   = dx  ;		sinfo->Lx   = Lx  ;
	sinfo->dy   = dy  ;		sinfo->Ly   = Ly  ;
	sinfo->dz   = dz  ;		sinfo->Lz   = Lz  ;

	sinfo->xoff = xoff;
	sinfo->yoff = yoff;
	sinfo->zoff = zoff;

	const int	nt = (nxs + 1)*(nys + 1)*(nzs + 1);

	// 距離関数等のメモリ確保
	parts->coi = (char  * ) malloc(sizeof(char   )*nx*ny*nz);
	parts->fs  = (float **) malloc(sizeof(float *)*nx*ny*nz);

	fread(parts->coi, sizeof(char), nx*ny*nz, fp);

//	static FLOAT	f_max;
	float	*f_tmp = new float[nt];
	for (int i=0; i<nx*ny*nz; i++) {
		if (parts->coi[i] == 'x') {
			int		ij;

			fread(&ij, sizeof(int), 1, fp);
			if(i != ij) { printf("error in load_solid_v2:\n");   exit(0); }

			parts->fs[i] = (float *) malloc(sizeof(float)*nt);
			const int	ie = fread(f_tmp, sizeof(float), nt, fp);

			for (int j=0; j<nt; j++) {
				parts->fs[i][j] = f_tmp[j];

//				f_max = fmax(f_max, f_tmp[j]);
			}

			if (ie == 0)	{ printf("error: \n");  exit(-1); }
		}
	}
	delete [] f_tmp;

//	std::cout << "f_max = " << f_max << std::endl;

	sinfo->ox = 0.0;	sinfo->oy = 0.0;	sinfo->oz = 0.0;
	sinfo->hx = 0.0;	sinfo->hy = 0.0;	sinfo->hz = 0.0;
	sinfo->lx = 0.0;	sinfo->ly = 0.0;	sinfo->lz = 0.0;

	sinfo->sinhx = 0.0;	sinfo->coshx = 1.0;
	sinfo->sinhy = 0.0;	sinfo->coshy = 1.0;
	sinfo->sinhz = 0.0;	sinfo->coshz = 1.0;

	fclose(fp);
}


void classReadSTL::
allocate_device_memory_solid
(
	const stl_solid_info	*sinfo_h,		/* level-set data for solid parts (Host)	*/
	const stl_solid_parts	*parts_h,		/* level-set data for solid parts (Host)	*/
	      stl_solid_info	*sinfo_d,		/* level-set data for solid parts (Device)	*/
	      stl_solid_parts	*parts_d		/* level-set data for solid parts (Device)	*/
)
// -----------------------------------------------------------------------
{
	// host to device
	// copy solid_infom
	cudaMemcpy(sinfo_d, sinfo_h, sizeof(stl_solid_info), cudaMemcpyDefault);

	const int	nx   = sinfo_h->nx;		const int	nxs  = sinfo_h->nxs;
	const int	ny   = sinfo_h->ny;		const int	nys  = sinfo_h->nys;
	const int	nz   = sinfo_h->nz;		const int	nzs  = sinfo_h->nzs;

	const int	nt = (nxs + 1)*(nys + 1)*(nzs + 1);

	// malloc (device memory)
	// char *coi
	cudaMalloc((void **)&parts_d->coi,     sizeof(char)*(nx*ny*nz)); 
	cudaMemcpy(parts_d->coi, parts_h->coi, sizeof(char)*(nx*ny*nz), cudaMemcpyDefault);


	// FLOAT **fs
	cudaMalloc((void **)&parts_d->fs, sizeof(float *) * nx*ny*nz); 


	// deviceメモリで有効な配列の個数
	int		coi_num_sum = 0;
	int		*coi_num    = (int *) malloc(sizeof(int)*nx*ny*nz);

	for (int i=0; i<nx*ny*nz; i++) {
		coi_num[i] = coi_num_sum;

		if (parts_h->coi[i] == 'x') {
			coi_num_sum++;
		}
	}

	// deviceメモリ確保
	// deviceメモリ 実体
	float	*fs_tmp_d;
	cudaMalloc((void **)&fs_tmp_d,     sizeof(float)*(nt*coi_num_sum)); 

	// deviceメモリ アドレス
	float	**fs_address = (float **) malloc(sizeof(float *)*nx*ny*nz);
	for (int i=0; i<nx*ny*nz; i++) { fs_address[i] = NULL; }

	for (int i=0; i<nx*ny*nz; i++) {
		if (parts_h->coi[i] == 'x') {
			const int	fs_offset = coi_num[i] * nt;
			fs_address[i] = fs_tmp_d + fs_offset;

			// ２重ポインタ先のメモリコピー
			cudaMemcpy(&fs_tmp_d[fs_offset], parts_h->fs[i], sizeof(float)*(nt), cudaMemcpyDefault);
		}
	}

//	cudaMemcpy(parts_d->fs, fs_address, sizeof(FLOAT *)*(nx*ny*nz), cudaMemcpyDefault);
	cudaMemcpy(parts_d->fs, fs_address, sizeof(float *)*(nx*ny*nz), cudaMemcpyDefault);

	delete [] coi_num;
	delete [] fs_address;
}


// LevelSet関数の補間
void classReadSTL::
interpolate_solid_data (
	      int	*id_obs,
	      FLOAT	*solid_ls,	/* level-set data for solid		*/
	const FLOAT	*x,
	const FLOAT	*y,
	const FLOAT	*z,
	      int	nx,
	      int	ny,
	      int	nz,
		  FLOAT	dx,
	      FLOAT	scale,
	      int	nparts,
	      stl_solid_info	info_1,
	const stl_solid_parts	*parts_1
	)
{
	const FLOAT	coef    = 5.0;
	const FLOAT	stl_dis = dx / scale * coef;


	for (int id_z=0; id_z<nz; id_z++) {
		for (int id_y=0; id_y<ny; id_y++) {
			for (int id_x=0; id_x<nx; id_x++) {
				const int	id_g = id_x + nx*id_y + nx*ny*id_z;


				// global axis //
				const FLOAT	x_stl = x[id_x] / scale; 
				const FLOAT	y_stl = y[id_y] / scale;
				const FLOAT	z_stl = z[id_z] / scale;


				// calculation //
//				const FLOAT	lv1 = ls_parts_v2 ( x_stl, y_stl, z_stl, info_1, parts_1 );
				const FLOAT	lv1 = ls_parts_v2 ( x_stl, y_stl, z_stl, info_1, parts_1->coi, parts_1->fs );

				FLOAT	lv = lv1;

				int		id_obstacle = 0;	// solid_info0 //
				if      (lv < -stl_dis)	{ lv = -stl_dis;	id_obstacle = -1; }
				else if (lv >  stl_dis)	{ lv =  stl_dis; 	id_obstacle = -1; }


				// update //
				id_obs  [id_g] = id_obstacle;
				solid_ls[id_g] = -lv * scale;
			}
		}
	}
}


__global__ void
interpolate_solid_data_gpu (
	      int	*id_obs,
	      FLOAT	*solid_ls,	/* level-set data for solid		*/
	const FLOAT	*x,
	const FLOAT	*y,
	const FLOAT	*z,
	      int	nx,
	      int	ny,
	      int	nz,
		  FLOAT	dx,
	      FLOAT	scale,
	      int	nparts,
	      stl_solid_info	info_1,
	const char				*coi, 
	      float				**fs,
		  int	halo
	)
{
	const FLOAT	coef    = 5.0;
	const FLOAT	stl_dis = dx / scale * coef;


	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// local index
		const int	index[3] = { id_x, id_y, id_z };
		const int	grid [3] = { nx, ny, nz };

		// global index
		const int	id_c0_c0_c0 = indexLib::get_index (index, grid,  0,  0,  0);

		// global axis //
		const FLOAT	x_stl = x[id_x] / scale; 
		const FLOAT	y_stl = y[id_y] / scale;
		const FLOAT	z_stl = z[id_z] / scale;


		// calculation //
		const FLOAT	lv1 = ls_parts_v2_gpu ( x_stl, y_stl, z_stl, info_1, coi, fs );

		FLOAT	lv = lv1;

		int		id_obstacle = 0;	// solid_info0 //
		if      (lv < -stl_dis)	{ lv = -stl_dis;	id_obstacle = -1; }
		else if (lv >  stl_dis)	{ lv =  stl_dis; 	id_obstacle = -1; }


		// update //
		id_obs  [id_c0_c0_c0] = id_obstacle;
		solid_ls[id_c0_c0_c0] = -lv * scale;
	}
}


__global__ void
interpolate_solid_velocity_gpu (
	      FLOAT	*u_obs,
	      FLOAT	*v_obs,
	      FLOAT	*w_obs,
	const FLOAT	*l_obs,
		  FLOAT	x_s,	// position //
		  FLOAT	y_s,
		  FLOAT	z_s,
		  FLOAT	u_s,	// velocity //
		  FLOAT	v_s,
		  FLOAT	w_s,
	      int	nx,
	      int	ny,
	      int	nz,
		  FLOAT	dx,
		  int	halo
	)
{
	const FLOAT	coef    = 5.0;

	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// local index
		const int	index[3] = { id_x, id_y, id_z };
		const int	grid [3] = { nx, ny, nz };

		// global index
		const int	id_c0_c0_c0 = indexLib::get_index (index, grid,  0,  0,  0);

		const FLOAT	lv = l_obs[id_c0_c0_c0];

		FLOAT	u_obstacle = u_s;
		FLOAT	v_obstacle = v_s;
		FLOAT	w_obstacle = w_s;

		if (fabs(lv) > coef*dx) {
			u_obstacle = 0.0;
			v_obstacle = 0.0;
			w_obstacle = 0.0;
		}

		// update //
		u_obs  [id_c0_c0_c0] = u_obstacle;
		v_obs  [id_c0_c0_c0] = v_obstacle;
		w_obs  [id_c0_c0_c0] = w_obstacle;
	}
}


__global__ void
interpolate_solid_velocity_spin_gpu (
	      FLOAT*	u_obs,
	      FLOAT*	v_obs,
	      FLOAT*	w_obs,
	const FLOAT*	l_obs,
	const FLOAT*	x_n,
	const FLOAT*	y_n,
	const FLOAT*	z_n,
	const FLOAT		rps,
	const int		nx,
	const int		ny,
	const int		nz,
	const FLOAT		dx,
	const int		halo
	)
{
	const FLOAT	coef    = 5.0;

	const int	id_x  = halo + threadIdx.x + BLOCKDIM_X*(blockIdx.x),
				id_y  = halo + threadIdx.y + BLOCKDIM_Y*(blockIdx.y),
				id_zs = halo + threadIdx.z + BLOCKDIM_Z*(blockIdx.z);


	for (int k=0; k<BLOCKDIM_Z; k++) {
		const int	id_z  = id_zs + k;

		// local index
		const int	index[3] = { id_x, id_y, id_z };
		const int	grid [3] = { nx, ny, nz };

		// global index
		const int	id_c0_c0_c0 = indexLib::get_index (index, grid,  0,  0,  0);

		const FLOAT	lv = l_obs[id_c0_c0_c0];

		// rotation //
//		const FLOAT	omega = 10.0 / c_ref;

		const FLOAT	x_th = x_n[id_x];
		const FLOAT	y_th = y_n[id_y];
		const FLOAT	z_th = z_n[id_z];

		const FLOAT	ep = coefficient::NON_ZERO_EP;
		const FLOAT	r_th = sqrt( pow(x_th, 2) + pow(y_th, 2) ) + ep;

		// xy rotation //
//		const FLOAT	sin_th = y_th / r_th;
//		const FLOAT	cos_th = x_th / r_th;
//
//		FLOAT	u_obstacle = -r_th * rps * sin_th;
//		FLOAT	v_obstacle =  r_th * rps * cos_th;
//		FLOAT	w_obstacle =  0.0;

		// yz rotation //
		const FLOAT	sin_th = z_th / r_th;
		const FLOAT	cos_th = y_th / r_th;

		FLOAT	u_obstacle =  0.0;
		FLOAT	v_obstacle = -r_th * rps * sin_th;
		FLOAT	w_obstacle =  r_th * rps * cos_th;


		if (fabs(lv) > coef*dx) {
			u_obstacle = 0.0;
			v_obstacle = 0.0;
			w_obstacle = 0.0;
		}

		// update //
		u_obs  [id_c0_c0_c0] = u_obstacle;
		v_obs  [id_c0_c0_c0] = v_obstacle;
		w_obs  [id_c0_c0_c0] = w_obstacle;
	}
}


void classReadSTL::
translate_solid
(
	FLOAT	lx,		/* x-directional translation length	*/
	FLOAT	ly,		/* y-directional translation length	*/
	FLOAT	lz,		/* z-directional translation length	*/
	stl_solid_info	*sinfo		/* level-set data for solid parts	*/
)
// -----------------------------------------------------------------------
{
	sinfo->lx = lx;
	sinfo->ly = ly;
	sinfo->lz = lz;
}


void classReadSTL::
rotate_solid
//
(
	FLOAT	hx,	/* rotational angle (radia) around x-axis	*/
	FLOAT	hy,	/* rotational angle (radia) around y-axis	*/
	FLOAT	hz,	/* rotational angle (radia) around z-axis	*/
	stl_solid_info	*sinfo		/* level-set data for solid parts	*/
)
// -----------------------------------------------------------------------
{
	sinfo->sinhx = sin(-hx);	sinfo->coshx = cos(-hx);
	sinfo->sinhy = sin(-hy);	sinfo->coshy = cos(-hy);
	sinfo->sinhz = sin(-hz);	sinfo->coshz = cos(-hz);
}


// classReadSTL.cu //
