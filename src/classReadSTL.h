#ifndef CLASSREADSTL_H_
#define CLASSREADSTL_H_


#include "definePrecision.h"
#include "defineMemory.h"

#include "stDomain.h"


#include "stSTL.h"
#include "classSolidData.h"


class
classReadSTL {
private:
	// mpiinfo //
	int		rank_;

	// domain //
	Domain	domain_;

	// solid //
	stl_solid	*stl_solid_h_;
	stl_solid	*stl_solid_d_;

	// cuda
	dim3	grid_,
			block_2d_;

public:	
	classReadSTL () {}

	classReadSTL (
		      int		rank,
		const Domain	&domain,
		const solidData &soliddata
		)
	{
		init_classReadSTL (
			rank,
			domain,
			soliddata
			);
	}

	~classReadSTL () {}


	stl_solid		*stl_solid_h() { return stl_solid_h_; }
	stl_solid		*stl_solid_d() { return stl_solid_d_; }

public:
	void
	init_classReadSTL (
			  int		rank,
		const Domain	&domain,
		const solidData &soliddata
		);


	// STLデータの読み込み
	void
	read_STL_levelset_data (
		      int		*id_obs,
		      FLOAT		*lv,
		const solidData &soliddata
		);


	void
	read_STL_levelset_data_gpu (
		      int		*id_obs,
		      FLOAT		*lv,
		const solidData &soliddata
		);


	void
	read_STL_velocity_data_gpu (
			const FLOAT		*l_obs,
				  FLOAT		*u_obs,
				  FLOAT		*v_obs,
				  FLOAT		*w_obs,
			const solidData &soliddata
		);

private:
	void
	set (
		      int		rank,
		const Domain	&domain
		);


	// STL //
	void
	read_disk_stl_solid (
		stl_solid	*&stl_solid_h,
		stl_solid	*&stl_solid_d
		);


	// load host //
	void
	load_solid_v2
	(
		stl_solid_info	*sinfo,		/* level-set data for solid parts	*/
		stl_solid_parts	*parts,		/* level-set data for solid parts	*/
		char	*filename	/* level-set data filename		*/
	);


	void
	allocate_device_memory_solid
	(
		const stl_solid_info	*sinfo_h,		/* level-set data for solid parts (Host)	*/
		const stl_solid_parts	*parts_h,		/* level-set data for solid parts (Host)	*/
		      stl_solid_info	*sinfo_d,		/* level-set data for solid parts (Device)	*/
		      stl_solid_parts	*parts_d		/* level-set data for solid parts (Device)	*/
	);

	void
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
		);


	void	
	translate_solid (
		FLOAT	lx,		/* x-directional translation length	*/
		FLOAT	ly,		/* y-directional translation length	*/
		FLOAT	lz,		/* z-directional translation length	*/
		stl_solid_info	*sinfo		/* level-set data for solid parts	*/
	);


	void
	rotate_solid (
		FLOAT	hx,	/* rotational angle (radia) around x-axis	*/
		FLOAT	hy,	/* rotational angle (radia) around y-axis	*/
		FLOAT	hz,	/* rotational angle (radia) around z-axis	*/
		stl_solid_info	*sinfo		/* level-set data for solid parts	*/
	);
};


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
	const char				*coi_1, 
	      float				**fs_1,
	      int	halo
);


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
	);




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
);


#endif
