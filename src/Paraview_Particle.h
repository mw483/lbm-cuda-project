#ifndef PARAVIEW_PARTICLE_H_
#define PARAVIEW_PARTICLE_H_

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <limits.h>
#include "Define.h"
#include "Variables.h"
#include "VariablesParticle.h"
#include "Math_Lib_Particle_GPU.h"


class ParaviewParticle {
private :
	// MPIプロセスのランク
	int		mpi_rank_;
	int		ncpu_;

	// ファイル分割数
	int		ncpu_div_p_;

	int		halo_;

	// lbm //
	FLOAT	c_ref_;

public:
	ParaviewParticle () :
		mpi_rank_(0), ncpu_(1), ncpu_div_p_(1), halo_(1) {}

	ParaviewParticle (int mpi_rank, int ncpu) : 
		mpi_rank_(mpi_rank), ncpu_(ncpu), ncpu_div_p_(1), halo_(1) {}

	ParaviewParticle (int mpi_rank, int ncpu, int ncpu_div_p, int halo, FLOAT c_ref) : 
		mpi_rank_(mpi_rank), ncpu_(ncpu), halo_(halo), c_ref_(c_ref) {	ncpu_div_p_ = get_ncpu_div_p(ncpu_div_p);	}


	~ParaviewParticle() {}


	// ファイル分割数の設定
	int
	get_ncpu_div_p(int ncpu_div_p)
		{	return	(ncpu_%ncpu_div_p == 0) ? ncpu_div_p : 1;	}

public:
	void
	output_particle_binary_scatter_all (
		int 	t,
		const Domain			*const cdo,
		int		num_particle,
		const ParticlePosition	*const ppos,
		const BasisVariables	*const cbq,
		const Stress			*const str,
		int		downsize
		);

	// (YOKOUCHI 2020)
	void
	output_particle_binary_scatter_all_LSM (
		int 	t,
		const Domain			*const domain,
		int		num_particle,
		const ParticlePosition	*const ppos,
		const BasisVariables	*const cbq,
		const Stress			*const str,
		int		downsize
		);

	void
	Output_Particle_binary_scatter_All (
		int 	t,
		const Domain			*const cdo,
		int		num_particle,
		const ParticlePosition	*const ppos,
		const BasisVariables	*const cbq,
		const Stress			*const str,
		int		downsize
		);

private:
	// 粒子数の数え上げ
	int
	count_particles (
		const ParticlePosition *ppos,
		int		num_particle,
		int		downsize);


	// 構造体の粒子の属性を配列へコピー
	void
	copy_particle_source_index (
		int		*f_int,
		const ParticlePosition	*ppos,
		int		num_particle,
		int		downsize);


	// 構造体の粒子の速度を配列へコピー
	void
	copy_particle_velocity (
		float	*velocity,
		const ParticlePosition	*const ppos,
		int		num_particle,
		int		downsize);


	// 構造体の粒子座標を配列へコピー
	void
	copy_particle_position (
		float	*position,
		const ParticlePosition	*const ppos,
		int		num_particle,
		int		downsize);
	
	// 構造体の粒子座標を配列へコピー
	void
	copy_particle_uvw_sgs (
		float	*uvw_sgs,
		const ParticlePosition	*const ppos,
		int		num_particle,
		int		downsize);


	// 複数ファイルの結合
	void
	Combine_Data_count (
		int *count_out, 
		int count_in);


	void
	Combine_Data_int (
		int *f_out,
		int *f_in,
		const int count_rank[],
		int count_in,
		int count_max);
	

	void
	Combine_Data_FLOAT (
		FLOAT *f_out,
		FLOAT *f_in,
		const int count_rank[],
		int count_in,
		int count_max);


	void 
	Combine_Data_FLOAT3 (
		FLOAT *f_out,
		FLOAT *f_in,
		const int count_rank[],
		int count_in,
		int count_max);
};


#endif
