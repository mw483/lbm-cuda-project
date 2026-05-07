#include "Paraview_Particle.h"

#include "fileLib.h"
#include "defineReferenceVel.h"
#include "defineCoefficient.h"

// (YOKOUCHI 2020)
#include "Define_user.h"


void	ParaviewParticle::
output_particle_binary_scatter_all (
	int 	t,
	const Domain			*const domain,
	int		num_particle,
	const ParticlePosition	*const ppos,
	const BasisVariables	*const cbq,
	const Stress			*const str,
	int		downsize
	)
{
	const int	rank   = mpi_rank_;
	const int	fid    = rank%ncpu_div_p_;
	const int	lid    = rank/ncpu_div_p_;


	// 粒子数の数え上げ //
	int		count = count_particles (ppos, num_particle, downsize);

	int		count_rank[ncpu_div_p_];
	for (int i=0; i<ncpu_div_p_; i++)	{ count_rank[i] = 0; }

	MPI_Barrier(MPI_COMM_WORLD);
	Combine_Data_count (count_rank, count);

	int		count_rank_all = 0;
	int		count_rank_max = 0;
	for (int i=0; i<ncpu_div_p_; i++) {
		count_rank_all +=  count_rank[i];
		count_rank_max  = (count_rank_max > count_rank[i]) ? count_rank_max : count_rank[i];
	}

	int		count_all = 0;
	MPI_Allreduce(&count, &count_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	// 粒子数の数え上げ //


	MPI_Barrier(MPI_COMM_WORLD);


	// fname folder //
	char	fname_folder[256];
	if	(downsize == 1) { sprintf(fname_folder  , "result_particle_scatter_binary"); }
	else				{ sprintf(fname_folder  , "result_particle_scatter_binary_downsize"); }
	// fname folder //


	// num files //
	if (mpi_rank_ == 0) {
		char	fname_num_files  [256];
		sprintf(fname_num_files  , "./result_particle/num_files.dat");

		const int	num_files = ncpu_/ncpu_div_p_;

		FILE	*fp_num_files = fopen(fname_num_files, "w");
		fwrite(&num_files, sizeof(int), 1, fp_num_files);
		fclose(fp_num_files  );
	}
	// num files //


	// header //
	if (mpi_rank_ == 0) {
		char	fname_header  [256];
		sprintf(fname_header  , "./%s/header-%d.dat"    , fname_folder,      t);

		FILE	*fp_header = fopen(fname_header  , "w");
		fwrite(&count_all, sizeof(int), 1, fp_header);
		fclose(fp_header  );
	}
	// header //


	// header_node //
	char	fname_header_node[256];
	sprintf(fname_header_node  , "./%s/header_node%d-%d.dat", fname_folder, lid, t);

	if (fid == 0) {
		FILE	*fp_header_node = fopen(fname_header_node  , "w");
		fwrite(&count_rank_all, sizeof(int), 1, fp_header_node);
		fclose(fp_header_node);
	}
	// header_node //


	// index //
	char	fname_index[256];
	sprintf(fname_index, "./%s/index%d-%d.dat", fname_folder, lid, t);

	// 構造体の粒子の属性を配列へコピー
	int		*index    = new int  [count];
	copy_particle_source_index (
		index,
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// index rank 
	int		*index_rank = new int[count_rank_all];
	Combine_Data_int    (index_rank,    index,    count_rank, count, count_rank_max);
	if (fid == 0)	{	fileLib::write_file (index_rank,	fname_index,	count_rank_all);	}

	delete [] index_rank;
	delete [] index;
	// index //


	// velocity //
	char	fname_velocity[256];
	sprintf(fname_velocity, "./%s/velocity%d-%d.dat", fname_folder, lid, t);

	// 構造体の粒子の速度を配列へコピー
	float	*vel      = new float[count];
	copy_particle_velocity (
		vel,
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// vel rank
	float	*vel_rank      = new float[count_rank_all];
	Combine_Data_FLOAT  (vel_rank, vel, count_rank, count, count_rank_max);
	if (fid == 0)	{ fileLib::write_file (vel_rank,	fname_velocity,	count_rank_all); }

	delete [] vel_rank;
	delete [] vel;
	// velocity //


	// position //
	char	fname_position[256];
	sprintf(fname_position, "./%s/position%d-%d.dat", fname_folder, lid, t);

	// 構造体の粒子座標を配列へコピー
	float	*position = new float[count*3];
	copy_particle_position (
		position, 
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// position rank
	float	*position_rank = new float[count_rank_all*3];
	Combine_Data_FLOAT3 (position_rank, position, count_rank, count, count_rank_max);
	if (fid == 0)	{	fileLib::write_file (position_rank,	fname_position,	count_rank_all*3);	}

	delete [] position_rank;
	delete [] position;
	// position //


	// uvw //
	char	fname_uvw[256];
	sprintf(fname_uvw     , "./%s/uvw%d-%d.dat"     , fname_folder, lid, t);

	const FLOAT	*u   = cbq->u_n;
	const FLOAT	*v   = cbq->v_n;
	const FLOAT	*w   = cbq->w_n;
	FLOAT	*uvw = new FLOAT[count*3];

	int		id;
	int		frg_count;
	for (int i=0; i<num_particle; i++) {
		if (i == 0)	{ id = 0; }
		if (i == 0)	{ frg_count = 0; }

		if (ppos[i].state_p == PARTICLE_CAL) {
			if (frg_count%downsize == 0) {
				const FLOAT	x = ppos[i].x_p;
				const FLOAT	y = ppos[i].y_p;
				const FLOAT	z = ppos[i].z_p;

				const int	nx = domain->nx;	const FLOAT	dx = domain->dx;
				const int	ny = domain->ny;	const FLOAT	dy = domain->dx;
				const int	nz = domain->nz;	const FLOAT	dz = domain->dx;

				const FLOAT	u_tmp = Interpolate_Particle_Velocity (u, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);
				const FLOAT	v_tmp = Interpolate_Particle_Velocity (v, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);
				const FLOAT	w_tmp = Interpolate_Particle_Velocity (w, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);

				uvw[id*3  ] = u_tmp;
				uvw[id*3+1] = v_tmp;
				uvw[id*3+2] = w_tmp;

				id++;
			}
			frg_count++;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	float	*uvw_rank = new float[count_rank_all*3];
	Combine_Data_FLOAT3 (uvw_rank,  uvw,      count_rank, count, count_rank_max);
	if (fid == 0)	{ fileLib::write_file (uvw_rank, fname_uvw, count_rank_all*3); }

	delete [] uvw_rank;
	delete [] uvw;

	// uvw //

}

// (YOKOUCHI 2020)
void	ParaviewParticle::
output_particle_binary_scatter_all_LSM (
	int 	t,
	const Domain			*const domain,
	int		num_particle,
	const ParticlePosition	*const ppos,
	const BasisVariables	*const cbq,
	const Stress			*const str,
	int		downsize
	)
{
	const int	rank   = mpi_rank_;
	const int	fid    = rank%ncpu_div_p_;
	const int	lid    = rank/ncpu_div_p_;


	// 粒子数の数え上げ //
	int		count = count_particles (ppos, num_particle, downsize);

	int		count_rank[ncpu_div_p_];
	for (int i=0; i<ncpu_div_p_; i++)	{ count_rank[i] = 0; }

	MPI_Barrier(MPI_COMM_WORLD);
	Combine_Data_count (count_rank, count);

	int		count_rank_all = 0;
	int		count_rank_max = 0;
	for (int i=0; i<ncpu_div_p_; i++) {
		count_rank_all +=  count_rank[i];
		count_rank_max  = (count_rank_max > count_rank[i]) ? count_rank_max : count_rank[i];
	}

	int		count_all = 0;
	MPI_Allreduce(&count, &count_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	// 粒子数の数え上げ //


	MPI_Barrier(MPI_COMM_WORLD);


	// fname folder //
	char	fname_folder[256];
	if	(downsize == 1) { sprintf(fname_folder  , "result_particle_scatter_binary"); }
	else				{ sprintf(fname_folder  , "result_particle_scatter_binary_downsize"); }
	// fname folder //


	// num files //
	if (mpi_rank_ == 0) {
		char	fname_num_files  [256];
		sprintf(fname_num_files  , "./result_particle/num_files.bin");

		const int	num_files = ncpu_/ncpu_div_p_;

		FILE	*fp_num_files = fopen(fname_num_files, "w");
		fwrite(&num_files, sizeof(int), 1, fp_num_files);
		fclose(fp_num_files  );
	}
	// num files //


	// header //
	if (mpi_rank_ == 0) {
		char	fname_header  [256];
		sprintf(fname_header  , "./%s/header-%d.bin"    , fname_folder,      t);

		FILE	*fp_header = fopen(fname_header  , "w");
		fwrite(&count_all, sizeof(int), 1, fp_header);
		fclose(fp_header  );
	}
	// header //


	// header_node //
	char	fname_header_node[256];
	sprintf(fname_header_node  , "./%s/header_node%d-%d.bin", fname_folder, lid, t);

	if (fid == 0) {
		FILE	*fp_header_node = fopen(fname_header_node  , "w");
		fwrite(&count_rank_all, sizeof(int), 1, fp_header_node);
		fclose(fp_header_node);
	}
	// header_node //


	// index //
	char	fname_index[256];
	sprintf(fname_index, "./%s/index%d-%d.bin", fname_folder, lid, t);

	// 構造体の粒子の属性を配列へコピー
	int		*index    = new int  [count];
	copy_particle_source_index (
		index,
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// index rank 
	int		*index_rank = new int[count_rank_all];
	Combine_Data_int    (index_rank,    index,    count_rank, count, count_rank_max);
	if (fid == 0)	{	fileLib::write_file (index_rank,	fname_index,	count_rank_all);	}

	delete [] index_rank;
	delete [] index;
	// index //


	// velocity //
	char	fname_velocity[256];
	sprintf(fname_velocity, "./%s/velocity%d-%d.bin", fname_folder, lid, t);

	// 構造体の粒子の速度を配列へコピー
	float	*vel      = new float[count];
	copy_particle_velocity (
		vel,
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// vel rank
	float	*vel_rank      = new float[count_rank_all];
	Combine_Data_FLOAT  (vel_rank, vel, count_rank, count, count_rank_max);
	if (fid == 0)	{ fileLib::write_file (vel_rank,	fname_velocity,	count_rank_all); }

	delete [] vel_rank;
	delete [] vel;
	// velocity //


	// position //
	char	fname_position[256];
	sprintf(fname_position, "./%s/position%d-%d.bin", fname_folder, lid, t);

	// 構造体の粒子座標を配列へコピー
	float	*position = new float[count*3];
	copy_particle_position (
		position, 
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// position rank
	float	*position_rank = new float[count_rank_all*3];
	Combine_Data_FLOAT3 (position_rank, position, count_rank, count, count_rank_max);
	if (fid == 0)	{	fileLib::write_file (position_rank,	fname_position,	count_rank_all*3);	}

	delete [] position_rank;
	delete [] position;
	// position //


	// uvw //
	char	fname_uvw[256];
	sprintf(fname_uvw     , "./%s/uvw%d-%d.bin"     , fname_folder, lid, t);

	const FLOAT	*u   = cbq->u_n;
	const FLOAT	*v   = cbq->v_n;
	const FLOAT	*w   = cbq->w_n;
	FLOAT	*uvw = new FLOAT[count*3];

	int		id;
	int		frg_count;
	for (int i=0; i<num_particle; i++) {
		if (i == 0)	{ id = 0; }
		if (i == 0)	{ frg_count = 0; }

		if (ppos[i].state_p == PARTICLE_CAL) {
			if (frg_count%downsize == 0) {
				const FLOAT	x = ppos[i].x_p;
				const FLOAT	y = ppos[i].y_p;
				const FLOAT	z = ppos[i].z_p;

				const FLOAT	u_sgs = ppos[i].u_sgs;
				const FLOAT	v_sgs = ppos[i].v_sgs;
				const FLOAT	w_sgs = ppos[i].w_sgs;

				const int	nx = domain->nx;	const FLOAT	dx = domain->dx;
				const int	ny = domain->ny;	const FLOAT	dy = domain->dx;
				const int	nz = domain->nz;	const FLOAT	dz = domain->dx;

				const FLOAT	u_tmp = Interpolate_Particle_Velocity (u, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);
				const FLOAT	v_tmp = Interpolate_Particle_Velocity (v, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);
				const FLOAT	w_tmp = Interpolate_Particle_Velocity (w, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);

				// (YOKOUCHI 2020)
				if (user_flags::flg_particle == 1 || user_flags::flg_particle == 3) {
					uvw[id*3  ] = u_tmp + u_sgs;
					uvw[id*3+1] = v_tmp + v_sgs;
					uvw[id*3+2] = w_tmp + w_sgs;
				} else {
					uvw[id*3  ] = u_tmp;
					uvw[id*3+1] = v_tmp;
					uvw[id*3+2] = w_tmp;
				}

				id++;
			}
			frg_count++;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	float	*uvw_rank = new float[count_rank_all*3];
	Combine_Data_FLOAT3 (uvw_rank,  uvw,      count_rank, count, count_rank_max);
	if (fid == 0)	{ fileLib::write_file (uvw_rank, fname_uvw, count_rank_all*3); }

	delete [] uvw_rank;
	delete [] uvw;

	// uvw //

	// (YOKOUCHI 2020)
	if (user_flags::flg_particle == 1 || user_flags::flg_particle == 2 || user_flags::flg_particle == 3) {
	// uvw_sgs //
	char	fname_uvw_sgs[256];
	sprintf(fname_uvw_sgs, "./%s/uvw_sgs%d-%d.bin", fname_folder, lid, t);

	// 構造体のSGS粒子速度を配列へコピー
	float	*uvw_sgs = new float[count*3];
	copy_particle_uvw_sgs (
		uvw_sgs, 
		ppos, num_particle, downsize);

	MPI_Barrier(MPI_COMM_WORLD);

	// uvw_sgs rank
	float	*uvw_sgs_rank = new float[count_rank_all*3];
	Combine_Data_FLOAT3 (uvw_sgs_rank, uvw_sgs, count_rank, count, count_rank_max);
	if (fid == 0)	{	fileLib::write_file (uvw_sgs_rank,	fname_uvw_sgs,	count_rank_all*3);	}

	delete [] uvw_sgs_rank;
	delete [] uvw_sgs;
	// uvw_sgs //
	}

}

void	ParaviewParticle::
Output_Particle_binary_scatter_All (
	int 	t,
	const Domain			*const domain,
	int		num_particle,
	const ParticlePosition	*const ppos,
	const BasisVariables	*const cbq,
	const Stress			*const str,
	int		downsize
	)
{
	// 粒子数の数え上げ
	int		count = count_particles (ppos, num_particle, downsize);


	float	*position = new float[count*3];
	float	*vel      = new float[count];
	int		*index    = new int  [count];


	// 構造体の粒子の属性を配列へコピー
	copy_particle_source_index (
		index,
		ppos, num_particle, downsize);


	// 構造体の粒子の速度を配列へコピー
	copy_particle_velocity (
		vel,
		ppos, num_particle, downsize);


	// 構造体の粒子座標を配列へコピー
	copy_particle_position (
		position, 
		ppos, num_particle, downsize);


	// vector, turbulent statistics *****
	FLOAT	*u = cbq->u_n;
	FLOAT	*v = cbq->v_n;
	FLOAT	*w = cbq->w_n;
	FLOAT	*uvw = new FLOAT[count*3];


	int		id;
	int		frg_count;
	for (int i=0; i<num_particle; i++) {
		if (i == 0)	{ id = 0; }
		if (i == 0)	{ frg_count = 0; }

		if (ppos[i].state_p == PARTICLE_CAL) {
			if (frg_count%downsize == 0) {
				const FLOAT	x = ppos[i].x_p;
				const FLOAT	y = ppos[i].y_p;
				const FLOAT	z = ppos[i].z_p;

				const int	nx = domain->nx;	const FLOAT	dx = domain->dx;
				const int	ny = domain->ny;	const FLOAT	dy = domain->dx;
				const int	nz = domain->nz;	const FLOAT	dz = domain->dx;

				const FLOAT	u_tmp = Interpolate_Particle_Velocity (u, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);
				const FLOAT	v_tmp = Interpolate_Particle_Velocity (v, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);
				const FLOAT	w_tmp = Interpolate_Particle_Velocity (w, x, y, z, domain->x_min, domain->y_min, domain->z_min, dx, dy, dz, nx, ny, nz, halo_);

				uvw[id*3  ] = u_tmp;
				uvw[id*3+1] = v_tmp;
				uvw[id*3+2] = w_tmp;

				id++;
			}
			frg_count++;
		}
	}


	// data 
	const int	rank   = mpi_rank_;
	const int	fid    = rank%ncpu_div_p_;
	const int	lid    = rank/ncpu_div_p_;

	// count
	int		count_rank[ncpu_div_p_];
	for (int i=0; i<ncpu_div_p_; i++)	{ count_rank[i] = 0; }

	MPI_Barrier(MPI_COMM_WORLD);
	Combine_Data_count (count_rank, count);


	int		count_rank_all = 0;
	int		count_rank_max = 0;
	for (int i=0; i<ncpu_div_p_; i++) {
		count_rank_all +=  count_rank[i];
		count_rank_max  = (count_rank_max > count_rank[i]) ? count_rank_max : count_rank[i];
	}


	float	*position_rank = new float[count_rank_all*3];
	float	*vel_rank      = new float[count_rank_all];
	int		*index_rank    = new int  [count_rank_all];

	MPI_Barrier(MPI_COMM_WORLD);
	Combine_Data_FLOAT3 (position_rank, position, count_rank, count, count_rank_max);
	Combine_Data_int    (index_rank,    index,    count_rank, count, count_rank_max);
	Combine_Data_FLOAT  (vel_rank,      vel,      count_rank, count, count_rank_max);


	// vector, turbulent statistics *****
	float	*uvw_rank = new float[count_rank_all*3];

	MPI_Barrier(MPI_COMM_WORLD);
	Combine_Data_FLOAT3 (uvw_rank,  uvw,      count_rank, count, count_rank_max);


	// count_all
	int		count_all = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&count, &count_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	// file
	char	fname_position[256];
	char	fname_velocity[256];
	char	fname_index   [256];
	char	fname_uvw[256];
	char	fname_header_node[256];


	char	fname_folder[256];
	if	(downsize == 1) { sprintf(fname_folder  , "result_particle_scatter_binary"); }
	else				{ sprintf(fname_folder  , "result_particle_scatter_binary_downsize"); }


	sprintf(fname_position, "./%s/position%d-%d.dat", fname_folder, lid, t);
	sprintf(fname_velocity, "./%s/velocity%d-%d.dat", fname_folder, lid, t);
	sprintf(fname_index   , "./%s/index%d-%d.dat"   , fname_folder, lid, t);

	sprintf(fname_uvw     , "./%s/uvw%d-%d.dat"     , fname_folder, lid, t);

	sprintf(fname_header_node  , "./%s/header_node%d-%d.dat", fname_folder, lid, t);


	if (mpi_rank_ == 0) {
		// num files
		char	fname_num_files  [256];
		sprintf(fname_num_files  , "./result_particle/num_files.dat");

		const int	num_files = ncpu_/ncpu_div_p_;

		FILE	*fp_num_files = fopen(fname_num_files, "w");
		fwrite(&num_files, sizeof(int), 1, fp_num_files);
		fclose(fp_num_files  );


		// header
		char	fname_header  [256];
		sprintf(fname_header  , "./%s/header-%d.dat"    , fname_folder,      t);

		FILE	*fp_header = fopen(fname_header  , "w");
		fwrite(&count_all, sizeof(int), 1, fp_header);
		fclose(fp_header  );

	}

	if (fid == 0) {
		// header_node
		FILE	*fp_header_node = fopen(fname_header_node  , "w");
		fwrite(&count_rank_all, sizeof(int), 1, fp_header_node);
		fclose(fp_header_node);


		// position
		fileLib::write_file (position_rank,	fname_position,	count_rank_all*3);

		// velocity
		fileLib::write_file (vel_rank,		fname_velocity,	count_rank_all);

		// index
		fileLib::write_file (index_rank,	fname_index,	count_rank_all);

		// uvw
		fileLib::write_file (uvw_rank,		fname_uvw,		count_rank_all*3);
	}


	// delete
	delete [] position_rank;
	delete [] vel_rank;
	delete [] index_rank;

	delete [] uvw_rank;

	// delete
	delete [] position;
	delete [] vel;
	delete [] index;

	// turbulent statistics
	delete [] uvw;
}


// private //
// 粒子数の数え上げ
int		ParaviewParticle::
count_particles (
	const ParticlePosition *ppos,
	int		num_particle,
	int		downsize)
{
	int		count     = 0;
	int		tmp_count = 0;

	for (int i=0; i<num_particle; i++) {
		if (ppos[i].state_p == PARTICLE_CAL) {
			if (tmp_count%downsize == 0) {
				count++;
			}
			tmp_count++;
		}
	}

	return	count;
}


// 構造体の粒子の属性を配列へコピー
void	ParaviewParticle::
copy_particle_source_index (
	int		*f_int,
	const ParticlePosition	*ppos,
	int		num_particle,
	int		downsize)
{
	int		index     = 0;
	int		tmp_count = 0;

	for (int i=0; i<num_particle; i++) {
		if (ppos[i].state_p == PARTICLE_CAL) {
			if (tmp_count%downsize == 0) {
				f_int[index] = ppos[i].source_index_p;

				index++;
			}
			tmp_count++;
		}
	}
}


// 構造体の粒子の速度を配列へコピー
void	ParaviewParticle::
copy_particle_velocity (
	float	*velocity,
	const ParticlePosition	*const ppos,
	int		num_particle,
	int		downsize)
{
	int		index     = 0;
	int		tmp_count = 0;

	for (int i=0; i<num_particle; i++) {
		if (ppos[i].state_p == PARTICLE_CAL) {
			if (tmp_count%downsize == 0) {
				velocity[index] = ppos[i].vel_p * c_ref_;

				index++;
			}
			tmp_count++;
		}
	}
}


// 構造体の粒子座標を配列へコピー
void	ParaviewParticle::
copy_particle_position (
	float	*position,
	const ParticlePosition	*const ppos,
	int		num_particle,
	int		downsize)
{
	int		index     = 0;
	int		tmp_count = 0;

	for (int i=0; i<num_particle; i++) {
		if (ppos[i].state_p == PARTICLE_CAL) {
			if (tmp_count%downsize == 0) {
				position[index*3  ]   = (float)ppos[i].x_p;
				position[index*3+1]   = (float)ppos[i].y_p;
				position[index*3+2]   = (float)ppos[i].z_p;

//				position[index*3  ]   = (float)ppos[i].x_p * cal_scale + coefficient::x_offset;
//				position[index*3+1]   = (float)ppos[i].y_p * cal_scale + coefficient::y_offset;
//				position[index*3+2]   = (float)ppos[i].z_p * cal_scale + coefficient::z_offset;

				index++;
			}
			tmp_count++;
		}
	}
}

// 構造体のSGS粒子速度を配列へコピー
void	ParaviewParticle::
copy_particle_uvw_sgs (
	float	*uvw_sgs,
	const ParticlePosition	*const ppos,
	int		num_particle,
	int		downsize)
{
	int		index     = 0;
	int		tmp_count = 0;

	for (int i=0; i<num_particle; i++) {
		if (ppos[i].state_p == PARTICLE_CAL) {
			if (tmp_count%downsize == 0) {
				uvw_sgs[index*3  ]   = (float)ppos[i].u_sgs * c_ref_;
				uvw_sgs[index*3+1]   = (float)ppos[i].v_sgs * c_ref_;
				uvw_sgs[index*3+2]   = (float)ppos[i].w_sgs * c_ref_;
			
				index++;
			}
			tmp_count++;
		}
	}
}

// 複数ファイルの結合
void	ParaviewParticle::
Combine_Data_count (
	int *count_out, 
	int count_in)
{
	MPI_Status	stat;

	const int	rank = mpi_rank_;
	const int	fid  = rank%ncpu_div_p_;
	const int	lid  = rank/ncpu_div_p_;

	if (fid == 0) {
		int		count_tmp;
		for (int i=0; i<ncpu_div_p_; i++) {
			const int	id_rank = rank + i;

			if (i == 0) {
				count_tmp = count_in;
			}
			else {
				MPI_Recv(&count_tmp, 1, MPI_INT, id_rank, 0, MPI_COMM_WORLD, &stat);
			}

			count_out[i] = count_tmp;
		}
	}
	else {
		const int	rank_fid0 = lid*ncpu_div_p_;

		MPI_Send(&count_in, 1, MPI_INT,  rank_fid0, 0, MPI_COMM_WORLD);
	}
}


void	ParaviewParticle::
Combine_Data_int (
	int *f_out,
	int *f_in,
	const int count_rank[],
	int count_in,
	int count_max)
{
	MPI_Status	stat;

	const int	rank = mpi_rank_;
	const int	fid  = rank%ncpu_div_p_;

	if (fid == 0) {
		int		*f_tmp = new int[count_max+1];

		int		index = 0;
		for (int i=0; i<ncpu_div_p_; i++) {
			const int	id_rank = rank + i;

			if (i == 0) {
				memcpy(f_tmp, f_in,   sizeof(int)*(count_rank[i]));
			}
			else {
				MPI_Recv(f_tmp, count_rank[i], MPI_INT, id_rank, 0, MPI_COMM_WORLD, &stat);
			}

			// write
			memcpy(&f_out[index], f_tmp,   sizeof(int)*(count_rank[i]));
			index += count_rank[i];
		}
		delete [] f_tmp;
	}
	else {
		const int	lid  = rank/ncpu_div_p_;
		const int	rank_fid0 = lid*ncpu_div_p_;

		MPI_Send(f_in, count_in, MPI_INT,  rank_fid0, 0, MPI_COMM_WORLD);
	}
}


void	ParaviewParticle::
Combine_Data_FLOAT(
	FLOAT *f_out,
	FLOAT *f_in,
	const int count_rank[],
	int count_in,
	int count_max)
{
	MPI_Status	stat;

	const int	rank = mpi_rank_;
	const int	fid  = rank%ncpu_div_p_;

	if (fid == 0) {
		FLOAT	*f_tmp = new FLOAT[count_max+1];

		int		index = 0;
		for (int i=0; i<ncpu_div_p_; i++) {
			const int	id_rank = rank + i;

			if (i == 0) {
				memcpy(f_tmp, f_in,   sizeof(FLOAT)*(count_rank[i]));
			}
			else {
				MPI_Recv(f_tmp, count_rank[i], MFLOAT, id_rank, 0, MPI_COMM_WORLD, &stat);
			}

			// write
			memcpy(&f_out[index], f_tmp,   sizeof(FLOAT)*(count_rank[i]));
			index += count_rank[i];
		}
		delete [] f_tmp;
	}
	else {
		const int	lid  = rank/ncpu_div_p_;
		const int	rank_fid0 = lid*ncpu_div_p_;

		MPI_Send(f_in, count_in, MFLOAT,  rank_fid0, 0, MPI_COMM_WORLD);
	}
}


void	ParaviewParticle::
Combine_Data_FLOAT3 (
	FLOAT *f_out,
	FLOAT *f_in,
	const int count_rank[],
	int count_in,
	int count_max)
{
	MPI_Status	stat;

	const int	rank = mpi_rank_;
	const int	fid  = rank%ncpu_div_p_;

	if (fid == 0) {
		FLOAT	*f_tmp = new FLOAT[count_max*3+1];

		int		index = 0;
		for (int i=0; i<ncpu_div_p_; i++) {
			const int	id_rank = rank + i;

			if (i == 0) {
				memcpy(f_tmp, f_in,   sizeof(FLOAT)*(count_rank[i]*3));
			}
			else {
				MPI_Recv(f_tmp, count_rank[i]*3, MFLOAT, id_rank, 0, MPI_COMM_WORLD, &stat);
			}

			// write
			memcpy(&f_out[index], f_tmp,   sizeof(FLOAT)*(count_rank[i]*3));
			index += count_rank[i]*3;
		}
		delete [] f_tmp;
	}
	else {
		const int	lid  = rank/ncpu_div_p_;
		const int	rank_fid0 = lid*ncpu_div_p_;

		MPI_Send(f_in, count_in*3, MFLOAT,  rank_fid0, 0, MPI_COMM_WORLD);
	}
}


// Paraview.cu
