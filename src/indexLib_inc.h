#include "indexLib.h"


namespace	
indexLib {

// 3次元のindexとweight
inline __host__ __device__ void
get_cellId_weight_3D (
	int		id_cell[],
	FLOAT	weight_cell[], 
	const int	index[], 
	const int	grid[],
	const FLOAT vec3[],
	FLOAT		dx
	)
{
	int		id_off_x0, id_off_xn;
	int		id_off_y0, id_off_yn;
	int		id_off_z0, id_off_zn;

	get_offset_cellId_weight_1D (&id_off_x0, &id_off_xn, &weight_cell[0], vec3[0], dx);
	get_offset_cellId_weight_1D (&id_off_y0, &id_off_yn, &weight_cell[1], vec3[1], dx);
	get_offset_cellId_weight_1D (&id_off_z0, &id_off_zn, &weight_cell[2], vec3[2], dx);


	id_cell[0] = get_index (index, grid, id_off_x0, id_off_y0, id_off_z0);
	id_cell[1] = get_index (index, grid, id_off_xn, id_off_y0, id_off_z0);
	id_cell[2] = get_index (index, grid, id_off_x0, id_off_yn, id_off_z0);
	id_cell[3] = get_index (index, grid, id_off_xn, id_off_yn, id_off_z0);
	id_cell[4] = get_index (index, grid, id_off_x0, id_off_y0, id_off_zn);
	id_cell[5] = get_index (index, grid, id_off_xn, id_off_y0, id_off_zn);
	id_cell[6] = get_index (index, grid, id_off_x0, id_off_yn, id_off_zn);
	id_cell[7] = get_index (index, grid, id_off_xn, id_off_yn, id_off_zn);
}


// 1次元方向のindex(近遠)のoffsetとweight
inline __host__ __device__ void
get_offset_cellId_weight_1D (
	int		*id_l,
	int		*id_r,	// id_l : 近い方のindex, id_r : 遠い方のindex
	FLOAT	*weight_l,
	FLOAT	vec,
	FLOAT	dx
	)
{
	const int	sgn          = (vec >= 0.0) ? 1 : -1;
	const int	index_offset = sgn * (int)(fabs(vec)/dx);

	*id_l = index_offset;
	*id_r = index_offset + sgn;

	*weight_l = (fabs(vec) - fabs(index_offset*dx)) / dx;
}


} // namespace


// indexLib_inc.h //
