#ifndef INDEXLIB_H_
#define INDEXLIB_H_


#include "definePrecision.h"


namespace	
indexLib {

	// template //
	inline __host__ __device__
	int
	get_index (
		const int index[], 
		const int grid[],
		int x_offset, int y_offset, int z_offset)
	{
		const int	nx = grid[0],
					ny = grid[1];
	
		const int	id_x = index[0] + x_offset,
			  		id_y = index[1] + y_offset,
					id_z = index[2] + z_offset;
	
		return	id_x + nx*id_y + nx*ny*id_z;
	}


	inline __host__ __device__
	int
	get_index (
		const int index[], 
		const int grid[]
		)
	{
		const int	nx = grid[0],
					ny = grid[1];
	
		const int	id_x = index[0],
			  		id_y = index[1],
					id_z = index[2];
	
		return	id_x + nx*id_y + nx*ny*id_z;
	}


	inline __host__ __device__
	int
	get_index (
		int id_x, int id_y, int id_z,
		const int grid[])
	{
		const int	nx = grid[0],
					ny = grid[1];

		return	id_x + nx*id_y + nx*ny*id_z;
	}
	
	
	inline __host__ __device__
	int
	get_index (
		int id_x, int id_y, int id_z,
		int nx,   int ny,   int nz)
	{
		return	id_x + nx*id_y + nx*ny*id_z;
	}
	
	
	// function
	inline __host__ __device__
	int
	get_index_x (int id, int nx, int ny, int nz)
	{
		return	(id % (nx*ny)) % nx;
	}


	inline __host__ __device__
	int
	get_index_y (int id, int nx, int ny, int nz)
	{
		return	(id % (nx*ny)) / nx;
	}


	inline __host__ __device__
	int
	get_index_z (int id, int nx, int ny, int nz)
	{
		return	id / (nx*ny);
	}


	// indexLib_inc //
	inline __host__ __device__ void
	get_cellId_weight_3D (
		int		id_cell[],
		FLOAT	weight_cell[], 
		const int	index[], 
		const int	grid[],
		const FLOAT vec3[],
		FLOAT		dx
		);


	inline __host__ __device__ void
	get_offset_cellId_weight_1D (
		int		*id_l,
		int		*id_r,	// id_l : 近い方のindex, id_r : 遠い方のindex
		FLOAT	*weight_l,
		FLOAT	vec,
		FLOAT	dx
		);

}


#include "indexLib_inc.h"


#endif
