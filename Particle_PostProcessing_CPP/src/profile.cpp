#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <stdlib.h>

#include "profile.h"
#include "stSetting.h"
#include "stParticle.h"

/* ---------- Class ParticleProfile --------- */
void ParticleProfile::allocate_profile (Setting& setting) {
	// Copy setting
	const	int x_n_rank = setting.X_NUM_RANK;
	const int y_n_rank = setting.Y_NUM_RANK;
	const int x_rank	 = setting.X_RANK;
	const int y_rank	 = setting.Y_RANK;
	const	int	z_domain = setting.Z_DOMAIN;

	// Allocate
	std::cout << "Allocate profile memory -----" << std::endl;
	profile 		= new int[y_n_rank*x_n_rank*z_domain];
	std::cout << "Allocate map memory -----" << std::endl;
	fluid_grid	= new int[y_n_rank*x_n_rank*z_domain];

	// Initilization
	for (int i=0; i<y_n_rank; i++) {
		for (int j=0; j<x_n_rank; j++) {
			for (int k=0; k<z_domain; k++) {
				profile[k+j*z_domain+i*z_domain*x_n_rank]		= 0;
				fluid_grid[k+j*z_domain+i*z_domain*x_n_rank] = x_rank*y_rank;
			}
		}
	}
}

void ParticleProfile::read_map (Setting& setting) {
	// Copy setting
	const int 	x_n_rank	= setting.X_NUM_RANK;
	const int 	y_n_rank	= setting.Y_NUM_RANK;
	const int 	x_rank		= setting.X_RANK;
	const int 	y_rank		= setting.Y_RANK;
	const int 	x_domain 	= setting.X_DOMAIN;
	const int 	y_domain 	= setting.Y_DOMAIN;
	const int 	z_domain	= setting.Z_DOMAIN;
	const float	dx				= setting.dX;
	const char*	fname_map	= setting.FNAME_MAP;

	// Use std::vector instead of VLA
	std::vector<float> height_map(x_domain * y_domain);
	float max_height = 0.0;
	float sum_height = 0.0;
	int 	count_buil = 0;

	// Read
	std::cout << "Read map. File name >> " << fname_map << std::endl;
	
	std::fstream	mFile;
	mFile.open(fname_map, std::ios::in);
	if(!mFile.is_open()) {
		std::cout << "cannot open map file" << std::endl;
		exit(EXIT_FAILURE);
	}
	int x, y;
	mFile >> x >> y; // Fixed: combined the two operations
	if (x_domain > x || y_domain > y) {
		std::cout << "size of map file is too small" << std::endl;
	}
	for (int i=0; i<y_domain; i++) {
		for (int j=0; j<x_domain; j++) {
			mFile >> height_map[j+i*x_domain];
			sum_height += height_map[j+i*x_domain];
			if (height_map[j+i*x_domain] > 0.0) count_buil++;
			if (max_height < height_map[j+i*x_domain]) max_height = height_map[j+i*x_domain];
		}
	}
	mFile.close();

	// Output map status
	std::cout << " -max height = " << max_height << std::endl;
	std::cout << " -ave height = " << sum_height / count_buil << std::endl;
	std::cout << std::endl;

	// Count fluid grids
	int i, j, ii, jj, k;
//	#pragma omp parallel for private(ii,jj,k)
	for (i=0; i<y_n_rank; i++) {
		for (j=0; j<x_n_rank; j++) {
			for (ii=0; ii<y_rank; ii++) {
				for (jj=0; jj<x_rank; jj++) {
					float height 	= height_map[jj + j*x_rank + (ii + i*y_rank)*x_domain]; // Fixed indexing
					int		h_solid	= (int)ceil(height/dx);
					for (k=0; k<h_solid; k++) fluid_grid[k+j*z_domain+i*z_domain*x_n_rank]--;
				}
			}
		}
	}
}

void ParticleProfile::cal_profile (float x, float y, float z, Setting& setting) {
	// Copy setting
	const int x_n_rank 	= setting.X_NUM_RANK;
	const int y_n_rank 	= setting.Y_NUM_RANK;
	const int x_rank	 	= setting.X_RANK;
	const int y_rank	 	= setting.Y_RANK;
	const int z_domain 	= setting.Z_DOMAIN;
	const float dx		 	= setting.dX;
	
	// cal
	if (-0.5<x && x<0.0) x = 0.0;
	if (-0.5<y && y<0.0) y = 0.0;
	if (-0.5<z && z<0.0) z = 0.0;

	const int p_x_rank	= (int)x / (int)((float)x_rank * dx);
	const int p_y_rank	= (int)y / (int)((float)y_rank * dx);
	const	int p_z				= (int)z / (int)dx;

	if (p_x_rank < 0 || x_n_rank-1 < p_x_rank) {
		std::cout << "Error!! (out of domain at profile)\n";
		std::cout << "x = " << x << " : x_rank = " << p_x_rank << std::endl;
	}
	if (p_y_rank < 0 || y_n_rank-1 < p_y_rank) {
		std::cout << "Error!! (out of domain at profile)\n";
		std::cout << "y = " << y << " : y_rank = " << p_y_rank << std::endl;
	}
	if (p_z			 < 0 || z_domain-1 < p_z)	{
		std::cout << "Error!! (out of domain at profile)\n";
		std::cout << "z = " << z << std::endl;
	}

	profile[p_z + p_x_rank*z_domain + p_y_rank*z_domain*x_n_rank]++;
}



void ParticleProfile::output_profile (Setting& setting) {
	// Copy setting
	const int		x_n_rank 	= setting.X_NUM_RANK;
	const int 	y_n_rank 	= setting.Y_NUM_RANK;
	const int 	z_domain 	= setting.Z_DOMAIN;
	const char*	dir_out		= setting.DIR_OUT; 

	// Output
	std::cout << "Output profile >>>>> ";
	std::fstream 	pFile;
	char					profFile[100];

	sprintf(profFile, "./%s/profile.csv", dir_out);
	pFile.open(profFile, std::ios::out);
	for (int i=0; i<y_n_rank; i++) {
		for (int j=0; j<x_n_rank; j++) {
			pFile << "Rank X:: " << j << " : Rank Y:: " << i << std::endl;
			for (int k=0; k<z_domain; k++) {
				pFile << profile[k + j*z_domain + i*z_domain*x_n_rank];
				if (k<z_domain-1) pFile << ",";
			}
			pFile << std::endl;
			for (int k=0; k<z_domain; k++) {
				pFile << fluid_grid[k + j*z_domain + i*z_domain*x_n_rank];
				if (k<z_domain-1) pFile << ",";
			}
			pFile << std::endl;
		}
	}
	pFile.close();
	std::cout << "Finish!!" << std::endl;
}

void ParticleProfile::delete_profile () {
	delete[] profile;
	delete[] fluid_grid;
}
	

