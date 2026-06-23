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

#include "stSetting.h"
#include "density.h"

void ParticleDensity::allocate_density (Setting& setting) {
	// Copy setting
	const int x_domain = setting.X_DOMAIN;
	const int y_domain = setting.Y_DOMAIN;
	const int z_domain = setting.Z_DOMAIN;
	const int n_slice_xy = setting.N_XY;
	const int n_slice_xz = setting.N_XZ;
	const int n_slice_yz = setting.N_YZ;
	
	// Allocate
	std::cout << "Allocate density memory -----" << std::endl;
	xy_number_density = new int[n_slice_xy*y_domain*x_domain];
	xz_number_density = new int[n_slice_xz*z_domain*x_domain];
	yz_number_density = new int[n_slice_yz*z_domain*y_domain];
	
	// Initialization
	for (int i=0; i<n_slice_xy; i++) {
		for (int j=0; j<y_domain; j++) {
			for (int k=0; k<x_domain; k++) {
				xy_number_density[k+j*x_domain+i*x_domain*y_domain] = 0;
			}
		}
	}
	for (int i=0; i<n_slice_xz; i++) {
		for (int j=0; j<z_domain; j++) {
			for (int k=0; k<x_domain; k++) {
				xz_number_density[k+j*x_domain+i*x_domain*z_domain] = 0;
			}
		}
	}
	for (int i=0; i<n_slice_yz; i++) {
		for (int j=0; j<z_domain; j++) {
			for (int k=0; k<y_domain; k++) {
				yz_number_density[k+j*y_domain+i*y_domain*z_domain] = 0;
			}
		}
	}	
}

void ParticleDensity::cal_density (float x, float y, float z, Setting& setting) {
	// Copy setting
	const int	x_domain  = setting.X_DOMAIN;
	const int y_domain	= setting.Y_DOMAIN;
	const int z_domain  = setting.Z_DOMAIN;
	const int n_slice_xy = setting.N_XY;
	const int n_slice_xz = setting.N_XZ;
	const int n_slice_yz = setting.N_YZ;
	const float dx 			= setting.dX;
	const float h_ave		= setting.H_AVE;
	
	// Use vectors instead of VLAs
	std::vector<float> z_out(n_slice_xy);
	std::vector<float> y_out(n_slice_xz);
	std::vector<float> x_out(n_slice_yz);
	
	std::copy(setting.Z_OUT, setting.Z_OUT+n_slice_xy, z_out.data());
	std::copy(setting.Y_OUT, setting.Y_OUT+n_slice_xz, y_out.data());
	std::copy(setting.X_OUT, setting.X_OUT+n_slice_yz, x_out.data());

	// cal
	// xy
	for (int i=0; i<n_slice_xy; i++) {
		float z_ctr = z_out[i];
		float z_min = z_ctr - h_ave / 2;
		float z_max = z_ctr + h_ave / 2;

		if (x < 0.0) x = 0.0;
		if (y < 0.0) y = 0.0;
		if (z < 0.0) z = 0.0;

		if (z_min < z && z < z_max) {
			int x_index = (int)(x/dx);
			int y_index = (int)(y/dx);
			xy_number_density[x_index+y_index*x_domain+i*y_domain*x_domain]++;
		}
	}
		
	for (int i=0; i<n_slice_xz; i++) {
		float y_ctr = y_out[i];
		float y_min = y_ctr - h_ave / 2;
		float y_max = y_ctr + h_ave / 2;

		if (x < 0.0) x = 0.0;
		if (y < 0.0) y = 0.0;
		if (z < 0.0) z = 0.0;

		if (y_min < y && y < y_max) {
			int x_index = (int)(x/dx);
			int z_index = (int)(z/dx);
			xz_number_density[x_index+z_index*x_domain+i*z_domain*x_domain]++;
		}
	}
	for (int i=0; i<n_slice_yz; i++) {
		float x_ctr = x_out[i];
		float x_min = x_ctr - h_ave / 2;
		float x_max = x_ctr + h_ave / 2;

		if (x < 0.0) x = 0.0;
		if (y < 0.0) y = 0.0;
		if (z < 0.0) z = 0.0;

		if (x_min < x && x < x_max) {
			int z_index = (int)(z/dx);
			int y_index = (int)(y/dx);
			yz_number_density[y_index+z_index*y_domain+i*z_domain*y_domain]++;
		}
	}
}

void ParticleDensity::output_density (Setting& setting) {
	// Copy setting
	const int x_domain 	= setting.X_DOMAIN;
	const int y_domain 	= setting.Y_DOMAIN;
	const int z_domain	= setting.Z_DOMAIN;
	const int n_slice_xy	= setting.N_XY;
	const int n_slice_xz  = setting.N_XZ;
	const int n_slice_yz  = setting.N_YZ;
	const char*	dir_out = setting.DIR_OUT;
	
	// Use vectors instead of VLAs
	std::vector<float> z_out(n_slice_xy);
	std::vector<float> y_out(n_slice_xz);
	std::vector<float> x_out(n_slice_yz);
	
	std::copy(setting.Z_OUT, setting.Z_OUT+n_slice_xy, z_out.data());
	std::copy(setting.Y_OUT, setting.Y_OUT+n_slice_xz, y_out.data());
	std::copy(setting.X_OUT, setting.X_OUT+n_slice_yz, x_out.data());

	// Output
	for (int i=0; i<n_slice_xy; i++) {
		float zi_out = z_out[i];

		std::cout << "Output density (" << zi_out << ") >>>>> ";
		std::fstream	dFile;
		char					densityFile[256];

		sprintf(densityFile, "./%s/xy_number_density_%dm.csv", dir_out, (int)zi_out);
		dFile.open(densityFile, std::ios::out);
		if (!dFile.is_open()) {
			std::cout << "cannot open the output file" << std::endl;
			std::cout << "File (density file)" << std::endl;
			exit(EXIT_FAILURE);
		}
		for (int j=0; j<y_domain; j++) {
			for (int k=0; k<x_domain; k++) {
				dFile << xy_number_density[k+j*x_domain+i*y_domain*x_domain];
				if (k<x_domain-1) dFile << ",";
			}
			dFile << std::endl;
		}
		dFile.close();
		std::cout << "Finish!!" << std::endl;
	}

	for (int i=0; i<n_slice_xz; i++) {
		float yi_out = y_out[i];

		std::cout << "Output density (" << yi_out << ") >>>>> ";
		std::fstream	dFile;
		char					densityFile[256];

		sprintf(densityFile, "./%s/xz_number_density_%dm.csv", dir_out, (int)yi_out);
		dFile.open(densityFile, std::ios::out);
		if (!dFile.is_open()) {
			std::cout << "cannot open the output file" << std::endl;
			std::cout << "File (density file)" << std::endl;
			exit(EXIT_FAILURE);
		}
		for (int j=0; j<z_domain; j++) {
			for (int k=0; k<x_domain; k++) {
				dFile << xz_number_density[k+j*x_domain+i*z_domain*x_domain];
				if (k<x_domain-1) dFile << ",";
			}
			dFile << std::endl;
		}
		dFile.close();
		std::cout << "Finish!!" << std::endl;
	}

	for (int i=0; i<n_slice_yz; i++) {
		float xi_out = x_out[i];

		std::cout << "Output density (" << xi_out << ") >>>>> ";
		std::fstream	dFile;
		char					densityFile[256];

		sprintf(densityFile, "./%s/yz_number_density_%dm.csv", dir_out, (int)xi_out);
		dFile.open(densityFile, std::ios::out);
		if (!dFile.is_open()) {
			std::cout << "cannot open the output file" << std::endl;
			std::cout << "File (density file)" << std::endl;
			exit(EXIT_FAILURE);
		}
		for (int j=0; j<z_domain; j++) {
			for (int k=0; k<y_domain; k++) {
				dFile << yz_number_density[k+j*y_domain+i*z_domain*y_domain];
				if (k<y_domain-1) dFile << ",";
			}
			dFile << std::endl;
		}
		dFile.close();
		std::cout << "Finish!!" << std::endl;
	}
}

void ParticleDensity::delete_density () {
	delete[] xy_number_density;
	delete[] xz_number_density;
	delete[] yz_number_density;
}