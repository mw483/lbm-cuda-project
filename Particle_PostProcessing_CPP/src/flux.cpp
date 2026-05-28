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

#include "flux.h"
#include "function.h"


void ParticleFlux::allocate_flux(Setting& setting) {
	// Copy setting
	const int x_domain = setting.X_DOMAIN;
	const int y_domain = setting.Y_DOMAIN;
	const int n_slice	 = setting.N_FLUX;

	// Aillocate
	std::cout << "Allocate flux memory -----" << std::endl;
	flux = new int[n_slice*y_domain*x_domain];

	num_p_flux = new int[n_slice];

	// Initialization
	for (int i=0; i<n_slice; i++) {
		for (int j=0; j<y_domain; j++) {
			for (int k=0; k<x_domain; k++) {
				flux[k+j*x_domain+i*y_domain*x_domain] = 0;
			}
		}
	}
}

void ParticleFlux::cal_flux (int t, std::vector<Particle_Index>& p_flux_odd, std::vector<Particle_Index>& p_flux_even, Setting& setting) {
	// Copy setting
	const int start		 = setting.FILE_START;
	const int x_domain = setting.X_DOMAIN;
	const int y_domain = setting.Y_DOMAIN;
	const int dx 			 = setting.dX;
	const int n_slice  = setting.N_FLUX;
	
	// Use vector instead of VLA
	std::vector<float> z_flux(n_slice);
	std::copy(setting.Z_FLUX, setting.Z_FLUX+n_slice, z_flux.data());

	// cal
	std::size_t n_odd 	= p_flux_odd.size();
	std::size_t n_even	= p_flux_even.size();
	int n_new;
	int n_old;
	if ((t-start)%2 == 1) {
			n_new = (int)n_odd;
			n_old = (int)n_even;
	} else {
			n_new = (int)n_even;
			n_old = (int)n_odd;
	}
	
	Particle_Index	p_new;
	Particle_Index	p_old;
	int jj = 0;

	for (int i=0; i<n_new; i++) {
		if ((t-start)%2 == 1) {
			// new
			p_new.pos_x = p_flux_odd[i].pos_x;
			p_new.pos_y = p_flux_odd[i].pos_y;
			p_new.pos_z = p_flux_odd[i].pos_z;
			p_new.index = p_flux_odd[i].index;
		} else {
			// new
			p_new.pos_x = p_flux_even[i].pos_x;
			p_new.pos_y = p_flux_even[i].pos_y;
			p_new.pos_z = p_flux_even[i].pos_z;
			p_new.index = p_flux_even[i].index;
		}
		for (int j=jj; j<n_old; j++) {
			if ((t-start)%2 == 1) {
				// old
				p_old.pos_x = p_flux_even[j].pos_x;
				p_old.pos_y = p_flux_even[j].pos_y;
				p_old.pos_z = p_flux_even[j].pos_z;
				p_old.index = p_flux_even[j].index;
			} else {
				// old
				p_old.pos_x = p_flux_odd[j].pos_x;
				p_old.pos_y = p_flux_odd[j].pos_y;
				p_old.pos_z = p_flux_odd[j].pos_z;
				p_old.index = p_flux_odd[j].index;
			}

			// check index
			if (p_new.index == p_old.index) {
				// cal
				for (int k=0; k<n_slice; k++) {
					float zk_flux = z_flux[k];
					if ((p_new.pos_z-zk_flux)*(p_old.pos_z-zk_flux) < 0) {
						float dz_new;
						float dz_old;
						if ((p_new.pos_z-zk_flux) < 0) {
							dz_new = -(p_new.pos_z-zk_flux);
							dz_old = 	 p_old.pos_z-zk_flux;
						} else {
							dz_new = 	 p_new.pos_z-zk_flux;
							dz_old = -(p_old.pos_z-zk_flux);
						}
						float x_cross = p_new.pos_x + (p_old.pos_x-p_new.pos_x) * dz_new / (dz_new + dz_old);
						float y_cross = p_new.pos_y + (p_old.pos_y-p_new.pos_y) * dz_new / (dz_new + dz_old);

						int x_cross_id = (int)(x_cross / dx);
						int y_cross_id = (int)(y_cross / dx);

						if ((p_new.pos_z-zk_flux)<0) flux[x_cross_id + y_cross_id*x_domain + k*y_domain*x_domain]--;
						else 												 flux[x_cross_id + y_cross_id*x_domain + k*y_domain*x_domain]++;

						num_p_flux[k]++;
					}
				}
				jj++;
				break;
			} else if (p_new.index > p_old.index) {
				jj++;
			} else {
				break;
			}
		}
	}
}

void ParticleFlux::output_flux (Setting& setting) {
	// Copy setting
	const int	x_domain 	= setting.X_DOMAIN;
	const int y_domain 	= setting.Y_DOMAIN;
	const int n_slice	 	= setting.N_FLUX;
	const char* dir_out	= setting.DIR_OUT;
	
	// Use vector instead of VLA
	std::vector<float> z_flux(n_slice);
	std::copy(setting.Z_FLUX,	setting.Z_FLUX+n_slice, z_flux.data());

	// Output
	for (int i=0; i<n_slice; i++) {
		float zi_flux	= z_flux[i];

		std::cout << "Output flux (" << zi_flux << ") >>>>> ";
		std::fstream fFile;
		char				 fluxFile[100];

		sprintf(fluxFile, "./%s/flux_%dm.csv", dir_out, (int)zi_flux);
		fFile.open(fluxFile, std::ios::out);
		if (!fFile.is_open()) {
			std::cout << "cannot open the output file" << std::endl;
			std::cout << "File (flux file)" << std::endl;
			exit(EXIT_FAILURE);
		}
		for (int j=0; j<y_domain; j++) {
			for (int k=0; k<x_domain; k++) {
				fFile << flux[k+j*x_domain+i*y_domain*x_domain];
				if (k<x_domain-1) fFile << ",";
			}
			fFile << std::endl;
		}
		fFile.close();
		std::cout << "Finish!" << std::endl;
	}
}
		
void ParticleFlux::delete_flux () {
	delete[] flux;
}