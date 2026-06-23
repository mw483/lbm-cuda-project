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

#include "residence.h"
#include "stSetting.h"
#include "stParticle.h"
#include "function.h"

void ParticleResidenceTime::allocate_residence (Setting& setting) {
	// Copy setting 
	const int x_domain = setting.X_DOMAIN;
	const int y_domain = setting.Y_DOMAIN;
	const int n_source = setting.N_SOURCE;
	const int n_gen		 = setting.NUM_GEN;

	// Allocate
	std::cout << "Allocate residence time memory -----" << std::endl;
	n_residence_map				= new int[y_domain*x_domain];
	residence_particle 		= new float[n_source*n_gen];
	residence_map      		= new float[y_domain*x_domain];
	
	// Inititialize
	for (int i=0; i<n_source*n_gen; i++) { 
		residence_particle[i] 	= 0.0;
	}
	for (int i=0; i<y_domain; i++) {
		for (int j=0; j<x_domain; j++) {
			n_residence_map[j+i*x_domain] = 0;
			residence_map[j+i*x_domain] 	= 0.0;
		}
	}
}

void ParticleResidenceTime::cal_residence (int t, std::vector<Particle_Index> p_resid_odd, std::vector<Particle_Index> p_resid_even, Setting& setting) {
	// Copy setting
	const	int		pout	 = setting.POUT;
	const int start		 = setting.FILE_START;
	const int x_domain = setting.X_DOMAIN;
	const int dx 			 = setting.dX;
	const int id_digit = setting.ID_DIGIT;
	const int n_gen	   = setting.NUM_GEN;
	const float dt     = setting.dT;
			float z_resid  = setting.Z_RESID;

	// cal
	std::size_t n_odd 	= p_resid_odd.size();
	std::size_t n_even	= p_resid_even.size();
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
			p_new.pos_x = p_resid_odd[i].pos_x;
			p_new.pos_y = p_resid_odd[i].pos_y;
			p_new.pos_z = p_resid_odd[i].pos_z;
			p_new.index = p_resid_odd[i].index;
		} else {
			// new
			p_new.pos_x = p_resid_even[i].pos_x;
			p_new.pos_y = p_resid_even[i].pos_y;
			p_new.pos_z = p_resid_even[i].pos_z;
			p_new.index = p_resid_even[i].index;
		}
		for (int j=jj; j<n_old; j++) {
			if ((t-start)%2 == 1) {
				// old
				p_old.pos_x = p_resid_even[j].pos_x;
				p_old.pos_y = p_resid_even[j].pos_y;
				p_old.pos_z = p_resid_even[j].pos_z;
				p_old.index = p_resid_even[j].index;
			} else {
				// old
				p_old.pos_x = p_resid_odd[j].pos_x;
				p_old.pos_y = p_resid_odd[j].pos_y;
				p_old.pos_z = p_resid_odd[j].pos_z;
				p_old.index = p_resid_odd[j].index;
			}

			// check index
			if (p_new.index == p_old.index) {
				// cal
				if (p_old.pos_z <= z_resid && p_new.pos_z <= z_resid) {
					// residence time particle
					int id_p_source = p_new.index / (int)pow(10,id_digit);
					int id_gen_num  = p_new.index % (int)pow(10,id_digit);
					residence_particle[(id_gen_num) + n_gen * (id_p_source-1)] += dt*pout;

					// residence time map
					int id_x_old = (int)(p_old.pos_x/dx);
					int id_x_new = (int)(p_new.pos_x/dx);
					int id_y_old = (int)(p_old.pos_y/dx);
					int id_y_new = (int)(p_new.pos_y/dx);					

					int id_x_min = std::min(id_x_old, id_x_new);
					int id_x_max = std::max(id_x_old, id_x_new);
					int id_y_min = std::min(id_y_old, id_y_new);
					int id_y_max = std::max(id_y_old, id_y_new);

					int n_grid = (id_x_max - id_x_min + 1)*(id_y_max - id_y_min + 1);

					for (int i_id=id_y_min; i_id<=id_y_max; i_id++) {
						for (int j_id=id_x_min; j_id<=id_x_max; j_id++) {
							n_residence_map[j_id + i_id*x_domain]++;
							residence_map[j_id + i_id*x_domain] += dt*pout/(float)n_grid;
						}
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

void ParticleResidenceTime::output_residence (Setting& setting) {
	// Copy setting
	const int	x_domain 	= setting.X_DOMAIN;
	const int y_domain 	= setting.Y_DOMAIN;
	const int n_source  = setting.N_SOURCE;
	const int n_gen			= setting.NUM_GEN;
	const char* dir_out	= setting.DIR_OUT;

	// Average - Use std::vector instead of VLA
	std::vector<float> ave_residence_map(y_domain * x_domain);
	for (int i=0; i<y_domain; i++) {
		for (int j=0; j<x_domain; j++) {
			if (n_residence_map[j+i*x_domain] == 0) ave_residence_map[j+i*x_domain] = 0.0;
			else ave_residence_map[j+i*x_domain] = residence_map[j+i*x_domain] / n_residence_map[j+i*x_domain];
		}
	}

	// Output
	std::cout << "Output residence time particle >>>>> ";
		std::fstream rFile;
		char				 residFile[256];

		sprintf(residFile, "./%s/residence_time_particle.csv", dir_out);
		rFile.open(residFile, std::ios::out);
		if (!rFile.is_open()) {
			std::cout << "cannot open the output file" << std::endl;
			std::cout << "File (residence time particle file)" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		// Header row
		for (int j=0; j<n_gen+1; j++) {
			if (j==0) rFile << "Generation ID,";
			else rFile << j;
			if (j < n_gen) rFile << ","; // Fixed condition
		}
		rFile << std::endl; // Added missing newline after header
		
		// Data rows
		for (int i=0; i<n_source; i++) {
			rFile << "Source = " << i << ","; // Moved outside inner loop
			for (int j=0; j<n_gen; j++) {
				rFile << residence_particle[j + n_gen*i];
				if (j < n_gen-1) rFile << ","; // Fixed condition
			}
			rFile << std::endl;
		}
		rFile.close();
		std::cout << "Finish!" << std::endl;
	
	std::cout << "Output residence time map >>>>> ";
	sprintf(residFile, "./%s/residence_time_map.csv", dir_out);
		rFile.open(residFile, std::ios::out);
		if (!rFile.is_open()) {
			std::cout << "cannot open the output file" << std::endl;
			std::cout << "File (residence time map file)" << std::endl;
			exit(EXIT_FAILURE);
		}
		for (int i=0; i<y_domain; i++) {
			for (int j=0; j<x_domain; j++) {
				rFile << ave_residence_map[j+i*x_domain];
				if (j<x_domain-1) rFile << ",";
			}
			rFile << std::endl;
		}
		rFile.close();
		std::cout << "Finish!" << std::endl;
}

void ParticleResidenceTime::delete_residence () {
	delete[] n_residence_map;
	delete[] residence_particle;
	delete[] residence_map;
}