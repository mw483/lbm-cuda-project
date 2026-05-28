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
#include <vector>

#include "stSetting.h"
#include "stParticle.h"
#include "function.h"
#include "num.h"
#include "density.h"
#include "profile.h"
#include "footprint.h"
#include "flux.h"
#include "residence.h"

/* ---------- Function Calculation --------- */

void calculation_pos (Setting& setting) {
	// Class
	ParticleNum 			p_num;
	ParticleDensity 	p_density;
	ParticleProfile		p_profile;

	// Copy setting
	const int 	num_rank 		= setting.NUM_RANK;
	const int 	start				= setting.FILE_START;
	const int 	end					= setting.FILE_END;

	const int		flg_num			= setting.FLG_NUM;
	const int 	flg_density = setting.FLG_DENSITY;
	const int		flg_profile = setting.FLG_PROFILE;
	
	const char* dir_data	 	= setting.DIR_DATA;

	// Allocate
	if (flg_num 		== 1) p_num.allocate_num(setting);
	if (flg_density == 1) p_density.allocate_density(setting);
	if (flg_profile == 1) p_profile.allocate_profile(setting);
	
	// Read map
	if (flg_profile == 1) {
		p_profile.read_map(setting);
	}
	
	/* ----- Start Loop time ----- */
	for (int t=start; t<=end; t++) {
		std::cout << "Current processing :: " << t << " ( " << start << " - " << end << " ) ---------- " <<  std::endl;
		// Read position
		std::fstream	pFile;
		char		posFile[100];
  	
		float 	x, y, z;
		int			n=0;
		
		// Loop rank
		for (int rank = 0; rank < num_rank; rank++) {
			std::cout << "-";
			
			sprintf(posFile, "./%s/position%d-%d.bin", dir_data, rank, t);
			pFile.open(posFile  , std::ios::in | std::ios::binary);
			if (!pFile.is_open()) {
				std::cout<< "cannnot open the file" << std::endl;
				std::cout<< "File (Rank =  "	<< rank << " / t = " << t << ")" <<std::endl;
				exit(EXIT_FAILURE);
			}
			int i = 1;
			pFile.seekg(i * sizeof(int));
			while (true) {
				pFile.read((char *) &x	  , sizeof(float));
				if (pFile.eof()) {
					break;
				}
				pFile.read((char *) &y    , sizeof(float));
				pFile.read((char *) &z    , sizeof(float));
				
				// cal density
				if (flg_density == 1) p_density.cal_density(x, y, z, setting);
				if (flg_profile == 1) p_profile.cal_profile(x, y, z, setting);
	
				n++;
			}
			pFile.close();
			// cal total number
			if (flg_num == 1) p_num.cal_num(t, n, setting);
		}
		std::cout << std::endl;
		std::cout << "number of particles = " << n << std::endl;
		std::cout << std::endl;
	}
	/* -----  End Loop time ----- */
	
	// Output
	if (flg_num 		== 1) p_num.output_num(setting);
	if (flg_density == 1) p_density.output_density(setting);
	if (flg_profile == 1) p_profile.output_profile(setting);
	
	// Delete
	if (flg_num			== 1) p_num.delete_num(); 
	if (flg_density == 1) p_density.delete_density();
	if (flg_profile == 1) p_profile.delete_profile();
}

/* -------------------- Index -------------------- */

void calculation_index (Setting& setting) {
	// Class
	ParticleNum 				p_num;
	ParticleDensity 			p_density;
	ParticleProfile				p_profile;
	ParticleFootPrint			p_foot;
	ParticleFlux				p_flux;
	ParticleResidenceTime 		p_resid;

	// Struct
	std::vector<Particle_Index>	p_index_odd;
	std::vector<Particle_Index> p_index_even;

	// Copy setting
	const int 	num_rank 		= setting.NUM_RANK;
	const int 	start				= setting.FILE_START;
	const int 	end					= setting.FILE_END;
	const int 	n_sensor		= setting.N_SENSOR;
	const int 	n_flux			= setting.N_FLUX;

	const int		flg_num			= setting.FLG_NUM;
	const int 	flg_density = setting.FLG_DENSITY;
	const int		flg_profile = setting.FLG_PROFILE;
	const int 	flg_foot		= setting.FLG_FOOT;
	const int 	flg_flux 		= setting.FLG_FLUX;
	const int 	flg_resid		= setting.FLG_RESID;
	
	const char* dir_data	 	= setting.DIR_DATA;

	// Allocate
	if (flg_num 		== 1) p_num.allocate_num(setting);
	if (flg_density == 1) p_density.allocate_density(setting);
	if (flg_profile == 1) p_profile.allocate_profile(setting);
	if (flg_foot		== 1) p_foot.allocate_footprint_source(setting);
	if (flg_flux		== 1) p_flux.allocate_flux(setting);
	if (flg_resid   == 1) p_resid.allocate_residence(setting);

	std::cout << std::endl;
	
	// Read map
	if (flg_profile == 1) {
		p_profile.read_map(setting);
	}
	
	// Read source
	if (flg_foot == 1) {
		p_foot.read_source(setting);
		p_foot.allocate_footprint(setting);
	}
	
	/* ----- Start Loop time ----- */
	for (int t=start; t<=end; t++) {
		// Initialize
		if (flg_foot == 1) {
			for (int i_foot=0; i_foot<n_sensor; i_foot++) p_foot.num_p_sensor[i_foot] = 0;
		}
		if (flg_flux == 1) {
			for (int i_flux=0; i_flux<n_flux; i_flux++) p_flux.num_p_flux[i_flux]		= 0;
		}

		// clear memory
		if (flg_flux == 1 || flg_resid == 1) {
			if ((t-start) % 2 == 1) std::vector<Particle_Index>().swap(p_index_odd);
			else 										std::vector<Particle_Index>().swap(p_index_even);
		}
	
		// Calculation
		std::cout << "Current processing :: " << t << " ( " << start << " - " << end << " ) ---------- " <<  std::endl;
		// Read index and position
		std::fstream	pFile;
		char					posFile[100];
		
		std::fstream 	iFile;
		char					idFile[100];
		
 	 
		float 	x, y, z;
		int 		id;
		int			n=0;
		
		for (int rank = 0; rank < num_rank; rank++) {
			sprintf(posFile, "./%s/position%d-%d.bin", dir_data, rank, t);
			pFile.open(posFile  , std::ios::in | std::ios::binary);
			if (!pFile.is_open()) {
				std::cout<< "cannnot open the position file" << std::endl;
				std::cout<< "File (Rank =  "	<< rank << " / t = " << t << ")" <<std::endl;
				exit(EXIT_FAILURE);
			}

			sprintf(idFile, "./%s/index%d-%d.bin", dir_data, rank, t);
			iFile.open(idFile  , std::ios::in | std::ios::binary);
			if (!iFile.is_open()) {
				std::cout<< "cannnot open the index file" << std::endl;
				std::cout<< "File (Rank =  "	<< rank << " / t = " << t << ")" <<std::endl;
				exit(EXIT_FAILURE);
			}
		
			int i = 1;
			pFile.seekg(i * sizeof(int));
			iFile.seekg(i * sizeof(int));
			while (true) {
				pFile.read((char *) &x	  , sizeof(float));
				if (pFile.eof()) {
					break;
				}
				pFile.read((char *) &y    , sizeof(float));
				pFile.read((char *) &z    , sizeof(float));
				iFile.read((char *) &id 	, sizeof(int));
	
				// push back for flux
				if (flg_flux		== 1 || flg_resid == 1) {
					Particle_Index p_tmp;
					p_tmp.pos_x	= x;
					p_tmp.pos_y = y;
					p_tmp.pos_z = z;
					p_tmp.index = id;

					if ((t-start)%2 == 1) p_index_odd.push_back(p_tmp);
					else									p_index_even.push_back(p_tmp);
				}

				// main calculation in loop
				if (flg_density == 1) 								p_density.cal_density(x, y, z, setting);
				if (flg_profile == 1) 								p_profile.cal_profile(x, y, z, setting);
				if (flg_foot		== 1) 								p_foot.cal_footprint(x, y, z, id, setting);
	
				// count
				n++;
			} // loop for particle
			pFile.close();
			iFile.close();
		} // loop for rank
		
		// Sort vector
		if (flg_flux == 1 || flg_resid == 1 ) {
			if ((t-start)%2==1) 	sort(p_index_odd.begin(), p_index_odd.end(), sort_index);
			else 									sort(p_index_even.begin(), p_index_even.end(), sort_index);
		}

		// main calculation after loop
		if (flg_num 	== 1) p_num.cal_num(t, n, setting);
		if (flg_flux	== 1 && t >= start+1) p_flux.cal_flux(t, p_index_odd, p_index_even,  setting);
		if (flg_resid == 1 && t >= start+1) p_resid.cal_residence(t, p_index_odd, p_index_even, setting);

		std::cout << "--- number of particles = " << n << std::endl;
		if (flg_foot == 1) {
			for (int i_out=0; i_out<n_sensor; i_out++) { 
				std::cout << "--- number of particles in sensor(" << i_out << ") = " << p_foot.num_p_sensor[i_out] << std::endl;
			}
		}
		if (flg_flux == 1 && t >= start + 1) {
			for (int i_flux=0; i_flux<n_flux; i_flux++) {
			std::cout << "--- number of particles cross plane for flux (" << i_flux << ") = " << p_flux.num_p_flux[i_flux] << std::endl;
			}
		}

	std::cout << std::endl;

	} // loop for time

	// Output
	if (flg_num 		== 1) p_num.output_num(setting);
	if (flg_density == 1) p_density.output_density(setting);
	if (flg_profile == 1) p_profile.output_profile(setting);
	if (flg_foot 		== 1) p_foot.output_footprint(setting);
	if (flg_flux		== 1) p_flux.output_flux(setting);
	if (flg_resid   == 1) p_resid.output_residence(setting);
	
	// Delete
	if (flg_num			== 1) p_num.delete_num(); 
	if (flg_density == 1) p_density.delete_density();
	if (flg_profile == 1) p_profile.delete_profile();
	if (flg_foot		== 1) p_foot.delete_footprint();
	if (flg_flux		== 1) p_flux.delete_flux();
	if (flg_resid   == 1) p_resid.delete_residence();
}

// Add this main function to your calculation.cpp file
/*
/* int main() {
    // Create and initialize settings
    Setting setting;
    
    // You'll need to initialize your setting object here
    // This depends on how your Setting class is structured
    // Example initialization (adjust based on your stSetting.h):
    /*
    setting.NUM_RANK = 1;        // adjust as needed
    setting.FILE_START = 0;      // adjust as needed  
    setting.FILE_END = 10;       // adjust as needed
    setting.FLG_NUM = 1;         // enable number calculation
    setting.FLG_DENSITY = 0;     // disable density calculation
    setting.FLG_PROFILE = 0;     // disable profile calculation
    setting.FLG_FOOT = 0;        // disable footprint calculation
    setting.FLG_FLUX = 0;        // disable flux calculation
    setting.FLG_RESID = 0;       // disable residence calculation
    strcpy(setting.DIR_DATA, "data");  // data directory
    */
    
    // std::cout << "Starting calculation..." << std::endl;
    
    // Choose which calculation to run:
    
    // Option 1: Run position-only calculation
    // calculation_pos(setting);
    
    // Option 2: Run index-based calculation (comment out the line above if using this)
    // calculation_index(setting);
    
    // Option 3: Run both calculations
    // calculation_pos(setting);
    // calculation_index(setting);
    
    // std::cout << "Calculation completed." << std::endl;
    
    /*return 0;
} */