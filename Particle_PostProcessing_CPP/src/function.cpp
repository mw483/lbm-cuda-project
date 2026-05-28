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

#include "stParticle.h"
#include "stSetting.h"
#include "function.h"


/* ---------- Function sort 		---------- */

int sort_compare (const void* A, const void* B) {
	const Particle_Index* a = (const Particle_Index*)A;
	const Particle_Index* b = (const Particle_Index*)B;

	int diff = a->index - b->index;
	if (diff == 0) {
		std::cout << "Error!! index is incorrect." << std::endl;
		exit(EXIT_FAILURE);
	}

	return diff;
}

bool sort_index (const Particle_Index& A, const Particle_Index& B) {
	return A.index < B. index;
}

void read_source (Particle_Index* p_source, Setting& setting) {
	// Copy setting
	const int 	n_source 			= setting.N_SOURCE;
	const int 	id_digit			= setting.ID_DIGIT;
	const char*	fname_source 	= setting.FNAME_SOURCE;

	// Read
	std::cout << "Read source data. File name >> " << fname_source <<  std::endl;
	
	std::fstream	sFile;

	sFile.open(fname_source, std::ios::in);
	if(!sFile.is_open()) {
		std::cout << "cannot open source file" << std::endl;
		exit(EXIT_FAILURE);
	}

	int group;
	int index_;
	Particle_Index p_min;
		p_min.index = 10000;
		p_min.pos_x = 0.0;
		p_min.pos_y = 0.0;
		p_min.pos_z = 0.0;

	Particle_Index p_max;
		p_max.index = 0;
		p_max.pos_x = 0.0;
		p_max.pos_y = 0.0;
		p_max.pos_z = 0.0;

	for (int i=0; i<n_source; i++) {
		sFile >> p_source[i].pos_x;
		sFile >> p_source[i].pos_y;
		sFile >> p_source[i].pos_z;
		sFile >> group;
		sFile >> index_;
		p_source[i].index = index_ / pow(10, id_digit);
		
		if (p_min.index > p_source[i].index) {
			p_min.index = p_source[i].index;
			p_min.pos_x = p_source[i].pos_x;
			p_min.pos_y = p_source[i].pos_y;
			p_min.pos_z = p_source[i].pos_z;
		}
		if (p_max.index < p_source[i].index) {
			p_max.index = p_source[i].index;
			p_max.pos_x = p_source[i].pos_x;
			p_max.pos_y = p_source[i].pos_y;
			p_max.pos_z = p_source[i].pos_z;
		}

		if (p_source[i].index <= 0 || n_source < p_source[i].index) {
			std::cout << "Error!! Source data is incorrect.";
		}
	}
	std::cout << " -min::index = " << p_min.index << " (x=" << p_min.pos_x << " y=" << p_min.pos_y  << " z=" << p_min.pos_z << ")" << std::endl;
	std::cout << " -max::index = " << p_max.index << " (x=" << p_max.pos_x << " y=" << p_max.pos_y  << " z=" << p_max.pos_z << ")" << std::endl;
	std::cout << std::endl;

	// sort
	qsort(p_source, n_source, sizeof *p_source, sort_compare);
}

