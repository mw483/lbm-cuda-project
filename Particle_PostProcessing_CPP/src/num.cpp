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

#include "num.h"
#include "stSetting.h"

/* ---------- Class ParticleNum ---------- */ 
void ParticleNum::allocate_num (Setting& setting) {
	// Copy setting
	const int end 	= setting.FILE_END;

	// Allocate
	std::cout << "Allocate num memory -----" << std::endl;
	total_num	= new int[2*(end+1)];

	// Initialization
	for (int i=0; i<2; i++) {
		for (int j=0; j<(end+1); j++) {
			if (i == 0) total_num[j+i*(end+1)] = j;
			else total_num[j+i*(end+1)] = 0;
		}
	}
}

void ParticleNum::cal_num (int t, int n, Setting& setting) {
	// Copy setting
	const int end = setting.FILE_END;
	// cal
	total_num[(end+1)+t] = n;
}

void ParticleNum::output_num (Setting& setting) {
	// Copy setting
	const int 	end			= setting.FILE_END;
	const char* dir_out	= setting.DIR_OUT;

	// Output
	std::cout << "Output num >>>>> ";
	std::fstream	nFile;
	char		numFile[100];
	
	sprintf(numFile, "./%s/total_num.csv", dir_out);
	
	nFile.open(numFile, std::ios::out);
	if (!nFile.is_open()) {
		std::cout << "cannot open the output file" << std::endl;
		std::cout << "File (num file)" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	for (int i=0; i<2; i++) {
		for (int j=0; j<end+1; j++) {
			nFile << total_num[j + i* (end+1)];
			if (j < end) nFile << ",";
		}
			nFile << std::endl;
	}
	nFile.close();
	std::cout << "Finish!!" << std::endl;
	
}

void ParticleNum::delete_num () {
	delete[] total_num;
}

