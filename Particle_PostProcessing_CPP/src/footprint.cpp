#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include "footprint.h"
#include "stSetting.h"
#include "stParticle.h"
#include "function.h" 

/* -------- Class ParticleFootPrint -------- */

// Helper for sorting (if needed by other parts of the code)
int ParticleFootPrint::binary_search_sort (int id, int n_source) {
    // Not used in this simplified direct-index version, 
    // but kept to prevent linker errors if declared in header.
    return 0; 
}

void ParticleFootPrint::allocate_footprint_source (Setting& setting) {
    // Copy setting
    const int n_source = setting.N_SOURCE;
    
    // Allocate
    std::cout << "Allocate source memory -----" << std::endl;
    p_source = new Particle_Index[n_source];
}

void ParticleFootPrint::allocate_footprint (Setting& setting) {
    // Copy setting
    const int n_sensor = setting.N_SENSOR;
    const int n_source = setting.N_SOURCE;
    
    // Allocate simple 2D Array flattened: [Sensor][SourceIndex]
    std::cout << "Allocate footprint memory (List Format: " << n_sensor << " x " << n_source << ")..." << std::endl;
    footprint    = new int[n_sensor * n_source];
    num_p_sensor = new int[n_sensor];

    // Initialization
    for (int i=0; i < n_sensor * n_source; i++) {
        footprint[i] = 0;
    }
    for (int i=0; i < n_sensor; i++) {
        num_p_sensor[i] = 0; 
    }
}

void ParticleFootPrint::read_source (Setting& setting) {
    const int   n_source      = setting.N_SOURCE;
    const char* fname_source  = setting.FNAME_SOURCE;

    std::cout << "Reading Source File: " << fname_source << std::endl;
    
    std::fstream sFile;
    sFile.open(fname_source, std::ios::in);
    if(!sFile.is_open()) {
        std::cout << "FATAL ERROR: Cannot open source file" << std::endl;
        exit(EXIT_FAILURE);
    }

    float temp_u, temp_v, temp_w; 
    int temp_group, temp_id;

    for (int i=0; i<n_source; i++) {
        // 1. Read Pos (Useful for verification, though not strictily needed for list output)
        sFile >> p_source[i].pos_x;
        sFile >> p_source[i].pos_y;
        sFile >> p_source[i].pos_z;
        
        // 2. Read & Skip Velocities
        sFile >> temp_u >> temp_v >> temp_w;
        
        // 3. Read Group & ID
        sFile >> temp_group;
        sFile >> temp_id;

        // 4. Map ID to Index
        // Formula: 10001 -> 0, 20001 -> 1
        p_source[i].index = (temp_id / 10000) - 1;

        if (i == 0) {
            std::cout << "Debug [Line 1]: ID " << temp_id << " mapped to Index " << p_source[i].index << std::endl;
        }
    }
    sFile.close();
    
    // Note: We don't strictly need to sort p_source if we trust 
    // the ID mapping formula matches the array index 0..N
}

void ParticleFootPrint::cal_footprint (float x, float y, float z, int index, Setting& setting) {
    const int   n_source    = setting.N_SOURCE;
    const int   n_sensor    = setting.N_SENSOR;
    
    std::vector<float> ctr_sensor(n_sensor * 3);
    std::vector<float> size_sensor(3);
    
    std::copy(setting.CTR_SENSOR, setting.CTR_SENSOR + n_sensor * 3, ctr_sensor.begin());
    std::copy(setting.SIZE_SENSOR, setting.SIZE_SENSOR + 3, size_sensor.begin());

    for (int i=0; i<n_sensor; i++) {
        float x_sensor = ctr_sensor[3*i  ];
        float y_sensor = ctr_sensor[3*i+1];
        float z_sensor = ctr_sensor[3*i+2];

        float xs_min   = x_sensor - size_sensor[0] / 2.0;
        float xs_max   = x_sensor + size_sensor[0] / 2.0;
        float ys_min   = y_sensor - size_sensor[1] / 2.0;
        float ys_max   = y_sensor + size_sensor[1] / 2.0;
        float zs_min   = z_sensor - size_sensor[2] / 2.0;
        float zs_max   = z_sensor + size_sensor[2] / 2.0;

        if (xs_min <= x && x <= xs_max && ys_min <= y && y <= ys_max && zs_min <= z && z <= zs_max ) {
            
            // 1. Map Particle ID directly to Array Index
            // 10001 -> 0, 20001 -> 1
            int id_s = (index / 10000) - 1; 

            // 2. Increment Counter directly
            // No binary search or grid mapping needed here.
            if(id_s >= 0 && id_s < n_source) {
                footprint[id_s + i*n_source]++;
                num_p_sensor[i]++;
            }
        }
    }
}

void ParticleFootPrint::output_footprint (Setting& setting) {
    const int n_sensor    = setting.N_SENSOR;
    const int n_source    = setting.N_SOURCE;
    const char* dir_out   = setting.DIR_OUT;
    
    std::vector<float> ctr_sensor(n_sensor * 3);
    std::copy(setting.CTR_SENSOR, setting.CTR_SENSOR + n_sensor * 3, ctr_sensor.begin());

    for (int i=0; i<n_sensor; i++) {
        float x_sensor = ctr_sensor[3*i  ];
        float y_sensor = ctr_sensor[3*i+1];
        float z_sensor = ctr_sensor[3*i+2];

        std::cout << "Output footprint (" << x_sensor << ", " << y_sensor << ", " << z_sensor << ") ..." << std::endl;
        
        char footFile[256];
        sprintf(footFile, "%s/footprint_%d_%d_%d.csv", dir_out, (int)x_sensor, (int)y_sensor, (int)z_sensor);
        
        std::fstream fFile;
        fFile.open(footFile, std::ios::out);
        if (!fFile.is_open()) {
            std::cout << "Error: Cannot open output file." << std::endl;
            continue; 
        }
        
        // Row 1: Source Indices (0, 1, 2, ... N)
        for (int j=0; j<n_source; j++) {
            fFile << j;
            if (j < n_source-1) fFile << ",";
        }
        fFile << std::endl; 
    
        // Row 2: Counts
        for (int j=0; j<n_source; j++) {
            fFile << footprint[j + i*n_source];
            if (j < n_source-1) fFile << ",";
        }
        fFile << std::endl;
        fFile.close();
    }
    std::cout << "Output Complete." << std::endl;
}

void ParticleFootPrint::delete_footprint () {
    if(footprint) delete[] footprint;
    if(p_source) delete[] p_source;
    if(num_p_sensor) delete[] num_p_sensor;
}