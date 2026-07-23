#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include "blending_footprint.h"
#include "stSetting.h"
#include "stParticle.h"
#include "function.h" 

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>

#include "blending_footprint.h"
#include "stSetting.h"
#include "stParticle.h"
#include "function.h"

void ParticleBlendingFootprint::allocate_blending(Setting& setting) {
    // 1. Calculate Dynamic Memory Ceiling based on your formula
    int base_multiplier = std::pow(10, setting.ID_DIGIT); // e.g., 10^3 = 1000 or 10^4 = 10000
    
    // Formula: (N_SOURCE * base_multiplier) + 1 + (FILE_END - FILE_START)
    max_particle_id = (setting.N_SOURCE * base_multiplier) + 1 
                    + (setting.FILE_END - setting.FILE_START + 1);
    
    // Add a small 10% safety buffer to prevent edge-case segfaults
    max_particle_id = max_particle_id + (max_particle_id / 10); 

    std::cout << "--- [Blending] Allocating Caches for Max ID: " << max_particle_id << " ---" << std::endl;

    // 2. Allocate the Forward Intercept Caches
    // Using new[] allows us to request contiguous blocks of memory.
    cached_x       = new float[max_particle_id];
    cached_y       = new float[max_particle_id];
    has_hit_sensor = new bool[max_particle_id];

    // Initialize all caches
    // It's critical to initialize to negative values or flags so we know if a cache is "empty"
    for (int i = 0; i < max_particle_id; i++) {
        cached_x[i] = -999.0f;       // -999 denotes 'no crossing recorded yet'
        cached_y[i] = -999.0f;
        has_hit_sensor[i] = false;
    }

    // 3. Allocate the 2D Blending Plane Output Grid
    // Copy domain parameters from setting
    nx = setting.X_DOMAIN; 
    ny = setting.Y_DOMAIN;
    dx = setting.dX;

    int total_cells = nx * ny;
    std::cout << "--- [Blending] Allocating 2D Grid (" << nx << " x " << ny << ") ---" << std::endl;
    blending_grid = new int[total_cells];

    for (int i = 0; i < total_cells; i++) {
        blending_grid[i] = 0;
    }
}


void ParticleBlendingFootprint::track_blending_footprint(float x, float y, float z, int id, Setting& setting) {
    
    // Safety check: Prevent memory violation if ID somehow exceeds allocation
    if (id < 0 || id >= max_particle_id) {
        // Silently skip to prevent spamming the cluster terminal
        return; 
    }

    debug_total_particles_seen++;

    // ========================================================
    // GATE A: The Forward Intercept Cache (The "Ascent")
    // ========================================================
    
    // We only care if the particle is currently above the blending height.
    // Replace '20.0f' with setting.Z_BLEND if you decide to make it a parameter later.
    float target_z_blend = setting.Z_BLEND; 

    // Check if the cache for this particle is currently empty (-999.0f) AND it has reached the height
    if (z >= target_z_blend && cached_x[id] == -999.0f) {
        
        // The particle has just pierced the blending plane for the first time.
        // Cache its exact horizontal coordinates.
        cached_x[id] = x;
        cached_y[id] = y;

        debug_cache_writes++; // TRACKER
        
        // Note: For absolute precision, we could interpolate between the previous 
        // time step's (z) and current (z) to find the exact (x,y) at exactly z=20.0, 
        // but for a 1-second LES timestep, taking the instantaneous (x,y) upon 
        // crossing is geometrically sufficient.
    }

    // ========================================================
    // GATE B: The Unique Sensor Intercept (The "Catch")
    // ========================================================
    
    // Check if this particle has already been counted by the sensor. 
    // If true, skip entirely to prevent residence-time biasing.
    if (has_hit_sensor[id] == true) {
        return; 
    }

    // Pull sensor boundaries from the setting file (assuming 1 sensor for this prototype)
    float x_sensor = setting.CTR_SENSOR_BLEND[0];
    float y_sensor = setting.CTR_SENSOR_BLEND[1];
    float z_sensor = setting.CTR_SENSOR_BLEND[2];

    float xs_min = x_sensor - setting.SIZE_SENSOR_BLEND[0] / 2.0;
    float xs_max = x_sensor + setting.SIZE_SENSOR_BLEND[0] / 2.0;
    float ys_min = y_sensor - setting.SIZE_SENSOR_BLEND[1] / 2.0;
    float ys_max = y_sensor + setting.SIZE_SENSOR_BLEND[1] / 2.0;
    float zs_min = z_sensor - setting.SIZE_SENSOR_BLEND[2] / 2.0;
    float zs_max = z_sensor + setting.SIZE_SENSOR_BLEND[2] / 2.0;

    // Is the particle currently inside the 3D bounding box?
    if (x >= xs_min && x <= xs_max && 
        y >= ys_min && y <= ys_max && 
        z >= zs_min && z <= zs_max) {
        
        // 1. Lock the gate so this ID is never counted again
        has_hit_sensor[id] = true;
        debug_sensor_hits++; // TRACKER
        
        // 2. Did this particle actually cross our blending plane before arriving?
        if (cached_x[id] != -999.0f) {
            
            // Retrieve the historical crossing coordinates
            float xb = cached_x[id];
            float yb = cached_y[id];

            // 3. Map continuous physical coordinates (meters) to discrete grid indices
            int grid_x = (int)(xb / dx);
            int grid_y = (int)(yb / dx); // Assuming dy exists, or dx if grid is uniform

            // Safety boundary check to prevent array index out-of-bounds
            if (grid_x >= 0 && grid_x < nx && grid_y >= 0 && grid_y < ny) {
                
                // 4. Increment the spatial cell on our flattened 2D grid matrix
                blending_grid[grid_x + (grid_y * nx)]++;
                debug_grid_writes++; // TRACKER
            }
        }
    }
}

void ParticleBlendingFootprint::output_blending_footprint(Setting& setting) {
    std::cout << "--- [DIAGNOSTICS] ---" << std::endl;
    std::cout << "Total Particles Seen: " << debug_total_particles_seen << std::endl;
    std::cout << "Successfully Cached at Z=" << setting.Z_BLEND << ": " << debug_cache_writes << std::endl;
    std::cout << "Successfully Hit Sensor Box: " << debug_sensor_hits << std::endl;
    std::cout << "Successfully Written to Grid: " << debug_grid_writes << std::endl;
    std::cout << "---------------------" << std::endl;
    
    // We only have one sensor in this prototype, but we pull its coords 
    // for the filename so you know exactly which sensor this map belongs to.
    float x_sensor = setting.CTR_SENSOR_BLEND[0];
    float y_sensor = setting.CTR_SENSOR_BLEND[1];
    float z_sensor = setting.CTR_SENSOR_BLEND[2];

    const char* dir_out = setting.DIR_OUT;
    
    std::cout << "--- [Blending] Outputting Virtual Footprint Plane (" 
              << x_sensor << ", " << y_sensor << ", " << z_sensor << ") ---" << std::endl;

    // Create a unique filename indicating the blending height used
    char blendFile[256];
    sprintf(blendFile, "%s/blending_footprint_z%d_sensor_%d_%d_%d.csv", 
            dir_out, (int)setting.Z_BLEND, (int)x_sensor, (int)y_sensor, (int)z_sensor);
            
    std::fstream fFile;
    fFile.open(blendFile, std::ios::out);
    
    if (!fFile.is_open()) {
        std::cout << "ERROR: Cannot open blending output file." << std::endl;
        return; 
    }

    // Write a clean CSV Header for Python (Pandas/Polars love this)
    fFile << "X_Index,Y_Index,Count" << std::endl;

    // Loop through the 2D spatial grid and only write non-zero cells 
    // to save disk space and I/O time on TSUBAME
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            
            // Map the 2D indices to our flattened 1D array
            int cell_index = i + (j * nx);
            int count = blending_grid[cell_index];
            
            if (count > 0) {
                fFile << i << "," << j << "," << count << std::endl;
            }
        }
    }

    fFile.close();
    std::cout << "--- [Blending] Output Complete: " << blendFile << " ---" << std::endl;
}


void ParticleBlendingFootprint::delete_blending() {
    if(cached_x)       delete[] cached_x;
    if(cached_y)       delete[] cached_y;
    if(has_hit_sensor) delete[] has_hit_sensor;
    if(blending_grid)  delete[] blending_grid;
    
    std::cout << "--- [Blending] Memory Freed ---" << std::endl;
}