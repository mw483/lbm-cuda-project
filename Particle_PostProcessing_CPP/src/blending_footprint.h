#ifndef CLASS_BLEND_FOOTPRINT
#define CLASS_BLEND_FOOTPRINT

#include <iostream>

#include "stSetting.h"
#include "stParticle.h"

class ParticleBlendingFootprint {
    private:
        // --- 1. The Global Memory Ceiling ---
        // You mentioned max ID is around 31,000,000. 
        // We will define this during allocation.
        int max_particle_id; 

        // --- 2. The Forward Intercept Cache ---
        // Direct arrays where the index = Particle ID
        float* cached_x;
        float* cached_y;
        bool* has_hit_sensor;

        // --- 3. The 2D Output Grid (The Blending Plane) ---
        // Flattened 2D array: [grid_x + grid_y * NX]
        int* blending_grid; 
        
        // Grid properties derived from stSetting.h
        int nx;
        int ny;
        float dx;
        float dy;

    public:
        // --- Core Pipeline Hooks ---
        void allocate_blending(Setting& setting);
        void track_blending_footprint(float x, float y, float z, int id, Setting& setting);
        void output_blending_footprint(Setting& setting);
        void delete_blending(); 
};

#endif