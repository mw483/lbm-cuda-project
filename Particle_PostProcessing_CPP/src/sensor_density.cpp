#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sensor_density.h"

void ParticleSensorDensity::allocate_and_load(Setting& setting) {
    n_source = setting.N_SOURCE;
    n_sensor = setting.N_SENSOR_DENSITY;
    
    // ---------------------------------------------------------
    // 1. ALLOCATE THE HIT LIST (TIME CAPSULE MEMORY)
    // ---------------------------------------------------------
    std::cout << "--- [Sensor Density] Allocating Hit List for " << n_sensor << " Sensors x " << n_source << " Sources ---" << std::endl;
    source_hit_sensor = new bool[n_sensor * n_source];
    
    for(int i = 0; i < (n_sensor * n_source); i++) {
        source_hit_sensor[i] = false;
    }

    // ---------------------------------------------------------
    // 2. STAGE 2: READ THE TIME CAPSULE IF APPLICABLE
    // ---------------------------------------------------------
    if (setting.FLG_SENSOR_DENSITY == 1) {
        char hitFile[256];
        sprintf(hitFile, "%s/sensor_hit_ids.txt", setting.DIR_OUT);
        
        std::fstream fFile;
        fFile.open(hitFile, std::ios::in);
        if (fFile.is_open()) {
            int sens_id, src_id;
            float temp_x, temp_y, temp_z;
            
            // Read all 5 columns. We use the coordinates for documentation, but only need the IDs here.
            while (fFile >> sens_id >> temp_x >> temp_y >> temp_z >> src_id) {
                if (sens_id >= 0 && sens_id < n_sensor && src_id >= 0 && src_id < n_source) {
                    source_hit_sensor[src_id + (sens_id * n_source)] = true;
                }
            }
            fFile.close();
            std::cout << "--- [Sensor Density] Successfully loaded Time Capsule: " << hitFile << " ---" << std::endl;
        } else {
            std::cout << "FATAL ERROR: FLG_SENSOR_DENSITY is 1, but sensor_hit_ids.txt is missing!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // ---------------------------------------------------------
    // 3. STAGE 2: ALLOCATE MASSIVE DENSITY GRIDS
    // ---------------------------------------------------------
    if (setting.FLG_SENSOR_DENSITY == 1) {
        const int x_domain = setting.X_DOMAIN;
        const int y_domain = setting.Y_DOMAIN;
        const int z_domain = setting.Z_DOMAIN;
        const int n_slice_xy = setting.N_XY;
        const int n_slice_xz = setting.N_XZ;
        const int n_slice_yz = setting.N_YZ;

        // Note the multiplication by n_sensor!
        xy_sensor_density = new int[n_sensor * n_slice_xy * y_domain * x_domain]();
        xz_sensor_density = new int[n_sensor * n_slice_xz * z_domain * x_domain]();
        yz_sensor_density = new int[n_sensor * n_slice_yz * z_domain * y_domain]();
    }
}


void ParticleSensorDensity::harvest_ids(float x, float y, float z, int id, Setting& setting) {
    // We do not waste time checking particles if they don't map to a valid source
    int source_index = (id / 10000) - 1; 
    if(source_index < 0 || source_index >= n_source) return;

    // Loop through all 33 sensors to see if this particle hit any of them
    for (int i = 0; i < n_sensor; i++) {
        
        // If this source is already flagged for this sensor, skip math
        if (source_hit_sensor[source_index + (i * n_source)]) continue;

        float x_sensor = setting.CTR_SENSOR_DENSITY[3*i];
        float y_sensor = setting.CTR_SENSOR_DENSITY[3*i+1];
        float z_sensor = setting.CTR_SENSOR_DENSITY[3*i+2];

        float xs_min = x_sensor - setting.SIZE_SENSOR_DENSITY[0] / 2.0;
        float xs_max = x_sensor + setting.SIZE_SENSOR_DENSITY[0] / 2.0;
        float ys_min = y_sensor - setting.SIZE_SENSOR_DENSITY[1] / 2.0;
        float ys_max = y_sensor + setting.SIZE_SENSOR_DENSITY[1] / 2.0;
        float zs_min = z_sensor - setting.SIZE_SENSOR_DENSITY[2] / 2.0;
        float zs_max = z_sensor + setting.SIZE_SENSOR_DENSITY[2] / 2.0;

        if (x >= xs_min && x <= xs_max && y >= ys_min && y <= ys_max && z >= zs_min && z <= zs_max) {
            // Particle is inside sensor 'i'. Flag it!
            source_hit_sensor[source_index + (i * n_source)] = true;
        }
    }
}


void ParticleSensorDensity::cal_sensor_density(float x, float y, float z, int id, Setting& setting) {
    int source_index = (id / 10000) - 1; 
    if(source_index < 0 || source_index >= n_source) return;

    const int x_domain = setting.X_DOMAIN;
    const int y_domain = setting.Y_DOMAIN;
    const int z_domain = setting.Z_DOMAIN;
    const int n_slice_xy = setting.N_XY;
    const int n_slice_xz = setting.N_XZ;
    const int n_slice_yz = setting.N_YZ;
    const float dx = setting.dX;
    const float h_ave = setting.H_AVE;

    std::vector<float> z_out(n_slice_xy);
    std::copy(setting.Z_OUT, setting.Z_OUT+n_slice_xy, z_out.data());

    std::vector<float> y_out(n_slice_xz);
    std::copy(setting.Y_OUT, setting.Y_OUT+n_slice_xz, y_out.data());
    
    std::vector<float> x_out(n_slice_yz);
    std::copy(setting.X_OUT, setting.X_OUT+n_slice_yz, x_out.data());

    // Loop through every sensor to see if we should count this particle for its specific grid
    for (int s = 0; s < n_sensor; s++) {
        
        // THE BOUNCER: Did this particle's source ever hit Sensor 's'?
        if (!source_hit_sensor[source_index + (s * n_source)]) continue;

        // If yes, log its position in Sensor 's's specific density grids
        
        // XY Planes
        for (int i=0; i<n_slice_xy; i++) {
            float z_ctr = z_out[i];
            float z_min = z_ctr - h_ave / 2;
            float z_max = z_ctr + h_ave / 2;

            if (z_min < z && z < z_max) {
                int x_index = std::max(0, (int)(x/dx));
                int y_index = std::max(0, (int)(y/dx));
                
                // Offset calculation for [Sensor][Z_Slice][Y][X]
                int grid_offset = (s * n_slice_xy * y_domain * x_domain) 
                                + (i * y_domain * x_domain) 
                                + (y_index * x_domain) 
                                + x_index;
                                
                xy_sensor_density[grid_offset]++;
            }
        }
        
        // (You can copy/paste your XZ and YZ logic here in the exact same format,
        // XZ Planes
        for (int i=0; i<n_slice_xz; i++) {
            float y_ctr = y_out[i];
            float y_min = y_ctr - h_ave / 2;
            float y_max = y_ctr + h_ave / 2;

            if (y_min < y && y < y_max) {
                int x_index = std::max(0, (int)(x/dx));
                int z_index = std::max(0, (int)(z/dx));
                
                // Offset calculation for [Sensor][Z_Slice][Y][X]
                int grid_offset = (s * n_slice_xz * z_domain * x_domain) 
                                + (i * z_domain * x_domain) 
                                + (z_index * x_domain) 
                                + x_index;
                                
                xz_sensor_density[grid_offset]++;
            }
        }

        // just remember to include the (s * n_slice_xz * z_domain * x_domain) sensor offset!)
        // XZ Planes
        for (int i=0; i<n_slice_yz; i++) {
            float x_ctr = x_out[i];
            float x_min = x_ctr - h_ave / 2;
            float x_max = x_ctr + h_ave / 2;

            if (x_min < x && x < x_max) {
                int y_index = std::max(0, (int)(y/dx));
                int z_index = std::max(0, (int)(z/dx));
                
                // Offset calculation for [Sensor][Z_Slice][Y][X]
                int grid_offset = (s * n_slice_yz * z_domain * y_domain) 
                                + (i * z_domain * y_domain) 
                                + (z_index * y_domain) 
                                + y_index;
                                
                yz_sensor_density[grid_offset]++;
            }
        }
    }
}


void ParticleSensorDensity::output_sensor_density(Setting& setting) {
    const char* dir_out = setting.DIR_OUT;

    // ---------------------------------------------------------
    // STAGE 1: OUTPUT THE TIME CAPSULE
    // ---------------------------------------------------------
    if (setting.FLG_HARVEST_IDS == 1) {
        char hitFile[256];
        sprintf(hitFile, "%s/sensor_hit_ids.txt", dir_out);
        std::fstream fFile;
        fFile.open(hitFile, std::ios::out);
    
        if (!fFile.is_open()) {
            std::cout << "FATAL ERROR: Cannot open the output file" << std::endl;
            std::cout << "File: " << hitFile << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int s = 0; s < n_sensor; s++) {
            float x_s = setting.CTR_SENSOR_DENSITY[3*s];
            float y_s = setting.CTR_SENSOR_DENSITY[3*s+1];
            float z_s = setting.CTR_SENSOR_DENSITY[3*s+2];
            
            for (int src = 0; src < n_source; src++) {
                if (source_hit_sensor[src + (s * n_source)]) {
                    // Output format: Sensor_ID  X  Y  Z  Source_ID
                    fFile << s << " " << x_s << " " << y_s << " " << z_s << " " << src << std::endl;
                }
            }
        }
        fFile.close();
        return; 
    }

    // ---------------------------------------------------------
    // STAGE 2: OUTPUT THE CSV GRIDS
    // ---------------------------------------------------------
    if (setting.FLG_SENSOR_DENSITY == 1) {
        const int x_domain = setting.X_DOMAIN;
        const int y_domain = setting.Y_DOMAIN;
        const int z_domain = setting.Z_DOMAIN;
        const int n_slice_xy = setting.N_XY;
        const int n_slice_xz = setting.N_XZ;
        const int n_slice_yz = setting.N_YZ;

        std::vector<float> z_out(n_slice_xy);
        std::copy(setting.Z_OUT, setting.Z_OUT+n_slice_xy, z_out.data());

        std::vector<float> y_out(n_slice_xz);
        std::copy(setting.Y_OUT, setting.Y_OUT+n_slice_xz, y_out.data());
        
        std::vector<float> x_out(n_slice_yz);
        std::copy(setting.X_OUT, setting.X_OUT+n_slice_yz, x_out.data());

        for (int s = 0; s < n_sensor; s++) {
            std::cout << "\n--- Processing outputs for Sensor " << s << " ---" << std::endl;

            float x_s = setting.CTR_SENSOR_DENSITY[3*s];
            float y_s = setting.CTR_SENSOR_DENSITY[3*s+1];
            float z_s = setting.CTR_SENSOR_DENSITY[3*s+2];
            
            // ---------------- XY PLANES ----------------
            for (int i = 0; i < n_slice_xy; i++) {
                float zi_out = z_out[i];
                
                std::cout << "Output XY density (" << zi_out << "m) >>>>> ";

                std::fstream dFile;
                char densityFile[256];
                // We add the sensor ID to the filename so Python can identify them!
                sprintf(densityFile, "./%s/sensor_%d_%d_%d_xy_number_density_%dm.csv", dir_out, (int)x_s, (int)y_s, (int)z_s, (int)zi_out);
                dFile.open(densityFile, std::ios::out);
                
                if (!dFile.is_open()) {
                    std::cout << "cannot open the output file" << std::endl;
                    std::cout << "File: " << densityFile << std::endl;
                    exit(EXIT_FAILURE);
                }

                for (int j=0; j<y_domain; j++) {
                    for (int k=0; k<x_domain; k++) {
                        int grid_offset = (s * n_slice_xy * y_domain * x_domain) 
                                        + (i * y_domain * x_domain) 
                                        + (j * x_domain) 
                                        + k;
                                        
                        dFile << xy_sensor_density[grid_offset];
                        if (k < x_domain-1) dFile << ",";
                    }
                    dFile << std::endl;
                }
                dFile.close();
                std::cout << "Finish!!" << std::endl;
            }

            // ---------------- XZ PLANES ----------------
            for (int i = 0; i < n_slice_xz; i++) {
                float yi_out = y_out[i];
                
                std::cout << "Output XZ density (" << yi_out << "m) >>>>> ";

                std::fstream dFile;
                char densityFile[256];
                // We add the sensor ID to the filename so Python can identify them!
                sprintf(densityFile, "./%s/sensor_%d_%d_%d_xz_number_density_%dm.csv", dir_out, (int)x_s, (int)y_s, (int)z_s, (int)yi_out);
                dFile.open(densityFile, std::ios::out);

                if (!dFile.is_open()) {
                    std::cout << "cannot open the output file" << std::endl;
                    std::cout << "File: " << densityFile << std::endl;
                    exit(EXIT_FAILURE);
                }
                
                for (int j=0; j<z_domain; j++) {
                    for (int k=0; k<x_domain; k++) {
                        int grid_offset = (s * n_slice_xz * z_domain * x_domain) 
                                        + (i * z_domain * x_domain) 
                                        + (j * x_domain) 
                                        + k;
                                        
                        dFile << xz_sensor_density[grid_offset];
                        if (k < x_domain-1) dFile << ",";
                    }
                    dFile << std::endl;
                }
                dFile.close();
                std::cout << "Finish!!" << std::endl;
            }

            // ---------------- YZ PLANES ----------------
            for (int i = 0; i < n_slice_yz; i++) {
                float xi_out = x_out[i];
                
                std::cout << "Output YZ density (" << xi_out << "m) >>>>> ";

                std::fstream dFile;
                char densityFile[256];
                // We add the sensor ID to the filename so Python can identify them!
                sprintf(densityFile, "./%s/sensor_%d_%d_%d_yz_number_density_%dm.csv", dir_out, (int)x_s, (int)y_s, (int)z_s, (int)xi_out);
                dFile.open(densityFile, std::ios::out);

                if (!dFile.is_open()) {
                    std::cout << "cannot open the output file" << std::endl;
                    std::cout << "File: " << densityFile << std::endl;
                    exit(EXIT_FAILURE);
                }
                
                for (int j=0; j<z_domain; j++) {
                    for (int k=0; k<y_domain; k++) {
                        int grid_offset = (s * n_slice_yz * z_domain * y_domain) 
                                        + (i * z_domain * y_domain) 
                                        + (j * y_domain) 
                                        + k;
                                        
                        dFile << yz_sensor_density[grid_offset];
                        if (k < y_domain-1) dFile << ",";
                    }
                    dFile << std::endl;
                }
                dFile.close();
                std::cout << "Finish!!" << std::endl;
            }
        }
        std::cout << "--- [Sensor Density] Finished Outputting All 3D Plume Arrays ---" << std::endl;
    }
}


void ParticleSensorDensity::delete_sensor_density() {
    delete[] source_hit_sensor;
    if (xy_sensor_density) delete[] xy_sensor_density;
    if (xz_sensor_density) delete[] xz_sensor_density;
    if (yz_sensor_density) delete[] yz_sensor_density;
}