#ifndef SENSOR_DENSITY_H
#define SENSOR_DENSITY_H

#include "stSetting.h"

class ParticleSensorDensity {
private:
    int n_source;
    int n_sensor;
    
    // The Time Capsule: Flattened 1D boolean array [n_sensor * n_source]
    bool* source_hit_sensor; 

    // The Density Grids: Flattened 1D arrays for all sensors and all slices combined
    int* xy_sensor_density;
    int* xz_sensor_density;
    int* yz_sensor_density;

public:
    void allocate_and_load(Setting& setting);
    void harvest_ids(float x, float y, float z, int id, Setting& setting);
    void cal_sensor_density(float x, float y, float z, int id, Setting& setting);
    void output_sensor_density(Setting& setting);
    void delete_sensor_density();
};

#endif