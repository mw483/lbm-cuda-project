#!/bin/sh
export OMP_NUM_THREADS=4

# The path to your compiled global C++ executable
ENGINE_PATH="./Particle_PostProcessing_CPP/src/run_analysis"

# Automatically create the output directory before running
mkdir -p "./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_sensor_density"

# Execute the analysis
$ENGINE_PATH \
    -NUM_RANK       1 \
    -X_NUM_RANK     1 \
    -Y_NUM_RANK     1 \
    -X_RANK         512 \
    -Y_RANK         128 \
    -X_DOMAIN       512 \
    -Y_DOMAIN       128 \
    -Z_DOMAIN       80 \
    -dX             2.0 \
    -dT             0.01 \
    -FILE_START     1200 \
    -FILE_END       1800 \
    -POUT           100 \
    -PGEN_STEP      100 \
    -NUM_GEN        1843200 \
    -FLG_NUM        0 \
    -FLG_DENSITY    0 \
    -FLG_PROFILE    0 \
    -FLG_FOOT       0 \
    -FLG_FLUX       0 \
    -FLG_RESID      0 \
    -FLG_BLEND_FOOT 0 \
    -FLG_HARVEST_IDS 0 \
    -FLG_SENSOR_DENSITY 1 \
    -N_XY           54 \
    -Z_OUT          52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140 142 144 146 148 150 152 154 156 158 \
    -N_XZ           1 \
    -Y_OUT          128 \
    -N_YZ           1 \
    -X_OUT          600 \
    -H_AVE          2 \
    -N_SOURCE       3072 \
    -ID_DIGIT       3 \
    -N_SENSOR       33 \
    -CTR_SENSOR     600 0 20 600 8 20 600 16 20 600 24 20 600 32 20 600 40 20 600 48 20 600 56 20 600 64 20 600 72 20 600 80 20 600 88 20 600 96 20 600 104 20 600 112 20 600 120 20 600 128 20 600 136 20 600 144 20 600 152 20 600 160 20 600 168 20 600 176 20 600 184 20 600 192 20 600 200 20 600 208 20 600 216 20 600 224 20 600 232 20 600 240 20 600 248 20 600 256 20 \
    -SIZE_SENSOR    8 8 8 \
    -N_FLUX         9 \
    -Z_FLUX         8 9 10 16 17 18 32 33 34 \
    -Z_RESID        10 \
    -CTR_SENSOR_BLEND 600 128 30 \
    -SIZE_SENSOR_BLEND 40 40 8 \
    -Z_BLEND        20 \
    -N_SENSOR_DENSITY       33 \
    -CTR_SENSOR_DENSITY     600 0 90 600 8 90 600 16 90 600 24 90 600 32 90 600 40 90 600 48 90 600 56 90 600 64 90 600 72 90 600 80 90 600 88 90 600 96 90 600 104 90 600 112 90 600 120 90 600 128 90 600 136 90 600 144 90 600 152 90 600 160 90 600 168 90 600 176 90 600 184 90 600 192 90 600 200 90 600 208 90 600 216 90 600 224 90 600 232 90 600 240 90 600 248 90 600 256 90 \
    -SIZE_SENSOR_DENSITY    40 40 8 \
    -DIR_DATA       ./20260612_particle_cube_3072 \
    -DIR_OUT        ./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_sensor_density \
    -FNAME_MAP      ./map/map_02_full_roughness.dat \
    -FNAME_SOURCE   ./particle_position/pos_cube_3072.txt \
    | tee -a "./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_sensor_density/log_analysis.txt"
