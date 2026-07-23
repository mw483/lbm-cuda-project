#!/bin/sh
export OMP_NUM_THREADS=4

# The path to your compiled global C++ executable
ENGINE_PATH="./Particle_PostProcessing_CPP/src/run_analysis"

# Automatically create the output directory before running
mkdir -p "./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_blend_foot"

# Execute the analysis
$ENGINE_PATH \
    -NUM_RANK       4 \
    -X_NUM_RANK     4 \
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
    -NUM_GEN        3273600 \
    -FLG_NUM        1 \
    -FLG_DENSITY    0 \
    -FLG_PROFILE    0 \
    -FLG_FOOT       0 \
    -FLG_FLUX       0 \
    -FLG_RESID      0 \
    -FLG_BLEND_FOOT 1 \
    -N_XY           27 \
    -Z_OUT          2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 40 42 44 46 48 50 60 70 80 \
    -N_XZ           5 \
    -Y_OUT          250 252 254 256 258 \
    -N_YZ           1 \
    -X_OUT          1500 \
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
    -DIR_DATA       ./20260612_particle_cube_3072 \
    -DIR_OUT        ./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_blend_foot \
    -FNAME_MAP      ./map/map_02_full_roughness.dat \
    -FNAME_SOURCE   ./particle_position/pos_cube_3072.txt \
    | tee -a "./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_blend_foot/log_analysis.txt"
