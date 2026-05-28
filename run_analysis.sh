#!/bin/sh
export OMP_NUM_THREADS=4

# The path to your compiled global C++ executable
ENGINE_PATH="/data/mikael/LBM_particle_test/Particle_PostProcessing_CPP/src"

# Automatically create the output directory before running
mkdir -p "./Particle_PostProcess_Outputs/20260527_particle_flat_3072/1200-1800_footprint"

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
    -FLG_NUM        1 \
    -FLG_DENSITY    0 \
    -FLG_PROFILE    0 \
    -FLG_FOOT       1 \
    -FLG_FLUX       0 \
    -FLG_RESID      0 \
    -N_XY           32 \
    -Z_OUT          2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 40 42 44 46 48 50 60 70 80 90 100 120 140 160 \
    -N_XZ           5 \
    -Y_OUT          124 126 128 130 132 \
    -N_YZ           1 \
    -X_OUT          500 \
    -H_AVE          2 \
    -N_SOURCE       3072 \
    -ID_DIGIT       3 \
    -N_SENSOR       3 \
    -CTR_SENSOR     600 96 10 600 128 10 600 160 10 \
    -SIZE_SENSOR    40 40 8 \
    -N_FLUX         9 \
    -Z_FLUX         8 9 10 16 17 18 32 33 34 \
    -Z_RESID        10 \
    -DIR_DATA       ./20260527_particle_flat_3072 \
    -DIR_OUT        ./Particle_PostProcess_Outputs/20260527_particle_flat_3072/1200-1800_footprint \
    -FNAME_MAP      ./map/map_01_flat_plane.dat \
    -FNAME_SOURCE   ./particle_position/particle_position.txt \
    | tee -a "./Particle_PostProcess_Outputs/20260527_particle_flat_3072/1200-1800_footprint/log_analysis.txt"
