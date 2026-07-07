#!/bin/sh
export OMP_NUM_THREADS=4

# The path to your compiled global C++ executable
ENGINE_PATH="/data/mikael/LBM_particle_test/Particle_PostProcessing_CPP/src/run_analysis"

# Automatically create the output directory before running
mkdir -p "./Particle_PostProcess_Outputs/20260703_particle_flat_shortroughness/sensor_8x8x8/1200-1800_footprint"

# Execute the analysis
$ENGINE_PATH \
    -NUM_RANK       1 \
    -X_NUM_RANK     1 \
    -Y_NUM_RANK     1 \
    -X_RANK         512 \
    -Y_RANK         128 \
    -X_DOMAIN       640 \
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
    -N_XY           27 \
    -Z_OUT          2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 40 42 44 46 48 50 60 70 80 \
    -N_XZ           5 \
    -Y_OUT          124 126 128 130 132 \
    -N_YZ           1 \
    -X_OUT          500 \
    -H_AVE          2 \
    -N_SOURCE       2816 \
    -ID_DIGIT       3 \
    -N_SENSOR       33 \
    -CTR_SENSOR     728 0 20 728 8 20 728 16 20 728 24 20 728 32 20 728 40 20 728 48 20 728 56 20 728 64 20 728 72 20 728 80 20 728 88 20 728 96 20 728 104 20 728 112 20 728 120 20 728 128 20 728 136 20 728 144 20 728 152 20 728 160 20 728 168 20 728 176 20 728 184 20 728 192 20 728 200 20 728 208 20 728 216 20 728 224 20 728 232 20 728 240 20 728 248 20 728 256 20 \
    -SIZE_SENSOR    8 8 8 \
    -N_FLUX         9 \
    -Z_FLUX         8 9 10 16 17 18 32 33 34 \
    -Z_RESID        10 \
    -DIR_DATA       ./20260703_particle_flat_shortroughness \
    -DIR_OUT        ./Particle_PostProcess_Outputs/20260703_particle_flat_shortroughness/sensor_8x8x8/1200-1800_footprint \
    -FNAME_MAP      ./map/map_flat_16m_shortroughness.dat \
    -FNAME_SOURCE   ./particle_position/pos_flat_3072_shortroughness.txt \
    | tee -a "./Particle_PostProcess_Outputs/20260703_particle_flat_shortroughness/sensor_8x8x8/1200-1800_footprint/log_analysis.txt"
