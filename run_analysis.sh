#!/bin/sh
export OMP_NUM_THREADS=4

# The path to your compiled global C++ executable
ENGINE_PATH="./Particle_PostProcessing_CPP/src/run_analysis"

# Automatically create the output directory before running
<<<<<<< HEAD
mkdir -p "./Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_40x40x8/1200-1800_blend_foot"
=======
mkdir -p "./Particle_PostProcess_Outputs/sensor_8x8x8/1200-1800_footprint"
>>>>>>> d87cd3cb95bda552f7b64f126d45f339cbc433c4

# Execute the analysis
$ENGINE_PATH \
    -NUM_RANK       4 \
    -X_NUM_RANK     4 \
    -Y_NUM_RANK     1 \
<<<<<<< HEAD
    -X_RANK         512 \
    -Y_RANK         128 \
    -X_DOMAIN       512 \
    -Y_DOMAIN       128 \
    -Z_DOMAIN       80 \
=======
    -X_RANK         0 \
    -Y_RANK         0 \
    -X_DOMAIN       2048 \
    -Y_DOMAIN       256 \
    -Z_DOMAIN       160 \
>>>>>>> d87cd3cb95bda552f7b64f126d45f339cbc433c4
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
<<<<<<< HEAD
    -FLG_BLEND_FOOT 1 \
    -N_XY           27 \
    -Z_OUT          2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 40 42 44 46 48 50 60 70 80 \
=======
    -N_XY           32 \
    -Z_OUT          2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 40 42 44 46 48 50 60 70 80 90 100 120 140 160 \
>>>>>>> d87cd3cb95bda552f7b64f126d45f339cbc433c4
    -N_XZ           5 \
    -Y_OUT          250 252 254 256 258 \
    -N_YZ           1 \
    -X_OUT          1500 \
    -H_AVE          2 \
<<<<<<< HEAD
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
=======
    -N_SOURCE       5456 \
    -ID_DIGIT       3 \
    -N_SENSOR       132 \
    -CTR_SENSOR     3672 0 10 3672 8 10 3672 16 10 3672 24 10 3672 32 10 3672 40 10 3672 48 10 3672 56 10 3672 64 10 3672 72 10 3672 80 10 3672 88 10 3672 96 10 3672 104 10 3672 112 10 3672 120 10 3672 128 10 3672 136 10 3672 144 10 3672 152 10 3672 160 10 3672 168 10 3672 176 10 3672 184 10 3672 192 10 3672 200 10 3672 208 10 3672 216 10 3672 224 10 3672 232 10 3672 240 10 3672 248 10 3672 256 10 3672 0 20 3672 8 20 3672 16 20 3672 24 20 3672 32 20 3672 40 20 3672 48 20 3672 56 20 3672 64 20 3672 72 20 3672 80 20 3672 88 20 3672 96 20 3672 104 20 3672 112 20 3672 120 20 3672 128 20 3672 136 20 3672 144 20 3672 152 20 3672 160 20 3672 168 20 3672 176 20 3672 184 20 3672 192 20 3672 200 20 3672 208 20 3672 216 20 3672 224 20 3672 232 20 3672 240 20 3672 248 20 3672 256 20 3672 0 30 3672 8 30 3672 16 30 3672 24 30 3672 32 30 3672 40 30 3672 48 30 3672 56 30 3672 64 30 3672 72 30 3672 80 30 3672 88 30 3672 96 30 3672 104 30 3672 112 30 3672 120 30 3672 128 30 3672 136 30 3672 144 30 3672 152 30 3672 160 30 3672 168 30 3672 176 30 3672 184 30 3672 192 30 3672 200 30 3672 208 30 3672 216 30 3672 224 30 3672 232 30 3672 240 30 3672 248 30 3672 256 30 3672 0 40 3672 8 40 3672 16 40 3672 24 40 3672 32 40 3672 40 40 3672 48 40 3672 56 40 3672 64 40 3672 72 40 3672 80 40 3672 88 40 3672 96 40 3672 104 40 3672 112 40 3672 120 40 3672 128 40 3672 136 40 3672 144 40 3672 152 40 3672 160 40 3672 168 40 3672 176 40 3672 184 40 3672 192 40 3672 200 40 3672 208 40 3672 216 40 3672 224 40 3672 232 40 3672 240 40 3672 248 40 3672 256 40 \
    -SIZE_SENSOR    8 8 8 \
    -N_FLUX         9 \
    -Z_FLUX         8 9 10 16 17 18 32 33 34 \
    -Z_RESID        20 \
    -DIR_DATA       ./20260630_particle_flat_16mapproach \
    -DIR_OUT        ./Particle_PostProcess_Outputs/sensor_8x8x8/1200-1800_footprint \
    -FNAME_MAP      ./map/map_flat_16m_approach.dat \
    -FNAME_SOURCE   ./particle_position/pos_tsubame_flat.txt \
    | tee -a "./Particle_PostProcess_Outputs/sensor_8x8x8/1200-1800_footprint/log_analysis.txt"
>>>>>>> d87cd3cb95bda552f7b64f126d45f339cbc433c4
