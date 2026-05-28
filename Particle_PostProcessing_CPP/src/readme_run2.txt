@echo off
REM Test script to isolate the crash
REM Set environment variable for OpenMP threads
set OMP_NUM_THREADS=12

main.exe ^
-NUM_RANK 1 ^
    -X_NUM_RANK 1 ^
    -Y_NUM_RANK 1 ^
    -X_RANK 416 ^
    -Y_RANK 192 ^
    -X_DOMAIN 416 ^
    -Y_DOMAIN 192 ^
    -Z_DOMAIN 160 ^
    -dX 2.0 ^
    -dT 0.01 ^
    -FILE_START 0 ^
    -FILE_END 2400 ^
    -POUT 100 ^
    -PGEN_STEP 1 ^
    -NUM_GEN 2400 ^
    -FLG_NUM 1 ^
    -FLG_DENSITY 0 ^
    -FLG_PROFILE 0 ^
    -FLG_FOOT 0 ^
    -FLG_FLUX 0 ^
    -FLG_RESID 0 ^
    -N_XY 62 ^
    -Z_OUT 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 ^
    -N_XZ 2 ^
    -Y_OUT 192 193 ^
    -N_YZ 1 ^
    -X_OUT 192 ^
    -H_AVE 1 ^
    -N_SOURCE 150 ^
    -ID_DIGIT 2 ^
    -N_SENSOR 1 ^
    -CTR_SENSOR 40 40 10 ^
    -SIZE_SENSOR 5 5 2 ^
    -N_FLUX 62 ^
    -Z_FLUX 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 ^
    -Z_RESID 10 ^
    -DIR_DATA ../20251028_particle_AIJ_CaseH_832x384 ^
    -DIR_OUT ../2200-2400_Output_20251028_particle_AIJ_CaseH_832x384 ^
    -FNAME_MAP ../map/c_AIJ_caseH_832x384 ^
    -FNAME_SOURCE ../particle_position_AIJ/AIJ_particle_position_CaseH_832x384.txt
