# config_tsubame.py
import numpy as np

ENV_PARAMS = {
    "map": {
        "path": "./map/map_flat_16m_approach.dat",
        "physical_dx": 2.0
    },
    "runlbm.sh": {
        "Time": 180005,
        "time_coef": 0.01,
        "length_z": 160,
        "velocity_lbm": 2.0,
        "flag_particle_generate": 1,
        "pout": 100,
        "max_particles": 50000000,
        "generate_step": 100
    },
    "Define_user.h": {
        "init": {
            "DTDZ_LOW": 0.00,
            "DTDZ_HIGH": 0.00,
            "hf": 0.0
        },
        "output": {
            "average_interval": 600.0,
            "skip_time": 0.0,
            "output_interval_ins": 600.0,
            "time_output_ins_ini": 0.0,
            "kout": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 25, 28, 30, 32, 40, 48, 56, 64, 72, 80],
            "jout": [256],
            "iout": [3000]
        },
        "flags": {
            "flg_buoyancy": 0,
            "flg_scalar": 0,
            "flg_particle": 1
        }
    },
    "tsubame_mpi": {
        "scheduler": {
            "node_f": 4,
            "h_rt": "10:32:00",
            "job_name": "LBM_1_0"
        },
        "mpi_run": {
            "npernode": 1,
            "n_total": 4
        },
        "lbm_args": {
            "Time": 180005,
            "NMPI": [4, 1, 1],
            "gpu_per_node": 1,
            "ncpu_div": [1, 1, 1, 1],
            "CFout": [500, 60000],
            "CFRfrg": [1, 0, 1]
        }
    },
    "automatic_transfer": {
        "DEST_CSV": "./20260630_output_flat_16mapproach",
        "DEST_PAR": "./20260630_particle_flat_16mapproach"
    },
    "read_particle_box": {
        # Shifted upwind start position due to the extended approach fetch
        "pstart": 1200,
        "pnum": 5000,
        "num_g": [100, 50, 5],
        "point_g": [0.0, 0.0, 5.0],
        "vec_g": [1500.0, 1000.0, 256.0]
    },
    "post_processing": {
        "execution": {
            "OMP_NUM_THREADS": 4
        },
        "mpi_grid": {
            "NUM_RANK": 4,
            "X_NUM_RANK": 4,
            "Y_NUM_RANK": 1,
            "X_RANK": 0,
            "Y_RANK": 0
        },
        "domain": {
            "X_DOMAIN": 2048,  # Expanded domain size
            "Y_DOMAIN": 256,
            "Z_DOMAIN": 160,
            "dX": 2.0,
            "dT": 0.01
        },
        "timing": {
            "FILE_START": 4000,
            "FILE_END": 12000,
            "POUT": 100,
            "PGEN_STEP": 100,
            "NUM_GEN": 5000000
        },
        "output_slices": {
            "N_XY": 32,
            "Z_OUT": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 42, 44, 46, 48, 50, 60, 70, 80, 90, 100, 120, 140, 160],
            "N_XZ": 5,
            "Y_OUT": [250, 252, 254, 256, 258],
            "N_YZ": 1,
            "X_OUT": [1800]  # Shifted downstream output slice
        },
        "footprint_sensors": {
            "H_AVE": 2,
            "N_SOURCE": 5000,
            "ID_DIGIT": 3,
            "N_SENSOR": 3,
            # Adjust these coordinates based on where the city block sits downstream of the fetch
            "CTR_SENSOR": [1800, 96, 20, 
                           1800, 128, 20, 
                           1800, 160, 20],
            "SIZE_SENSOR": [40, 40, 8]
        },
        "flux_resid": {
            "N_FLUX": 9,
            "Z_FLUX": [8, 9, 10, 16, 17, 18, 32, 33, 34],
            "Z_RESID": 20
        },
        "paths": {
            "DIR_DATA": "/gs/bs/tga-lbmcity/mikael/LBM_particle_test/production_run_extended",
            "DIR_OUT": "/gs/bs/tga-lbmcity/mikael/LBM_particle_test/Particle_PostProcess_Outputs/sensor_40x40x8",
            "FNAME_MAP": "./map/map_02_full_roughness_extended.dat",
            "FNAME_SOURCE": "./particle_position/particle_position.txt"
        }
    }
}

# Shifted particle release locations relative to the new map dimensions
PARTICLE_SOURCES = [
    {
        "type": "uniform",
        "spacing_x": 8.0,
        "spacing_y": 8.0,
        "heights": [0.1],
        "velocity": [0.0, 0.0, 0.1],
        "group": 1,
        "x_start": 3073.0,
        "x_end": 3773.0,
        "y_start": 0.0,
        "y_end": 512.0,
        "y_padding": 8.0
    }
]

PARTICLE_OUTPUT = {
    "filename_pos": "pos_tsubame_flat.txt",
    "filename_num": "num_tsubame_flat.txt"
}