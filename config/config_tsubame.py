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
        "max_particles": 2000000,
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
            "jout": [128],
            "iout": [1500]
        },
        "flags": {
            "flg_buoyancy": 0,
            "flg_scalar": 0,
            "flg_particle": 1
        }
    },
    "tsubame_mpi": {
        "scheduler": {
            "node_f": 4,          # Change from 4 to 1 (You only need 1 node for 4 ranks)
            "h_rt": "10:32:00",
            "job_name": "LBM_1_0"
        },
        "mpi_run": {
            "npernode": 4,        # Change from 1 to 4 (Pack 4 ranks on the node)
            "n_total": 4          # Keep total ranks at 4
        },
        "lbm_args": {
            "Time": 180005,
            "NMPI": [4, 1, 1],
            "gpu_per_node": 1,    # Change from 1 to 4 (Use all 4 GPUs on that single node)
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
            "FILE_START": 1200,
            "FILE_END": 1800,
            "POUT": 100,
            "PGEN_STEP": 100,
            "NUM_GEN": 3273600
        },
        "output_slices": {
            "N_XY": 32,
            "Z_OUT": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 42, 44, 46, 48, 50, 60, 70, 80, 90, 100, 120, 140, 160],
            "N_XZ": 5,
            "Y_OUT": [250, 252, 254, 256, 258],
            "N_YZ": 1,
            "X_OUT": [1500]  # Shifted downstream output slice
        },
        "footprint_sensors": {
            "H_AVE": 2,
            "N_SOURCE": 5456,
            "ID_DIGIT": 3,
            "N_SENSOR": 132,
            # Adjust these coordinates based on where the city block sits downstream of the fetch
            "CTR_SENSOR": [3672, 0, 10, 
                           3672, 8, 10, 
                           3672, 16, 10,
                           3672, 24, 10, 
                           3672, 32, 10, 
                           3672, 40, 10,
                           3672, 48, 10, 
                           3672, 56, 10, 
                           3672, 64, 10,
                           3672, 72, 10, 
                           3672, 80, 10, 
                           3672, 88, 10,
                           3672, 96, 10, 
                           3672, 104, 10, 
                           3672, 112, 10,
                           3672, 120, 10, 
                           3672, 128, 10, 
                           3672, 136, 10,
                           3672, 144, 10, 
                           3672, 152, 10, 
                           3672, 160, 10,
                           3672, 168, 10, 
                           3672, 176, 10, 
                           3672, 184, 10,
                           3672, 192, 10,
                           3672, 200, 10,
                           3672, 208, 10,
                           3672, 216, 10,
                           3672, 224, 10,
                           3672, 232, 10,
                           3672, 240, 10,
                           3672, 248, 10,
                           3672, 256, 10,
                           3672, 0, 20, 
                           3672, 8, 20, 
                           3672, 16, 20,
                           3672, 24, 20, 
                           3672, 32, 20, 
                           3672, 40, 20,
                           3672, 48, 20, 
                           3672, 56, 20, 
                           3672, 64, 20,
                           3672, 72, 20, 
                           3672, 80, 20, 
                           3672, 88, 20,
                           3672, 96, 20, 
                           3672, 104, 20, 
                           3672, 112, 20,
                           3672, 120, 20, 
                           3672, 128, 20, 
                           3672, 136, 20,
                           3672, 144, 20, 
                           3672, 152, 20, 
                           3672, 160, 20,
                           3672, 168, 20, 
                           3672, 176, 20, 
                           3672, 184, 20,
                           3672, 192, 20,
                           3672, 200, 20,
                           3672, 208, 20,
                           3672, 216, 20,
                           3672, 224, 20,
                           3672, 232, 20,
                           3672, 240, 20,
                           3672, 248, 20,
                           3672, 256, 20,
                           3672, 0, 30, 
                           3672, 8, 30, 
                           3672, 16, 30,
                           3672, 24, 30, 
                           3672, 32, 30, 
                           3672, 40, 30,
                           3672, 48, 30, 
                           3672, 56, 30, 
                           3672, 64, 30,
                           3672, 72, 30, 
                           3672, 80, 30, 
                           3672, 88, 30,
                           3672, 96, 30, 
                           3672, 104, 30, 
                           3672, 112, 30,
                           3672, 120, 30, 
                           3672, 128, 30, 
                           3672, 136, 30,
                           3672, 144, 30, 
                           3672, 152, 30, 
                           3672, 160, 30,
                           3672, 168, 30, 
                           3672, 176, 30, 
                           3672, 184, 30,
                           3672, 192, 30,
                           3672, 200, 30,
                           3672, 208, 30,
                           3672, 216, 30,
                           3672, 224, 30,
                           3672, 232, 30,
                           3672, 240, 30,
                           3672, 248, 30,
                           3672, 256, 30,
                           3672, 0, 40, 
                           3672, 8, 40, 
                           3672, 16, 40,
                           3672, 24, 40, 
                           3672, 32, 40, 
                           3672, 40, 40,
                           3672, 48, 40, 
                           3672, 56, 40, 
                           3672, 64, 40,
                           3672, 72, 40, 
                           3672, 80, 40, 
                           3672, 88, 40,
                           3672, 96, 40, 
                           3672, 104, 40, 
                           3672, 112, 40,
                           3672, 120, 40, 
                           3672, 128, 40, 
                           3672, 136, 40,
                           3672, 144, 40, 
                           3672, 152, 40, 
                           3672, 160, 40,
                           3672, 168, 40, 
                           3672, 176, 40, 
                           3672, 184, 40,
                           3672, 192, 40,
                           3672, 200, 40,
                           3672, 208, 40,
                           3672, 216, 40,
                           3672, 224, 40,
                           3672, 232, 40,
                           3672, 240, 40,
                           3672, 248, 40,
                           3672, 256, 40,
                           ],
            "SIZE_SENSOR": [8, 8, 8]
        },
        "flux_resid": {
            "N_FLUX": 9,
            "Z_FLUX": [8, 9, 10, 16, 17, 18, 32, 33, 34],
            "Z_RESID": 20
        },
        "paths": {
            "DIR_DATA": "./20260630_particle_flat_16mapproach",
            "DIR_OUT": "./Particle_PostProcess_Outputs/sensor_8x8x8",
            "FNAME_MAP": "./map/map_flat_16m_approach.dat",
            "FNAME_SOURCE": "./particle_position/pos_tsubame_flat.txt"
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