# config.py

PARAMS = {
    "map": {
        "path": "./map/map_cubes_small.dat",
        "physical_dx": 2.0
    },
    "runlbm.sh": {
        "Time": 25005,
        "time_coef": 0.01,
        "length_z": 128,
        "velocity_lbm": 5.0,
        "flag_particle_generate": 1,
        "pout": 100,
        "max_particles": 20000000,
        "generate_step": 100
    },
    "Define_user.h": {
        "init": {
            "DTDZ_LOW": 0.01,
            "DTDZ_HIGH": 0.01,
            "hf": -0.1
        },
        "output": {
            "average_interval": 200.0,
            "skip_time": 0.0,
            "output_interval_ins": 200.0,
            "time_output_ins_ini": 0.0,
            "kout": [10, 20, 30],
            "jout": [64],
            "iout": [256]
        },
        "flags": {
            "flg_buoyancy": 0,
            "flg_scalar": 2,
            "flg_particle": 3
        }
    },
    "read_particle_box": {
        "pstart": 0,
        "pnum": 1000,
        "num_g": [50, 50, 5],
        "point_g": [0.0, 0.0, 5.0],
        "vec_g": [1000.0, 1000.0, 256.0]
    },
    # Append this inside the PARAMS dictionary in config.py
    "post_processing": {
        "execution": {
            "OMP_NUM_THREADS": 4
        },
        "mpi_grid": {
            "NUM_RANK": 1,
            "X_NUM_RANK": 1,
            "Y_NUM_RANK": 1,
            "X_RANK": 512,
            "Y_RANK": 128
        },
        "domain": {
            # Note: You could dynamically pull dX from PARAMS["map"]["physical_dx"] in the generator
            "X_DOMAIN": 512,
            "Y_DOMAIN": 128,
            "Z_DOMAIN": 80,
            "dX": 2.0,
            "dT": 0.01
        },
        "timing": {
            "FILE_START": 879,
            "FILE_END": 1279,
            "POUT": 100,
            "PGEN_STEP": 100,
            "NUM_GEN": 921600
        },
        "flags": {
            "FLG_NUM": 1,
            "FLG_DENSITY": 1,
            "FLG_PROFILE": 0,
            "FLG_FOOT": 0,
            "FLG_FLUX": 0,
            "FLG_RESID": 0
        },
        "output_slices": {
            "N_XY": 32,
            "Z_OUT": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 42, 44, 46, 48, 50, 60, 70, 80, 90, 100, 120, 140, 160],
            "N_XZ": 5,
            "Y_OUT": [124, 126, 128, 130, 132],
            "N_YZ": 1,
            "X_OUT": [500]
        },
        "footprint_sensors": {
            "H_AVE": 2,
            "N_SOURCE": 1536,
            "ID_DIGIT": 3,
            "N_SENSOR": 3,
            # Flattened [X, Y, Z] list for all sensors
            "CTR_SENSOR": [770, 88, 24, 
                           770, 128, 24, 
                           770, 168, 24],
            "SIZE_SENSOR": [40, 40, 8]
        },
        "flux_resid": {
            "N_FLUX": 9,
            "Z_FLUX": [8, 9, 10, 16, 17, 18, 32, 33, 34],
            "Z_RESID": 10
        },
        "paths": {
            "DIR_DATA": "../20260310_particle_nothermal_particle",
            "DIR_OUT": "../Output_20260310_particle_nothermal_particle/879-1279_density_footprint_flux",
            "FNAME_MAP": "../map/map_02_full_roughness.dat",
            "FNAME_SOURCE": "../particle_position_sourcearea/particle_position_sourcearea_groundonly_sparse.txt"
        }
    }
}

PARTICLE_SOURCES = [
    # { # Comment or uncomment each block to set the desired type
    #     "type": "uniform",
    #     "spacing_x": 16.0,
    #     "spacing_y": 16.0,
    #     "heights": [0.1],
    #     "velocity": [0.0, 0.0, 0.1],
    #     "group": 1,
    #     "x_max_ratio": 5,
    #     "y_padding": 16.0
    # },
    # {
    #     "type": "point",
    #     "num_particles": 100,
    #     "coords": [100.0, 200.0, 0.01],
    #     "velocity": [10.0, 0.0, 0.1], 
    #     "group": 1
    # },
    # {
    #     "type": "line",
    #     "num_particles": 500,
    #     "start": [10.0, 10.0, 0.1],
    #     "end": [100.0, 10.0, 0.1],
    #     "velocity": [0.0, 5.0, 0.1],
    #     "group": 1
    # },
    { # For mobile source modeling
        "type": "waypoint",
        "geometry": "line",   # line vs point
        "mode": "pingpong",
        "start_pos": [40.0, 120.0, 0.1],  # Start of the line
        "end_pos": [50.0, 120.0, 0.1],    # End of the line
        "num_points": 5,                 # 5 physical locations on the line
        "particles_per_point": 10,       # 10 particles spawned at each location
        "waypoints": [
            (0.0, 0.0, 0.0, 0.0),        # Relative offsets from original particle position as waypoints
            (45.0, 80.0, 0.0, 0.0),
            (90.0, 80.0, -50.0, 0.0),
            (135.0, 160.0, -50.0, 0.0)
        ],
        "velocity": [0.0, 0.0, 0.1],
        "group": 1
    }
]

PARTICLE_OUTPUT = {
    "filename_pos": "particle_position.txt",
    "filename_num": "particle_number.txt"
}