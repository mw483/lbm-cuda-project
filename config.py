# config.py

PARAMS = {
    "map": {
        "path": "./map/takamatsu_single_building.dat",
        "physical_dx": 2.0
    },
    "runlbm.sh": {
        "Time": 125005,
        "time_coef": 0.01,
        "length_z": 128,
        "velocity_lbm": 10.0,
        "flag_particle_generate": 0,
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
            "average_interval": 300.0,
            "skip_time": 0.0,
            "output_interval_ins": 300.0,
            "time_output_ins_ini": 0.0,
            "kout": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 28, 32],
            "jout": [160],
            "iout": [80]
        },
        "flags": {
            "flg_buoyancy": 0,
            "flg_scalar": 0,
            "flg_particle": 1
        }
    },
    "read_particle_box": {
        "pstart": 1200,
        "pnum": 2000,
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
            "FILE_START": 1200,
            "FILE_END": 1800,
            "POUT": 100,
            "PGEN_STEP": 100,
            "NUM_GEN": 1843200
        },
        "flags": {
            "FLG_NUM": 1,
            "FLG_DENSITY": 0,
            "FLG_PROFILE": 0,
            "FLG_FOOT": 1,
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
            "N_SOURCE": 3072,
            "ID_DIGIT": 3,
            "N_SENSOR": 3,
            # Flattened [X, Y, Z] list for all sensors
            "CTR_SENSOR": [600, 96, 10, 
                           600, 128, 10, 
                           600, 160, 10],
            "SIZE_SENSOR": [40, 40, 8]
        },
        "flux_resid": {
            "N_FLUX": 9,
            "Z_FLUX": [8, 9, 10, 16, 17, 18, 32, 33, 34],
            "Z_RESID": 10
        },
        "paths": {
            "DIR_DATA": "./20260527_particle_flat_3072",
            "DIR_OUT": "./Output_20260527_particle_flat_3072/1200-1600_density_footprint_flux",
            "FNAME_MAP": "./map/map_01_flat_plane.dat",
            "FNAME_SOURCE": "./particle_position/particle_position.txt"
        }
    }
}

PARTICLE_SOURCES = [
    { # Comment or uncomment each block to set the desired type
        "type": "uniform",
        "spacing_x": 8.0,
        "spacing_y": 8.0,
        "heights": [0.1],
        "velocity": [0.0, 0.0, 0.1],
        "group": 1,
        "x_max_ratio": 1.33, #x_limit = domain_x / x_max_ratio
        "y_padding": 4.0
    },
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
    # { # For mobile source modeling
    #     "type": "waypoint",
    #     "geometry": "line",   # line vs point
    #     "mode": "pingpong",
    #     "start_pos": [40.0, 120.0, 0.1],  # Start of the line
    #     "end_pos": [50.0, 120.0, 0.1],    # End of the line
    #     "num_points": 5,                 # 5 physical locations on the line
    #     "particles_per_point": 10,       # 10 particles spawned at each location
    #     "waypoints": [
    #         (0.0, 0.0, 0.0, 0.0),        # Relative offsets from original particle position as waypoints
    #         (45.0, 80.0, 0.0, 0.0),
    #         (90.0, 80.0, -50.0, 0.0),
    #         (135.0, 160.0, -50.0, 0.0)
    #     ],
    #     "velocity": [0.0, 0.0, 0.1],
    #     "group": 1
    # }
]

PARTICLE_OUTPUT = {
    "filename_pos": "pos_flat_3072.txt",
    "filename_num": "num_flat_3072.txt"
}