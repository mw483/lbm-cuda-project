# config_base.py

BASE_PARAMS = {
    "Define_user.h": {
        "init": {
            "DTDZ_LOW": 0.00,
            "DTDZ_HIGH": 0.00,
            "hf": 0.0
        },
        "flags": {
            "flg_buoyancy": 0,
            "flg_scalar": 0,
            "flg_particle": 1
        }
    },
    "post_processing": {
        "flags": {
            "FLG_NUM": 1,
            "FLG_DENSITY": 0,
            "FLG_PROFILE": 0,
            "FLG_FOOT": 0,
            "FLG_FLUX": 0,
            "FLG_RESID": 0,
            "FLG_BLEND_FOOT": 1,
        }
    }
}