import os
import stat
# Import the dynamically merged PARAMS from your new config package layout
from config.config import PARAMS

def generate_postprocess_sh():
    # Access the post-processing configuration dictionary block
    p = PARAMS["post_processing"]
    
    # 1. DYNAMIC FOLDER NAMING
    # Automatically read flags to build the output suffix (e.g., "_density_footprint")
    active_flags = []
    if p['flags']['FLG_DENSITY'] == 1: active_flags.append("density")
    if p['flags']['FLG_FOOT'] == 1: active_flags.append("footprint")
    if p['flags']['FLG_FLUX'] == 1: active_flags.append("flux")
    if p['flags']['FLG_PROFILE'] == 1: active_flags.append("profile")
    if p['flags']['FLG_RESID'] == 1: active_flags.append("resid")

    flag_suffix = "_".join(active_flags) if active_flags else "basic_output"
    folder_name = f"{p['timing']['FILE_START']}-{p['timing']['FILE_END']}_{flag_suffix}"

    # Construct the final DIR_OUT path using a base directory
    base_out = p['paths'].get('DIR_OUT', './Particle_PostProcess_Outputs/20260612_particle_cube_3072/sensor_8x8x8')
    dir_out = f"{base_out}/{folder_name}"

    # Helper function to format arrays for Bash
    def to_str(param_list):
        return " ".join(map(str, param_list))

    # 2. GENERATE THE SHELL SCRIPT
    # Notice we point directly to the global Engine path (Zone 1)
    content = f"""#!/bin/sh
export OMP_NUM_THREADS={p['execution']['OMP_NUM_THREADS']}

# The path to your compiled global C++ executable
ENGINE_PATH="/data/mikael/LBM_particle_test/Particle_PostProcessing_CPP/src/run_analysis"

# Automatically create the output directory before running
mkdir -p "{dir_out}"

# Execute the analysis
$ENGINE_PATH \\
    -NUM_RANK       {p['mpi_grid']['NUM_RANK']} \\
    -X_NUM_RANK     {p['mpi_grid']['X_NUM_RANK']} \\
    -Y_NUM_RANK     {p['mpi_grid']['Y_NUM_RANK']} \\
    -X_RANK         {p['mpi_grid']['X_RANK']} \\
    -Y_RANK         {p['mpi_grid']['Y_RANK']} \\
    -X_DOMAIN       {p['domain']['X_DOMAIN']} \\
    -Y_DOMAIN       {p['domain']['Y_DOMAIN']} \\
    -Z_DOMAIN       {p['domain']['Z_DOMAIN']} \\
    -dX             {p['domain']['dX']} \\
    -dT             {p['domain']['dT']} \\
    -FILE_START     {p['timing']['FILE_START']} \\
    -FILE_END       {p['timing']['FILE_END']} \\
    -POUT           {p['timing']['POUT']} \\
    -PGEN_STEP      {p['timing']['PGEN_STEP']} \\
    -NUM_GEN        {p['timing']['NUM_GEN']} \\
    -FLG_NUM        {p['flags']['FLG_NUM']} \\
    -FLG_DENSITY    {p['flags']['FLG_DENSITY']} \\
    -FLG_PROFILE    {p['flags']['FLG_PROFILE']} \\
    -FLG_FOOT       {p['flags']['FLG_FOOT']} \\
    -FLG_FLUX       {p['flags']['FLG_FLUX']} \\
    -FLG_RESID      {p['flags']['FLG_RESID']} \\
    -N_XY           {p['output_slices']['N_XY']} \\
    -Z_OUT          {to_str(p['output_slices']['Z_OUT'])} \\
    -N_XZ           {p['output_slices']['N_XZ']} \\
    -Y_OUT          {to_str(p['output_slices']['Y_OUT'])} \\
    -N_YZ           {p['output_slices']['N_YZ']} \\
    -X_OUT          {to_str(p['output_slices']['X_OUT'])} \\
    -H_AVE          {p['footprint_sensors']['H_AVE']} \\
    -N_SOURCE       {p['footprint_sensors']['N_SOURCE']} \\
    -ID_DIGIT       {p['footprint_sensors']['ID_DIGIT']} \\
    -N_SENSOR       {p['footprint_sensors']['N_SENSOR']} \\
    -CTR_SENSOR     {to_str(p['footprint_sensors']['CTR_SENSOR'])} \\
    -SIZE_SENSOR    {to_str(p['footprint_sensors']['SIZE_SENSOR'])} \\
    -N_FLUX         {p['flux_resid']['N_FLUX']} \\
    -Z_FLUX         {to_str(p['flux_resid']['Z_FLUX'])} \\
    -Z_RESID        {p['flux_resid']['Z_RESID']} \\
    -DIR_DATA       {p['paths']['DIR_DATA']} \\
    -DIR_OUT        {dir_out} \\
    -FNAME_MAP      {p['paths']['FNAME_MAP']} \\
    -FNAME_SOURCE   {p['paths']['FNAME_SOURCE']} \\
    | tee -a "{dir_out}/log_analysis.txt"
"""

    output_filename = "run_analysis.sh"
    with open(output_filename, "w", newline='\n') as f:
        f.write(content)
        
    # Make executable on Linux targets
    os.chmod(output_filename, os.stat(output_filename).st_mode | stat.S_IEXEC)
    print(f"Successfully generated {output_filename}")
    print(f"Data will be routed to: {dir_out}")

if __name__ == "__main__":
    generate_postprocess_sh()