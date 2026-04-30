import json
import os

def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_map_dimensions(map_path):
    if not os.path.exists(map_path):
        print(f"Error, {map_path} not found")
        return None
    
    with open(map_path, 'r') as f:
        # Read first line of the map file, split by tab or whitespace
        line = f.readline().strip().split()
        nx, ny = [int(line[0]), int(line[1])]
    return nx, ny

def generate_sh(params, nx, ny):
    p = params['runlbm.sh']
    map = params['map']
    # Construct the multi-line command string
    # We use a f-string with backslashes for the shell script formatting
    cnn_z = int(p['length_z'] / 2)
    content = f"""#!/bin/sh
rm run
ln -s ./src/run
export OMP_NUM_THREADS=1
echo "gpu_linux slots=1" > hostfile.txt

mpirun -hostfile hostfile.txt -np 1 ./run \\
        -Time                           {p['Time']} \\
        -time_coef                      {p['time_coef']} \\
        -NMPI                           1       1       1 \\
        -CNN                            {cnn_z}              2 \\
        -velocity_lbm                   {p['velocity_lbm']}     0.02 \\
        -gpu_per_node                   1 \\
        -halo_grid                      1 \\
        -CFout                          500     60000 \\
        -CFRfrg                         1       0       0 \\
        -restart                        0 \\
        -fstart                         0 \\
        -domain_min                     -0.08   -0.08   -0.08 \\
        -length                         {nx}     {ny}     {p['length_z']} \\
        -ncpu_div                       1       1       1       1 \\
        -flag_particle_generate         {p['flag_particle_generate']} \\
        -prestart                       0 \\
        -pout                           {p['pout']} \\
        -pstartstep                     0 \\
        -particle                       {p['max_particles']} \\
        -generate_step                  {p['generate_step']} \\
        | tee  -a  log_t2sub.txt
"""
    
    with open("runlbm.sh", "w") as f:
        f.write(content)
    
    # Make the script executable automatically
    os.chmod("runlbm.sh", 0o755)
    print(f"Successfully generated runlbm.sh for domain: {nx}x{ny}x{p['length_z']} and map: {map['path']}")

def generate_define_user(params):
    p_h = params['Define_user.h']
    p_init = p_h['init']
    p_out = p_h['output']
    p_flags = p_h['flags']

    # 1. Format the arrays for C++ syntax: [1, 2] -> "{1, 2}"
    kout_str = "{" + ", ".join(map(str, p_out['kout'])) + "}"
    jout_str = "{" + ", ".join(map(str, p_out['jout'])) + "}"
    iout_str = "{" + ", ".join(map(str, p_out['iout'])) + "}"

    # 2. Read the template
    template_path = "./src/Define_user.h.template"
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return

    with open(template_path, 'r') as f:
        content = f.read()

    # 3. Perform safe text replacements
    replacements = {
        "{{DTDZ_LOW}}": str(p_init['DTDZ_LOW']),
        "{{DTDZ_HIGH}}": str(p_init['DTDZ_HIGH']),
        "{{HF}}": str(p_init['hf']),
        
        "{{AVG_INT}}": str(p_out['average_interval']),
        "{{SKIP_TIME}}": str(p_out['skip_time']),
        "{{OUT_INT_INS}}": str(p_out['output_interval_ins']),
        "{{TIME_OUT_INI}}": str(p_out['time_output_ins_ini']),
        
        # Auto-calculate array lengths to prevent kernel crashes
        "{{NZ_OUT}}": str(len(p_out['kout'])),
        "{{NJ_OUT}}": str(len(p_out['jout'])),
        "{{NI_OUT}}": str(len(p_out['iout'])),
        
        "{{KOUT_ARRAY}}": kout_str,
        "{{JOUT_ARRAY}}": jout_str,
        "{{IOUT_ARRAY}}": iout_str,
        
        "{{FLG_BUOY}}": str(p_flags['flg_buoyancy']),
        "{{FLG_SCALAR}}": str(p_flags['flg_scalar']),
        "{{FLG_PARTICLE}}": str(p_flags['flg_particle'])
    }

    for key, val in replacements.items():
        content = content.replace(key, val)

    # 4. Write the final C++ header file
    output_path = "./src/Define_user.h"
    with open(output_path, 'w') as f:
        f.write(content)
        
    print(f"Define_user.h generated successfully. (nz_out auto-set to {len(p_out['kout'])})")

if __name__ == "__main__":
    data = load_params("params.json")

    m_path = data['map']['path']
    dims = get_map_dimensions(m_path)

    if dims:
        nx, ny = dims
        generate_sh(data, nx, ny)
        generate_define_user(data)
