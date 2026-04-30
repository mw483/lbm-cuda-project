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

if __name__ == "__main__":
    data = load_params("params.json")

    m_path = data['map']['path']
    dims = get_map_dimensions(m_path)

    if dims:
        nx, ny = dims
        generate_sh(data, nx, ny)
