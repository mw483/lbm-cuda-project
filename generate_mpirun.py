import os
import stat
from config import PARAMS

def generate_mpirun_sh():
    p_run = PARAMS["runlbm.sh"]
    p_map = PARAMS["map"]
    p_mpi = PARAMS["tsubame_mpi"]

    # Calculate grid size dynamically from the Map parser (assuming you have this logic)
    # For now, pulling from your provided script defaults:
    length_x, length_y, length_z = 1024, 256, 160

    # Helper function to format arrays
    def to_str(param_list):
        return " ".join(map(str, param_list))
    
    content = f"""#!/bin/sh
#$ -cwd
#$ -l node_f={p_mpi['scheduler']['node_f']}
#$ -l h_rt={p_mpi['scheduler']['h_rt']}
#$ -N {p_mpi['scheduler']['job_name']}
#$ -v LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/12.3.2/lib64:/apps/t4/rhel9/free/openmpi/5.0.2-gcc/lib:$LD_LIBRARY_PATH

rm -f run
ln -s ./src/run

ulimit -c unlimited
. /etc/profile.d/modules.sh
module purge
module load cuda/12.3.2
module load openmpi/5.0.2-gcc

export OMP_NUM_THREADS=4
cp $PE_HOSTFILE hostfile.txt

# Execute via MPI
mpirun -x LD_LIBRARY_PATH -npernode {p_mpi['mpi_run']['npernode']} -n {p_mpi['mpi_run']['n_total']} ./run \\
    -Time                   {p_mpi['lbm_args']['Time']} \\
    -time_coef              {p_run['time_coef']} \\
    -NMPI                   {to_str(p_mpi['lbm_args']['NMPI'])} \\
    -CNN                    80 2 \\
    -velocity_lbm           {p_run['velocity_lbm']} 0.02 \\
    -gpu_per_node           {p_mpi['lbm_args']['gpu_per_node']} \\
    -halo_grid              1 \\
    -CFout                  {to_str(p_mpi['lbm_args']['CFout'])} \\
    -CFRfrg                 {to_str(p_mpi['lbm_args']['CFRfrg'])} \\
    -restart                0 \\
    -fstart                 0 \\
    -domain_min             -0.08 -0.08 -0.08 \\
    -length                 {length_x} {length_y} {length_z} \\
    -ncpu_div               {to_str(p_mpi['lbm_args']['ncpu_div'])} \\
    -flag_particle_generate {p_run['flag_particle_generate']} \\
    -prestart               0 \\
    -pout                   {p_run['pout']} \\
    -pstartstep             0 \\
    -particle               {p_run['max_particles']} \\
    -generate_step          {p_run['generate_step']} \\
    | tee -a log_t2sub.txt
"""
    output_filename = "mpirun.sh"
    with open(output_filename, "w", newline='\n') as f:
        f.write(content) 
    
    os.chmod(output_filename, os.stat(output_filename).st_mode | stat.S_IEXEC)
    print(f"[TSUBAME] Successfully generated {output_filename}")

if __name__ == "__main__":
    generate_mpirun_sh()