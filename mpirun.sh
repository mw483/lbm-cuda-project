#!/bin/sh
#$ -cwd
#$ -l node_f=4
#$ -l h_rt=10:32:00
#$ -N LBM_1_0
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
mpirun -x LD_LIBRARY_PATH -npernode 1 -n 4 ./run \
    -Time                   180005 \
    -time_coef              0.01 \
    -NMPI                   4 1 1 \
    -CNN                    80 2 \
    -velocity_lbm           2.0 0.02 \
    -gpu_per_node           1 \
    -halo_grid              1 \
    -CFout                  500 60000 \
    -CFRfrg                 1 0 1 \
    -restart                0 \
    -fstart                 0 \
    -domain_min             -0.08 -0.08 -0.08 \
    -length                 1024.0 256.0 160 \
    -ncpu_div               1 1 1 1 \
    -flag_particle_generate 1 \
    -prestart               0 \
    -pout                   100 \
    -pstartstep             0 \
    -particle               20000000 \
    -generate_step          100 \
    | tee -a log_t2sub.txt
