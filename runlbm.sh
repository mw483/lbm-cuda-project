#!/bin/sh
rm run
ln -s ./src/run
export OMP_NUM_THREADS=1
echo "gpu_linux slots=1" > hostfile.txt

mpirun -hostfile hostfile.txt -np 1 ./run \
        -Time                           125005 \
        -time_coef                      0.01 \
        -NMPI                           1       1       1 \
        -CNN                            64              2 \
        -velocity_lbm                   2.0     0.02 \
        -gpu_per_node                   1 \
        -halo_grid                      1 \
        -CFout                          500     60000 \
        -CFRfrg                         1       0       0 \
        -restart                        0 \
        -fstart                         0 \
        -domain_min                     -0.08   -0.08   -0.08 \
        -length                         640.0     320.0     128 \
        -ncpu_div                       1       1       1       1 \
        -flag_particle_generate         0 \
        -prestart                       0 \
        -pout                           100 \
        -pstartstep                     0 \
        -particle                       20000000 \
        -generate_step                  100 \
        | tee  -a  log_t2sub.txt
