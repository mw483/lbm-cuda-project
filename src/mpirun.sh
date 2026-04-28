#!/bin/sh
cd ${PBS_O_WORKDIR}


# path : cuda, mpi
source env.sh


rm run
ln -s ./src/run


mpirun -np	900	-hostfile $PBS_NODEFILE ./run \
	-Time				450000  						\
	-time_coef			0.002							\
	-NMPI				60	15	1					\
	-CNN				500		2					\
	-velocity_lbm			1.0	0.04						\
	-gpu_per_node			3							\
	-halo_grid			1							\
	-CFout 				500	10000						\
	-CFRfrg 			1	0	1 					\
	-restart			1 							\
	-fstart				0							\
	-domain_min			-0.08	-0.08	-0.08 					\
	-length				9600	2400	500	 				\
	-ncpu_div 			1	1	1	1				\
        -flag_particle_generate         0                                               	\
	-prestart			0							\
	-pout				3000000							\
	-pstartstep			0	   						\
	-particle			5000000							\
	-generate_step			3000000							\
	| tee  -a  log_t2sub.txt
