#!/bin/sh

qsub -g tga-lbmcity mpirun.sh

# If reserving a node, add  -ar {reservation id}, for example:
# qsub -g tga-lbmcity -ar 1408 mpirun.sh

# old command for old group
# qsub -g jh250051 mpirun.sh

#qsub mpirun.sh