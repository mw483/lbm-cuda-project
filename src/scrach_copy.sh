#!/bin/sh

rank=$OMPI_COMM_WORLD_RANK

mv /scr/init-rank${rank}.dat  ./init_profile/
#mv /scr/init-rank*.dat  ./init_profile/

