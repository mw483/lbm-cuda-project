#!/bin/sh

source tsubame_gcc64-1.4.2.sh


mpirun -np 36  -hostfile hostfile.txt ./scrach_copy.sh
#mpirun -np 36  -hostfile hostfile.txt hostname
