t2sub \
 -N ondNode  \
 -q G \
 -r y \
 -W group_list=t2g-kaken-Cont \
 -l walltime=16:00:00 \
 -l select=12:mpiprocs=3:ncpus=3:mem=21gb:gpus=3  \
 -l place=scatter ./sleep.sh

