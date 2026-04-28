#ifndef SET_OBSTACLE_H_
#define SET_OBSTACLE_H_

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include "Math_Lib.h"
#include "Variables.h"


// obstacle
void Read_Obstacle_Data(MPIinfo *mpi, Domain *cdo, char *status);

// with levelset
void Set_Obstacle_Channel(MPIinfo *mpi, Domain *cdo, char *status, FLOAT *l_obs);
void Set_Obstacle_Sphere(MPIinfo *mpi, Domain *cdo, char *status, FLOAT *l_obs);

void Set_Obstacle_ground_surface ( 
	MPIinfo *mpi, Domain *cdo, char *status, FLOAT *l_obs);


#endif
