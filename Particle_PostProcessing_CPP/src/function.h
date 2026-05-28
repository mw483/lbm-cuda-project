#ifndef FUNCTION
#define FUNCTION

#include "stParticle.h"
#include "stSetting.h"

int 	sort_compare (const void* A, const void* B);
bool 	sort_index (const Particle_Index& A, const Particle_Index& B);
void	read_source (Particle_Index p_source[], Setting& setting);
#endif
